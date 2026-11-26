#!/usr/bin/env bash
# Reproduce the first kvlab milestone on the pinned n15 checkout, in a fresh trial.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
study=$(realpath -m "${1:?Usage: bash scripts/reproduce_kvlab_n15.sh NEW_TRIAL_DIR}")
base_python=${K3_PYTHON:-/home/yzh/miniconda3/bin/python3.13}
plot_python=${K3_PLOT_PYTHON:-/home/yzh/miniconda3/envs/SAN/bin/python}
pkg="$root/.venv/lib/python3.13/site-packages"
mkdir -p "$(dirname "$study")"
mkdir "$study"
"$base_python" -m venv --without-pip "$study/py"
py="$study/py/bin/python"
ln -s "$pkg/mooncake/mooncake_master" "$study/py/bin/mooncake_master"
export PYTHONPATH="$root/src:$pkg" LD_LIBRARY_PATH="$(dirname "$(dirname "$base_python")")/lib"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=1
export VLLM_SSM_CONV_STATE_LAYOUT=SD VLLM_CPU_KVCACHE_SPACE=0 VLLM_HOST_IP=127.0.0.1
export CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo KV_SSD_USE_SYSTEM_LIBRARIES=1
export VLLM_CACHE_ROOT="$study/cache" KV_SSD_OBJECT_MASTER_DIR="$study/build"
unset PYTHONHASHSEED
cache="$root/.object-replay-build/kvlab-source"
source_file="$cache/replay-40d-oct14-337a03810637.jsonl"
"$py" scripts/fetch_kvlab.py --cache "$cache" --ca /etc/ssl/certs/ca-certificates.crt
archive="$root/.object-replay-build/sources/Mooncake-e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c.tar.gz"
if [[ ! -f "$archive" ]]; then
    archive="$study/Mooncake.tar.gz"
    curl -fL --retry 2 https://codeload.github.com/kvcache-ai/Mooncake/tar.gz/e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c -o "$archive"
fi
"$py" scripts/build_object_master.py --archive "$archive" --out "$study/build"
"$py" scripts/build_object_observer.py --archive "$archive" --out "$study/build"
observer="$study/build/observer.so"

# The comparison deliberately retains any mismatch in final memory residency.
tar -xzf evidence/object-replay-n15-20261123/inputs.tar.gz -C "$study" input/0504.json
fixture=femu-n15-0504-20261120-074735
tar -xzf "evidence/n15-kvm-20261120-074735/$fixture.tar.gz" -C "$study" \
    "$fixture/joint/server/events.jsonl" "$fixture/joint/server/summary.json"
"$py" -m kv_ssd_sim simulate "$study/input/0504.json" --restore-path legacy --output "$study/regression-0504"
if ! "$py" scripts/compare_object_replay.py --prediction "$study/regression-0504" \
    --server "$study/$fixture/joint/server" --retrospective --output "$study/regression.json"; then
    "$py" -c 'import json,sys; c=json.load(open(sys.argv[1]))["checks"]; sys.exit(not all(v for k,v in c.items() if k!="final_memory_keys"))' "$study/regression.json"
fi
"$py" -m kv_ssd_sim workload kvlab audit "$source_file" --output "$study/corpus"
"$py" -m kv_ssd_sim workload kvlab windows "$source_file" --audit "$study/corpus" --limit 12288 --output "$study/paired"
"$py" -m kv_ssd_sim workload kvlab windows "$source_file" --audit "$study/corpus" --limit 131072 --output "$study/reference"

cal="$study/paired/calibration/input.json"
held="$study/paired/holdout/input.json"
"$py" -m kv_ssd_sim simulate "$cal" --restore-path legacy --output "$study/pred-cal"
"$py" scripts/run_object_holdout.py "$cal" --prediction "$study/pred-cal" --observer "$observer" --output "$study/online-cal"
"$py" scripts/analyze_kvlab_native.py fit --native "$study/online-cal" --output "$study/calibration.json"
"$py" scripts/analyze_kvlab_native.py score --native "$study/online-cal" --prediction "$study/pred-cal" --output "$study/score-cal"
"$py" -m kv_ssd_sim simulate "$held" --restore-path legacy --output "$study/frozen-holdout" --service-profile "$study/calibration.json"
for i in 1 2 3; do
    "$py" scripts/run_object_holdout.py "$held" --prediction "$study/frozen-holdout" --observer "$observer" \
        --native-prefill-ms 10 --native-decode-ms 2 --output "$study/online-holdout-$i"
    "$py" scripts/analyze_kvlab_native.py score --native "$study/online-holdout-$i" \
        --prediction "$study/frozen-holdout" --output "$study/score-$i"
done
for set_name in paired reference; do
    "$py" scripts/run_kvlab_matrix.py "$study/$set_name" --profile reference-20261124 --output "$study/demand-$set_name"
done

"$py" -m pytest -q -m 'not integration' > "$study/tests-final.log"
export KV_SSD_RUN_NATIVE_TESTS=1
KV_SSD_OBSERVED_DSO="$pkg/mooncake/_store.cpython-313-x86_64-linux-gnu.so" \
    KV_SSD_OBSERVER_LOG="$study/controlled-read.tsv" LD_PRELOAD="$observer" \
    "$py" -m pytest -q tests/test_integration.py::test_disk_restore_releases_native_scheduler \
    --basetemp "$study/positive-fixture" > "$study/controlled-read.log"
"$py" -m pytest -q -m integration -k 'not test_disk_restore_releases_native_scheduler' > "$study/integration.log"
PYTHONPATH="$root/src" MPLCONFIGDIR="$study/cache/matplotlib" \
    "$plot_python" scripts/summarize_kvlab.py "$study" --output "$study/final-results"
"$py" scripts/archive_kvlab.py "$study" --output "$study/evidence"
(cd "$study/evidence" && sha256sum -c SHA256SUMS)
