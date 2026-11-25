#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
default_repo="$HOME/projects/kv-ssd-sim"
if [[ -f "$script_dir/../scripts/analyze_nand.py" ]]; then
  default_repo="$script_dir/.."
fi
repo=${1:-"$default_repo"}
repo=$(git -C "$repo" rev-parse --show-toplevel)
commit=$(git -C "$repo" rev-parse HEAD)
test -f "$repo/scripts/analyze_nand.py" || { echo "Missing baseline code in $repo; pull main first." >&2; exit 1; }
printf 'Using checkout %s at %s\n' "$repo" "$commit"
image=kv-ssd-n15:baseline-v2
python3 - <<'PY'
import fcntl
import os

with open("/dev/kvm", "r+b", buffering=0) as kvm:
    if fcntl.ioctl(kvm.fileno(), 0xAE00, 0) != 12:
        raise RuntimeError("unexpected KVM API version")
    os.close(fcntl.ioctl(kvm.fileno(), 0xAE01, 0))
print("KVM CREATE_VM OK")
PY
docker info >/dev/null
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build -t "$image" - <<'IMAGE'
FROM ubuntu:26.04
RUN apt-get update && apt-get install -y --no-install-recommends e2fsprogs libstdc++6 libgomp1 libnuma1 libcurl4t64 util-linux procps ca-certificates
IMAGE
fi

runtime_root="$repo/.runtime/n15"
mkdir -p -- "$runtime_root"
bundle=${2:-"$runtime_root"}
if [[ $# -lt 2 && ! -f "$bundle/SHA256SUMS" ]]; then
  if [[ -f "$script_dir/SHA256SUMS" ]]; then
    bundle="$script_dir"
  else
    bundle="$HOME/n15-kv-baseline"
  fi
fi
if [[ ! -f "$bundle/SHA256SUMS" ]]; then
  echo "Missing SHA256SUMS. Supply the baseline archive directory as the second argument." >&2
  exit 1
fi
archive_sha=$(awk '$2 == "baseline.tar.gz" {print $1}' "$bundle/SHA256SUMS")
[[ "$archive_sha" =~ ^[[:xdigit:]]{64}$ ]] || { echo "Invalid baseline checksum in $bundle/SHA256SUMS" >&2; exit 1; }
work="$runtime_root/runtime-$archive_sha"
(
  flock -x 9
  if [[ ! -d "$work" ]]; then
    (cd "$bundle" && printf '%s  baseline.tar.gz\n' "$archive_sha" | sha256sum -c -)
    staging=$(mktemp -d "$runtime_root/.install-XXXXXXXX")
    trap 'rm -rf -- "$staging"' EXIT
    tar -xzf "$bundle/baseline.tar.gz" --no-same-owner -C "$staging" --wildcards \
      'miniconda3/*' 'projects/kv-ssd-sim/.venv/*' 'projects/kv-ssd-sim/.venv-aiperf/*' \
      'projects/kv-ssd-sim/.native/*' 'projects/kv-ssd-sim/.femu/*'
    mv -T -- "$staging" "$work"
  fi
  # Keep the selected version available when the external archive is offline.
  if [[ ! "$bundle/SHA256SUMS" -ef "$runtime_root/SHA256SUMS" ]]; then
    printf '%s  baseline.tar.gz\n' "$archive_sha" > "$runtime_root/SHA256SUMS.new"
    mv -fT -- "$runtime_root/SHA256SUMS.new" "$runtime_root/SHA256SUMS"
  fi
) 9>"$runtime_root/install.lock"
printf 'Using project runtime %s\n' "$work"
mkdir -p "$repo/outputs"
container_repo=/home/plusyu/projects/kv-ssd-sim
mounts=()
for asset in .venv .venv-aiperf .native .femu; do
  mkdir -p "$repo/$asset"
  mounts+=(-v "$work/projects/kv-ssd-sim/$asset:$container_repo/$asset")
done
docker image inspect --format '{{.Id}}' "$image" > "$repo/outputs/container-image-id.txt"
docker run --rm -i --device /dev/kvm --user "$(id -u):$(id -g)" \
  --group-add "$(stat -c %g /dev/kvm)" \
  -e XDG_CACHE_HOME=/tmp/kv-cache -e MPLCONFIGDIR=/tmp/kv-mpl -e HF_HOME=/tmp/kv-hf \
  -e USER=kvbench -e LOGNAME=kvbench \
  -e "CHECKOUT_COMMIT=$commit" -v "$repo:$container_repo" "${mounts[@]}" \
  -v "$work/miniconda3:/home/plusyu/miniconda3:ro" \
  -w /home/plusyu/projects/kv-ssd-sim "$image" bash -seu <<'GUEST_LAUNCHER'
set -o pipefail
.venv/bin/python -m kv_ssd_sim doctor | tee outputs/doctor.json
.venv-aiperf/bin/python -c 'import aiperf, matplotlib, tokenizers'
(cd evidence/ssd-e && sha256sum -c SHA256SUMS)
tag=$(date -u +%Y%m%d-%H%M%S)
printf '%s\n' "$CHECKOUT_COMMIT" > "outputs/checkout-$tag.txt"
probe=femu-n15-probe-$tag
trial=femu-n15-0504-$tag
.venv/bin/python scripts/run_femu.py --accel kvm --workload nand-probe --trial "$probe" \
  --guest-ram-mib 1024 --namespace-mib 48 --blocks-per-plane 32 --timeout 240
.venv/bin/python scripts/run_femu.py --accel kvm --workload aiperf --trial "$trial" \
  --natural-cache --block-capture stream --nand-stats \
  --input datasets/k3-c2/slice.json --hybrid-config configs/k3-c1-online.json \
  --guest-ram-mib 8192 --namespace-mib 3072 --blocks-per-plane 2048 \
  --max-model-len 12288 --num-blocks 22 --prefill-ms 10 --decode-ms 2 --timeout 2400
.venv-aiperf/bin/python scripts/verify_femu.py "outputs/$trial" > "outputs/$trial/verification-n15.json"
.venv-aiperf/bin/python scripts/analyze_k3_ssd.py "outputs/$trial" --output "outputs/$trial-analysis"
.venv/bin/python - "$trial" <<'PY'
import json
import sys
from pathlib import Path

trial = sys.argv[1]
def objects(path):
    return json.loads(Path(path).read_text())["runs"][0]["objects"]

old = objects("evidence/ssd-c2/summary.json")
new = objects(f"outputs/{trial}-analysis/summary.json")
for key in ("unique_keys", "logical_bytes", "key_content_sha256"):
    if old[key] != new[key]:
        raise SystemExit(f"0504 object mismatch for {key}: {new[key]} vs {old[key]}")
    print(key, new[key])
print("PASS: KVM NAND probe and 0504 baseline")
PY
GUEST_LAUNCHER
printf 'Results: %s/outputs\n' "$repo"
