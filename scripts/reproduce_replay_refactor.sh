#!/usr/bin/env bash
# Run selected 20316bf regression fixtures in the active n15 replay environment.
set -euo pipefail
if (( $# < 3 )); then
  echo "Usage: REPLAY_PYTHON=... bash $0 FIXTURES NEW_OUTPUT CASE [CASE ...]" >&2
  exit 2
fi
fixtures=$(realpath "$1")
stage=$(realpath -m "$2")
shift 2
cd "$(dirname "$0")/.."
python_bin=${REPLAY_PYTHON:-python3}
mkdir "$stage"
for sample in "$@"; do
  input="$fixtures/inputs/return-idle.json"
  reference="$fixtures/runs/$sample"
  output="$stage/$sample"
  flags=(--profile reference --clients 1 --num-blocks 11 --max-model-len 12288
         --segment-mib 1450 --staging-mib 96 --quota-mib 2840)
  unset PYTHONHASHSEED
  case "$sample" in
    warm)
      export PYTHONHASHSEED=0
      reference="$fixtures/warm/volume-1x"
      input="$fixtures/inputs/warm.json"
      flags=(--profile reference --volume-service 1x)
      ;;
    base-staged|matched-legacy) ;;
    tp8-pressure)
      flags=(--profile reference --clients 8 --num-blocks 11 --max-model-len 12288
             --segment-mib 182 --staging-mib 96 --quota-mib 355)
      ;;
    child-staged|child-legacy) input="$fixtures/inputs/return-with-child.json" ;;
    probe)
      mkdir "$output"
      "$python_bin" scripts/check_object_probe.py --output "$output/report.json" \
        --events "$output/events.jsonl" > "$stage/$sample.log" 2>&1
      cmp "$fixtures/controlled/report.json" "$output/report.json"
      cmp "$fixtures/controlled/events.jsonl" "$output/events.jsonl"
      echo "$sample passed"
      continue
      ;;
    *) echo "Unknown fixture: $sample" >&2; exit 2 ;;
  esac
  if [[ $sample == *-legacy ]]; then flags+=(--restore-path legacy); fi
  "$python_bin" scripts/run_object_replay.py "$input" "${flags[@]}" --output "$output" \
    > "$stage/$sample.log" 2>&1
  "$python_bin" scripts/verify_replay_equivalence.py "$reference" "$output" --source-root . \
    > "$stage/$sample-check.json"
  echo "$sample passed"
done
