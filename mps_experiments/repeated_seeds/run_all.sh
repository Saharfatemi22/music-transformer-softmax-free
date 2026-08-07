#!/bin/bash
# mps_experiments/repeated_seeds/run_all.sh
# ────────────────────────────────────────────────────────────────────────
# Runs the 15 repeated-seed training jobs (5 configs x 3 seeds) sequentially
# on MPS: 50 epochs, 1000 train chunks, 200 val chunks, same model size as
# every other MPS experiment in this project.
#
# Does NOT touch mps_experiments/checkpoints/ or mps_experiments/standardization_ablation/
# — all output goes under mps_experiments/repeated_seeds/.
#
# Deliberately does NOT use `set -e`: a single run failing should not skip
# the remaining 14. Each run's own exit code is echoed to the progress log.

cd "$(dirname "$0")/../.." || exit 1
PYTHON=/opt/anaconda3/bin/python
OUT=mps_experiments/repeated_seeds

run() {
  local name="$1"; shift
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] START  $name ==="
  $PYTHON train.py "$@" \
    --dataset maestro --data_dir data/maestro-v3.0.0 \
    --epochs 50 --train_chunks 1000 --val_chunks 200 --batch_size 8 \
    --save_dir "$OUT/checkpoints/$name" \
    > "$OUT/logs/$name.log" 2>&1
  local status=$?
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] DONE   $name (exit $status) ==="
}

for seed in 0 1 2; do
  run "gaussian_ce_seed${seed}" --attention gaussian --loss ce --seed "${seed}"
done

for seed in 0 1 2; do
  run "gaussian_ce_standardized_seed${seed}" --attention gaussian --loss ce --seed "${seed}" --standardize_qk
done

for seed in 0 1 2; do
  run "cosine_ce_seed${seed}" --attention cosine --loss ce --seed "${seed}"
done

for seed in 0 1 2; do
  run "cosine_ce_standardized_seed${seed}" --attention cosine --loss ce --seed "${seed}" --standardize_qk
done

for seed in 0 1 2; do
  run "cosformer_ce_seed${seed}" --attention cosformer --loss ce --seed "${seed}"
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] ALL 15 RUNS COMPLETE ==="
