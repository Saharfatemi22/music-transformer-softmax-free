#!/bin/bash
# mps_experiments_100ep/run_seed_extension.sh
# ────────────────────────────────────────────────────────────────────────
# Adds seeds 1 and 2 for Gaussian+CE and CosFormer+CE at 100 epochs, to
# resolve the Gaussian-vs-CosFormer statistical tie found at n=1 seed.
# Reuses the existing seed-0 100-epoch runs already in
# mps_experiments_100ep/checkpoints/{gaussian_ce,cosformer_ce}/ — those are
# NOT touched or renamed; this script only adds new _seed1/_seed2 directories.
#
# Cosine is deliberately excluded (already conclusively worse, no open
# question to resolve there). CE loss, no standardization, matching every
# other 100-epoch run in this project.
#
# Deliberately does NOT use `set -e`: one run failing should not skip the
# rest.

cd "$(dirname "$0")/.." || exit 1
PYTHON=/opt/anaconda3/bin/python
OUT=mps_experiments_100ep

run() {
  local name="$1"; shift
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] START  $name ==="
  $PYTHON train.py "$@" \
    --dataset maestro --data_dir data/maestro-v3.0.0 \
    --epochs 100 --train_chunks 1000 --val_chunks 200 --batch_size 8 \
    --save_dir "$OUT/checkpoints/$name" \
    > "$OUT/logs/$name.log" 2>&1
  local status=$?
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] DONE   $name (exit $status) ==="
}

run "gaussian_ce_seed1" --attention gaussian --loss ce --seed 1
run "gaussian_ce_seed2" --attention gaussian --loss ce --seed 2
run "cosformer_ce_seed1" --attention cosformer --loss ce --seed 1
run "cosformer_ce_seed2" --attention cosformer --loss ce --seed 2

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] ALL 4 SEED-EXTENSION RUNS COMPLETE ==="
