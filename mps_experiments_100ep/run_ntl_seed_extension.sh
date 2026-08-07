#!/bin/bash
# mps_experiments_100ep/run_ntl_seed_extension.sh
# ────────────────────────────────────────────────────────────────────────
# Brings repeated-seed rigor to the loss-function comparison (Daniel's
# actual second research axis): CE+NTL vs. the existing CE runs, for
# Gaussian and CosFormer, 3 seeds each, at the same 100-epoch scale as the
# existing gaussian_ce/cosformer_ce (+seed1/seed2) runs already in
# mps_experiments_100ep/checkpoints/ — none of which are touched or renamed.
#
# Cosine excluded (already conclusively worse on CE; not the priority).
# "mse" loss excluded (different scale, was an internal ablation, not the
# paper's core CE-vs-NTL comparison).
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

run "gaussian_ntl_seed0" --attention gaussian --loss "ce+ntl" --seed 0
run "gaussian_ntl_seed1" --attention gaussian --loss "ce+ntl" --seed 1
run "gaussian_ntl_seed2" --attention gaussian --loss "ce+ntl" --seed 2
run "cosformer_ntl_seed0" --attention cosformer --loss "ce+ntl" --seed 0
run "cosformer_ntl_seed1" --attention cosformer --loss "ce+ntl" --seed 1
run "cosformer_ntl_seed2" --attention cosformer --loss "ce+ntl" --seed 2

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] ALL 6 NTL SEED RUNS COMPLETE ==="
