"""
mps_experiments/ordinal_mse_evaluation/evaluate_ordinal_mse.py
──────────────────────────────────────────────────────────────
The pure-CE decomposition (mps_experiments_100ep/decompose_ntl_loss.py)
showed NTL training doesn't change next-token classification quality. But
that isn't what NTL is actually FOR -- the Number Token Loss papers claim
it improves NUMERICAL CLOSENESS on ordinal-token errors (predicting pitch
61 instead of 60 should be a smaller error than predicting pitch 5), which
plain cross-entropy cannot express and pure-CE cannot measure.

This script tests that actual claim directly: re-evaluates all 12 existing
100-epoch checkpoints (gaussian/cosformer x {ce,ce+ntl} x 3 seeds) on
ordinal-token MSE alone -- using the exact same _mse_ordinal() function the
NTL loss itself optimizes -- regardless of what each checkpoint was trained
with. This isolates: does training with the NTL term actually produce a
model with better numerical-closeness on pitch/duration/velocity errors?

Read-only: loads checkpoints with map_location='cpu', eval()/no_grad()
throughout. Does not retrain or modify any model, and does not touch
mps_experiments_100ep/ in any way.

Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments/ordinal_mse_evaluation/evaluate_ordinal_mse.py
"""

import os
import sys
import csv
import statistics
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from model.transformer import build_model                          # noqa: E402
from data.midi_dataset import MusicDataset                          # noqa: E402
from utils.losses import build_token_value_tensor, _mse_ordinal     # noqa: E402

CKPT_100EP_DIR = os.path.join(PROJECT_ROOT, "mps_experiments_100ep", "checkpoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "mps_experiments", "ordinal_mse_evaluation")
RESULTS_DIR = os.path.join(OUT_DIR, "results")

CONFIGS = {
    "gaussian_ce": ["gaussian_ce", "gaussian_ce_seed1", "gaussian_ce_seed2"],
    "gaussian_ntl": ["gaussian_ntl_seed0", "gaussian_ntl_seed1", "gaussian_ntl_seed2"],
    "cosformer_ce": ["cosformer_ce", "cosformer_ce_seed1", "cosformer_ce_seed2"],
    "cosformer_ntl": ["cosformer_ntl_seed0", "cosformer_ntl_seed1", "cosformer_ntl_seed2"],
}


def load_checkpoint(name):
    ckpt_path = os.path.join(CKPT_100EP_DIR, name, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def evaluate_ordinal_mse(model, loader, token_values, device):
    total = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = y.view(-1)
        mse = _mse_ordinal(logits_flat, targets_flat, token_values)
        total += mse.item()
        n_batches += 1
    return total / max(1, n_batches)


def main():
    device = torch.device("cpu")

    print("Loading MAESTRO validation split (200 chunks, matching training-time val set)...")
    val_ds = MusicDataset(
        dataset="maestro",
        data_dir=os.path.join(PROJECT_ROOT, "data", "maestro-v3.0.0"),
        seq_len=512,
        split="val",
        max_chunks=200,
    )
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    token_values = build_token_value_tensor(device)

    rows = []
    group_stats = {}

    for group_name, ckpt_names in CONFIGS.items():
        vals = []
        for ckpt_name in ckpt_names:
            model, cfg, ckpt = load_checkpoint(ckpt_name)
            model = model.to(device)
            ordinal_mse = evaluate_ordinal_mse(model, val_loader, token_values, device)
            vals.append(ordinal_mse)
            print(f"{ckpt_name}: ordinal_token_MSE={ordinal_mse:.6f}")
            rows.append([group_name, ckpt_name, round(ordinal_mse, 6)])

        mean = statistics.mean(vals)
        std = statistics.stdev(vals)
        group_stats[group_name] = (vals, mean, std)
        print(f"  -> {group_name} mean={mean:.6f} std={std:.6f}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ordinal_mse_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "checkpoint", "ordinal_token_mse"])
        w.writerows(rows)
    print("Wrote ordinal_mse_summary.csv")

    print("\n=== Does CE+NTL training improve ordinal-token (pitch/duration/velocity) MSE? ===")
    verdict_rows = []
    for attn in ["gaussian", "cosformer"]:
        ce_vals, ce_mean, ce_std = group_stats[f"{attn}_ce"]
        ntl_vals, ntl_mean, ntl_std = group_stats[f"{attn}_ntl"]
        diff = ntl_mean - ce_mean
        pct = 100 * diff / ce_mean
        avg_std = (ce_std + ntl_std) / 2
        print(f"{attn}: CE-trained ordinal-MSE={ce_mean:.6f}±{ce_std:.6f}  "
              f"NTL-trained ordinal-MSE={ntl_mean:.6f}±{ntl_std:.6f}  "
              f"diff={diff:+.6f} ({pct:+.1f}%)  ({diff/avg_std:+.2f}x avg std)")
        verdict_rows.append([attn, round(ce_mean, 6), round(ce_std, 6), round(ntl_mean, 6),
                              round(ntl_std, 6), round(diff, 6), round(pct, 2)])

    with open(os.path.join(RESULTS_DIR, "ordinal_mse_verdict.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attention_type", "ce_mean", "ce_std", "ntl_mean", "ntl_std",
                     "diff_ntl_minus_ce", "pct_change"])
        w.writerows(verdict_rows)
    print("Wrote ordinal_mse_verdict.csv")


if __name__ == "__main__":
    main()
