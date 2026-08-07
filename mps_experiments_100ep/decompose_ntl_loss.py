"""
mps_experiments_100ep/decompose_ntl_loss.py
──────────────────────────────────────────────
The logged val_loss for "ce+ntl" checkpoints is CE + lambda*MSE_ordinal,
not pure cross-entropy -- so it is NOT on the same scale as a "ce"
checkpoint's logged val_loss, even though the project's earlier docs
described them as "directly comparable." This script re-evaluates every
CE and CE+NTL checkpoint (gaussian/cosformer, 3 seeds each = 12 checkpoints)
under a SINGLE FIXED METRIC -- plain cross-entropy, ignore_index=0, forced
via cfg override regardless of how the model was trained -- on the same
200-chunk validation set used at training time. This isolates the real
question: does NTL training change next-token prediction quality (pure CE),
or does the earlier "NTL is worse" comparison just reflect the extra
additive MSE term inflating the logged number?

Read-only: loads checkpoints with map_location='cpu', eval()/no_grad()
throughout. Does not modify any checkpoint or existing result file.

Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments_100ep/decompose_ntl_loss.py
"""

import os
import sys
import csv
import copy
import statistics
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from model.transformer import build_model              # noqa: E402
from data.midi_dataset import MusicDataset              # noqa: E402
from utils.losses import combined_loss, build_token_value_tensor  # noqa: E402

CKPT_DIR = os.path.join(PROJECT_ROOT, "mps_experiments_100ep", "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "mps_experiments_100ep", "results")

CONFIGS = {
    "gaussian_ce": ["gaussian_ce", "gaussian_ce_seed1", "gaussian_ce_seed2"],
    "gaussian_ntl": ["gaussian_ntl_seed0", "gaussian_ntl_seed1", "gaussian_ntl_seed2"],
    "cosformer_ce": ["cosformer_ce", "cosformer_ce_seed1", "cosformer_ce_seed2"],
    "cosformer_ntl": ["cosformer_ntl_seed0", "cosformer_ntl_seed1", "cosformer_ntl_seed2"],
}


def load_checkpoint(name):
    ckpt_path = os.path.join(CKPT_DIR, name, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def evaluate(model, loader, cfg, token_values, device):
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = combined_loss(logits, y, cfg, token_values)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


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
        pure_ce_vals = []
        logged_vals = []
        for ckpt_name in ckpt_names:
            model, cfg, ckpt = load_checkpoint(ckpt_name)
            model = model.to(device)

            # Force pure CE evaluation regardless of how the model was trained
            cfg_pure_ce = copy.deepcopy(cfg)
            cfg_pure_ce["loss_type"] = "ce"
            pure_ce = evaluate(model, val_loader, cfg_pure_ce, token_values, device)

            logged_val_loss = ckpt["val_loss"]
            ntl_contribution = logged_val_loss - pure_ce  # only meaningful for ce+ntl checkpoints

            pure_ce_vals.append(pure_ce)
            logged_vals.append(logged_val_loss)

            print(f"{ckpt_name}: logged_val_loss(own objective)={logged_val_loss:.4f}  "
                  f"pure_CE(re-evaluated)={pure_ce:.4f}  ntl_term_estimate={ntl_contribution:+.4f}")

            rows.append([group_name, ckpt_name, round(logged_val_loss, 4), round(pure_ce, 4),
                         round(ntl_contribution, 4)])

        mean_pure_ce = statistics.mean(pure_ce_vals)
        std_pure_ce = statistics.stdev(pure_ce_vals)
        group_stats[group_name] = (pure_ce_vals, mean_pure_ce, std_pure_ce)
        print(f"  -> {group_name} pure-CE mean={mean_pure_ce:.4f} std={std_pure_ce:.4f}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ntl_pure_ce_decomposition.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "checkpoint", "logged_val_loss_own_objective", "pure_ce_reevaluated",
                     "estimated_ntl_term_contribution"])
        w.writerows(rows)
    print("Wrote ntl_pure_ce_decomposition.csv")

    print("\n=== Pure-CE comparison: does NTL training change next-token prediction quality? ===")
    for attn in ["gaussian", "cosformer"]:
        ce_vals, ce_mean, ce_std = group_stats[f"{attn}_ce"]
        ntl_vals, ntl_mean, ntl_std = group_stats[f"{attn}_ntl"]
        diff = ntl_mean - ce_mean
        avg_std = (ce_std + ntl_std) / 2
        print(f"{attn}: CE-trained pure-CE={ce_mean:.4f}±{ce_std:.4f}  "
              f"NTL-trained pure-CE={ntl_mean:.4f}±{ntl_std:.4f}  "
              f"diff={diff:+.4f}  ({diff/avg_std:+.2f}x avg std)")


if __name__ == "__main__":
    main()
