"""
mps_experiments/test_evaluation/evaluate_test_set.py
──────────────────────────────────────────────────────
Read-only held-out test-set evaluation of the 100-epoch checkpoints in
mps_experiments_100ep/checkpoints/{gaussian_ce,cosine_ce,cosformer_ce}/best.pt.

Every result reported elsewhere in this project (the original 9-run matrix,
the standardization ablation, the repeated-seed study, the 100-epoch check)
used training/validation loss only. Daniel's framing was specifically about
"out-of-sample performance" — this script is the first to touch the MAESTRO
test split (177 held-out pieces, never used for training or validation
anywhere in this project).

What this script does:
  - Loads each checkpoint READ-ONLY (map_location='cpu', model.eval(),
    torch.no_grad() throughout) — no training, no gradient updates, nothing
    written back to any checkpoint.
  - Tokenizes the full MAESTRO test split (177 pieces -> 11,451 chunks of
    512 tokens, no chunk cap — cheap because this is inference-only).
  - Computes test loss with the exact same combined_loss() function and
    batch size used during training/validation, for apples-to-apples
    comparability with each checkpoint's own recorded val_loss.
  - Reports test loss, perplexity, and the val->test gap (generalization
    check) per model.

Does NOT modify mps_experiments_100ep/, mps_experiments/checkpoints/,
mps_experiments/standardization_ablation/, or mps_experiments/repeated_seeds/.
All output goes to mps_experiments/test_evaluation/.

Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments/test_evaluation/evaluate_test_set.py
"""

import os
import sys
import csv
import math
import time
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from model.transformer import build_model             # noqa: E402
from data.midi_dataset import MusicDataset             # noqa: E402
from utils.losses import combined_loss, build_token_value_tensor  # noqa: E402

CKPT_100EP_DIR = os.path.join(PROJECT_ROOT, "mps_experiments_100ep", "checkpoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "mps_experiments", "test_evaluation")
RESULTS_DIR = os.path.join(OUT_DIR, "results")

CONFIGS = ["gaussian_ce", "cosine_ce", "cosformer_ce"]
SEQ_LEN = 512
BATCH_SIZE = 8  # matches training/val batch size for exact comparability


def load_checkpoint(name):
    ckpt_path = os.path.join(CKPT_100EP_DIR, name, "best.pt")
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
    device = torch.device("cpu")  # inference-only, CPU is sufficient and keeps this reproducible off-MPS too
    print(f"Device: {device}")

    print("Loading MAESTRO test split (full, uncapped)...")
    test_ds = MusicDataset(
        dataset="maestro",
        data_dir=os.path.join(PROJECT_ROOT, "data", "maestro-v3.0.0"),
        seq_len=SEQ_LEN,
        split="test",
        max_chunks=None,
    )
    n_test_chunks = len(test_ds)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    token_values = build_token_value_tensor(device)

    rows = []
    for name in CONFIGS:
        print(f"\nEvaluating {name} ...")
        model, cfg, ckpt = load_checkpoint(name)
        model = model.to(device)

        t0 = time.time()
        test_loss = evaluate(model, test_loader, cfg, token_values, device)
        elapsed = time.time() - t0

        val_loss_ep100 = ckpt["val_loss"]
        epoch = ckpt["epoch"]
        gap = test_loss - val_loss_ep100
        test_ppl = math.exp(test_loss)
        val_ppl = math.exp(val_loss_ep100)

        print(f"  epoch (checkpoint): {epoch}")
        print(f"  val_loss (recorded @ training time): {val_loss_ep100:.4f}  (perplexity {val_ppl:.2f})")
        print(f"  test_loss (this evaluation):          {test_loss:.4f}  (perplexity {test_ppl:.2f})")
        print(f"  val -> test gap: {gap:+.4f}")
        print(f"  eval time: {elapsed:.1f}s over {n_test_chunks} test chunks")

        rows.append([
            name, epoch, round(val_loss_ep100, 4), round(test_loss, 4), round(gap, 4),
            round(val_ppl, 2), round(test_ppl, 2), n_test_chunks, round(elapsed, 1),
        ])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "test_set_summary.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "checkpoint_epoch", "val_loss", "test_loss", "test_minus_val_gap",
                     "val_perplexity", "test_perplexity", "n_test_chunks", "eval_seconds"])
        w.writerows(rows)

    print(f"\nWrote {out_path}")
    print("\nDone. No existing checkpoint, log, result, or plot file was modified.")


if __name__ == "__main__":
    main()
