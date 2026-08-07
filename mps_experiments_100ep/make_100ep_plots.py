"""
mps_experiments_100ep/make_100ep_plots.py
──────────────────────────────────────────
Generates the 100-epoch comparison summary and plot from the 3 training
logs (gaussian_ce, cosine_ce, cosformer_ce; CE loss, no standardization,
seed 0). Reads only from mps_experiments_100ep/checkpoints/*/training_log.csv.
Does not modify any existing file outside mps_experiments_100ep/.

Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments_100ep/make_100ep_plots.py
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

CONFIGS = [
    ("gaussian_ce", "Gaussian CE", "#2a78d6"),
    ("cosine_ce", "Cosine CE", "#1baf7a"),
    ("cosformer_ce", "CosFormer CE", "#eb6834"),
]
RANDOM_BASELINE = 5.2983

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    summary_rows = []
    curves = {}

    for prefix, label, color in CONFIGS:
        rows = load(os.path.join(CKPT_DIR, prefix, "training_log.csv"))
        val_curve = [float(r["val_loss"]) for r in rows]
        curves[prefix] = val_curve
        ep50 = val_curve[49]
        ep100 = val_curve[-1]
        total_time = sum(float(r["time_s"]) for r in rows)
        still_decreasing = val_curve[-1] < val_curve[89]
        summary_rows.append([prefix, label, round(ep50, 4), round(ep100, 4),
                              round(ep100 - ep50, 4), still_decreasing, round(total_time / 60, 1)])

    with open(os.path.join(RESULTS_DIR, "100ep_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "label", "val_loss_ep50", "val_loss_ep100",
                     "delta_ep50_to_ep100", "still_decreasing_at_ep100", "total_minutes"])
        w.writerows(summary_rows)
    print("Wrote 100ep_summary.csv")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for prefix, label, color in CONFIGS:
        curve = curves[prefix]
        ax.plot(range(1, len(curve) + 1), curve, color=color, linewidth=2, label=label, zorder=3)

    ax.axvline(50, color=BASELINE_LINE, linestyle="--", linewidth=1.0, zorder=2)
    ax.text(50.5, ax.get_ylim()[1] if False else 5.42, "epoch 50", fontsize=8, color=INK_MUTED)
    ax.axhline(RANDOM_BASELINE, color="#d03b3b", linestyle=":", linewidth=1.2,
                label=f"random chance ({RANDOM_BASELINE})", zorder=2)

    ax.set_xlabel("epoch", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylabel("validation loss", fontsize=10.5, color=INK_SECONDARY)
    ax.set_title("100-epoch comparison — CE loss, no standardization, seed 0 [MPS]",
                  fontsize=12.5, color=INK_PRIMARY, pad=10)
    ax.tick_params(labelsize=9, colors=INK_SECONDARY)
    ax.yaxis.grid(True, linestyle=":", color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE_LINE)
    ax.legend(fontsize=9, loc="upper right", frameon=False)

    caption = "d_model=64 | n_layers=2 | n_heads=4 | batch=8 | 1000 train / 200 val chunks | seed 0 | MPS. Single seed — not a repeated-seed result."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(PLOTS_DIR, "100ep_val_loss_comparison.png"), dpi=150, facecolor=SURFACE)
    plt.close()
    print("Wrote 100ep_val_loss_comparison.png")


if __name__ == "__main__":
    main()
