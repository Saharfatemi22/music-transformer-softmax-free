"""
mps_experiments/test_evaluation/make_test_eval_plot.py
────────────────────────────────────────────────────────
Generates the val-vs-test comparison plot from results/test_set_summary.csv.
Does not modify any existing file outside mps_experiments/test_evaluation/.

Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments/test_evaluation/make_test_eval_plot.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

RANDOM_BASELINE = 5.2983

CONFIG_COLOR = {
    "gaussian_ce": "#2a78d6",
    "cosine_ce": "#1baf7a",
    "cosformer_ce": "#eb6834",
}
CONFIG_LABEL = {
    "gaussian_ce": "Gaussian",
    "cosine_ce": "Cosine",
    "cosformer_ce": "CosFormer",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def main():
    with open(os.path.join(RESULTS_DIR, "test_set_summary.csv")) as f:
        rows = list(csv.DictReader(f))

    labels = [CONFIG_LABEL[r["config"]] for r in rows]
    colors = [CONFIG_COLOR[r["config"]] for r in rows]
    val_vals = [float(r["val_loss"]) for r in rows]
    test_vals = [float(r["test_loss"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(labels))
    w = 0.32

    bars_val = ax.bar(x - w / 2, val_vals, w, color=colors, alpha=0.55, zorder=3,
                        label="validation loss (200 chunks, training-time)")
    bars_test = ax.bar(x + w / 2, test_vals, w, color=colors, alpha=0.95, zorder=3,
                         hatch="///", edgecolor="white", linewidth=0.5,
                         label="test loss (11,451 chunks, held-out)")

    for bar in list(bars_val) + list(bars_test):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                 f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8.5, color=INK_PRIMARY)

    ax.axhline(RANDOM_BASELINE, color="#d03b3b", linestyle=":", linewidth=1.2,
                label=f"random chance ({RANDOM_BASELINE})", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color=INK_SECONDARY)
    ax.set_ylabel("Loss (epoch 100 checkpoint)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_title("Held-out test-set evaluation vs. training-time validation loss",
                  fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.set_ylim(4.95, 5.35)
    ax.yaxis.grid(True, linestyle=":", color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE_LINE)
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)

    caption = "Solid = validation (used during training); hatched = held-out test split, never seen in training or validation. 177 test pieces -> 11,451 chunks, no cap."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(PLOTS_DIR, "val_vs_test_comparison.png"), dpi=150, facecolor=SURFACE)
    plt.close()
    print("Wrote val_vs_test_comparison.png")


if __name__ == "__main__":
    main()
