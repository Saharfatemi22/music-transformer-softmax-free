"""
mps_experiments/ordinal_mse_evaluation/make_ordinal_mse_plot.py
──────────────────────────────────────────────────────────────
Generates the ordinal-token MSE comparison plot from results/ordinal_mse_summary.csv.
Does not modify any existing file outside mps_experiments/ordinal_mse_evaluation/.

Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments/ordinal_mse_evaluation/make_ordinal_mse_plot.py
"""

import os
import csv
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"
ATTN_COLOR = {"gaussian": "#2a78d6", "cosformer": "#eb6834"}


def main():
    with open(os.path.join(RESULTS_DIR, "ordinal_mse_summary.csv")) as f:
        rows = list(csv.DictReader(f))

    grouped = {}
    for r in rows:
        grouped.setdefault(r["group"], []).append(float(r["ordinal_token_mse"]))

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    labels = ["Gaussian\nCE", "Gaussian\nCE+NTL", "CosFormer\nCE", "CosFormer\nCE+NTL"]
    keys = ["gaussian_ce", "gaussian_ntl", "cosformer_ce", "cosformer_ntl"]
    colors = [ATTN_COLOR["gaussian"], ATTN_COLOR["gaussian"], ATTN_COLOR["cosformer"], ATTN_COLOR["cosformer"]]
    alphas = [0.95, 0.55, 0.95, 0.55]
    hatches = [None, "///", None, "///"]

    x = np.arange(len(labels))
    means = [statistics.mean(grouped[k]) for k in keys]
    stds = [statistics.stdev(grouped[k]) for k in keys]

    for xi, mean, std, color, alpha, hatch, key in zip(x, means, stds, colors, alphas, hatches, keys):
        ax.bar(xi, mean, 0.6, yerr=std, capsize=5, color=color, alpha=alpha, hatch=hatch,
               edgecolor="white" if hatch else color, linewidth=0.8,
               error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1.3, capthick=1.3), zorder=3)
        ax.text(xi, mean + std + 0.0008, f"{mean:.5f}\n±{std:.5f}", ha="center", va="bottom",
                 fontsize=8, color=INK_PRIMARY)
        vals = grouped[key]
        jitter = np.linspace(-0.06, 0.06, len(vals))
        ax.scatter([xi + j for j in jitter], vals, color=INK_PRIMARY, s=22, zorder=5,
                    edgecolor="white", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("Ordinal-token MSE (pitch/duration/velocity, normalised [0,1] scale)",
                  fontsize=9.5, color=INK_SECONDARY)
    ax.set_title("Does CE+NTL training improve ordinal-token numerical closeness?",
                  fontsize=12, color=INK_PRIMARY, pad=12)
    ax.set_ylim(0.060, 0.078)
    ax.yaxis.grid(True, linestyle=":", color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE_LINE)

    caption = ("Solid = CE-trained, hatched = CE+NTL-trained. Lower = better numerical closeness on pitch/duration/velocity\n"
               "errors -- the metric NTL is actually designed to improve. Black dots = individual seeds. n=3 seeds, 100 epochs.")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.3, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(os.path.join(PLOTS_DIR, "ordinal_mse_comparison.png"), dpi=150, facecolor=SURFACE)
    plt.close()
    print("Wrote ordinal_mse_comparison.png")


if __name__ == "__main__":
    main()
