"""
mps_experiments_100ep/make_tie_resolution_plot.py
────────────────────────────────────────────────────
Generates the tie-resolution summary (Gaussian vs CosFormer, 3 seeds each,
100 epochs) from the existing seed-0 runs plus the new seed-1/seed-2 runs.
Does not modify mps_experiments_100ep/results/100ep_summary.csv,
RESULTS_100EP.md, or any checkpoint — writes only new files.

Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments_100ep/make_tie_resolution_plot.py
"""

import os
import csv
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

RANDOM_BASELINE = 5.2983

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"

CONFIGS = {
    "gaussian": {
        "color": "#2a78d6",
        "dirs": {0: "gaussian_ce", 1: "gaussian_ce_seed1", 2: "gaussian_ce_seed2"},
    },
    "cosformer": {
        "color": "#eb6834",
        "dirs": {0: "cosformer_ce", 1: "cosformer_ce_seed1", 2: "cosformer_ce_seed2"},
    },
}


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    summary_rows = []
    stats = {}

    for name, info in CONFIGS.items():
        vals = []
        for seed, dirname in info["dirs"].items():
            rows = load(os.path.join(CKPT_DIR, dirname, "training_log.csv"))
            final_val = float(rows[-1]["val_loss"])
            total_time = sum(float(r["time_s"]) for r in rows)
            vals.append(final_val)
            summary_rows.append([name, seed, dirname, final_val, round(total_time / 60, 1)])
        mean = statistics.mean(vals)
        std = statistics.stdev(vals)
        stats[name] = (vals, mean, std)

    with open(os.path.join(RESULTS_DIR, "tie_resolution_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attention_type", "seed", "checkpoint_dir", "val_loss_ep100", "total_minutes"])
        w.writerows(summary_rows)
    print("Wrote tie_resolution_summary.csv")

    # ── Plot: mean +/- std, Gaussian vs CosFormer, 3 seeds each ──────────
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    labels = ["Gaussian", "CosFormer"]
    colors = [CONFIGS["gaussian"]["color"], CONFIGS["cosformer"]["color"]]
    means = [stats["gaussian"][1], stats["cosformer"][1]]
    stds = [stats["gaussian"][2], stats["cosformer"][2]]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, 0.5, yerr=stds, capsize=6, color=colors, alpha=0.9,
                   error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1.4, capthick=1.4), zorder=3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.015,
                 f"{mean:.4f}\n±{std:.4f}", ha="center", va="bottom", fontsize=9.5, color=INK_PRIMARY)

    # overlay individual seed points
    for i, name in enumerate(["gaussian", "cosformer"]):
        vals = stats[name][0]
        jitter = np.linspace(-0.08, 0.08, len(vals))
        ax.scatter([i + j for j in jitter], vals, color=INK_PRIMARY, s=28, zorder=5,
                    edgecolor="white", linewidth=0.8)

    ax.axhline(RANDOM_BASELINE, color="#d03b3b", linestyle=":", linewidth=1.2,
                label=f"random chance ({RANDOM_BASELINE})", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5, color=INK_SECONDARY)
    ax.set_ylabel("Validation loss (epoch 100), mean ± std over 3 seeds", fontsize=10.5, color=INK_SECONDARY)
    ax.set_title("Tie resolution: Gaussian vs. CosFormer at 100 epochs (n=3 seeds)",
                  fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.set_ylim(4.75, 5.35)
    ax.yaxis.grid(True, linestyle=":", color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE_LINE)
    ax.legend(fontsize=9, loc="upper right", frameon=False)

    caption = "Black dots = individual seed values {0,1,2}. d_model=64 | n_layers=2 | 1000 train / 200 val chunks | 100 epochs | MPS."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(PLOTS_DIR, "tie_resolution_mean_std.png"), dpi=150, facecolor=SURFACE)
    plt.close()
    print("Wrote tie_resolution_mean_std.png")

    return stats


if __name__ == "__main__":
    main()
