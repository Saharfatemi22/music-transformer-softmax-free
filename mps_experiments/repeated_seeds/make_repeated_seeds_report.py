"""
mps_experiments/repeated_seeds/make_repeated_seeds_report.py
─────────────────────────────────────────────────────────────
Reads the 15 repeated-seed training logs (5 configs x 3 seeds) and writes:
  - results/repeated_seeds_summary.csv   (one row per run)
  - results/repeated_seeds_stats.csv     (one row per config: mean/std)
  - plots/mean_std_bar_comparison.png    (bar chart with error bars)
  - plots/all_seeds_val_curves.png       (small multiples, all seeds overlaid)

Reads only from mps_experiments/repeated_seeds/checkpoints/*/training_log.csv.
Does not modify any existing file.

Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments/repeated_seeds/make_repeated_seeds_report.py
"""

import os
import csv
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(__file__)
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

CONFIGS = [
    ("gaussian_ce", "Gaussian CE", "#2a78d6"),
    ("gaussian_ce_standardized", "Gaussian CE +std", "#5598e7"),
    ("cosine_ce", "Cosine CE", "#1baf7a"),
    ("cosine_ce_standardized", "Cosine CE +std", "#6dd6b3"),
    ("cosformer_ce", "CosFormer CE", "#eb6834"),
]
SEEDS = [0, 1, 2]
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
    per_run_rows = []
    stats_rows = []
    curves = {}  # prefix -> list of (seed, [val_loss per epoch])

    for prefix, label, color in CONFIGS:
        vals = []
        curves[prefix] = []
        for seed in SEEDS:
            path = os.path.join(CKPT_DIR, f"{prefix}_seed{seed}", "training_log.csv")
            rows = load(path)
            val_curve = [float(r["val_loss"]) for r in rows]
            final_val = val_curve[-1]
            total_time = sum(float(r["time_s"]) for r in rows)
            vals.append(final_val)
            curves[prefix].append(val_curve)
            per_run_rows.append([prefix, label, seed, final_val, round(total_time / 60, 2)])

        mean = statistics.mean(vals)
        std = statistics.stdev(vals)
        stats_rows.append([prefix, label, *[round(v, 4) for v in vals], round(mean, 4), round(std, 4)])

    with open(os.path.join(RESULTS_DIR, "repeated_seeds_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "label", "seed", "final_val_loss", "total_minutes"])
        w.writerows(per_run_rows)

    with open(os.path.join(RESULTS_DIR, "repeated_seeds_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "label", "seed0_val_loss", "seed1_val_loss", "seed2_val_loss",
                     "mean_val_loss", "std_val_loss"])
        w.writerows(stats_rows)

    print("Wrote repeated_seeds_summary.csv, repeated_seeds_stats.csv")

    # ── Plot 1: mean +/- std bar chart ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [c[1] for c in CONFIGS]
    colors = [c[2] for c in CONFIGS]
    means = [row[-2] for row in stats_rows]
    stds = [row[-1] for row in stats_rows]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, 0.55, yerr=stds, capsize=5, color=colors, alpha=0.9,
                   error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1.3, capthick=1.3), zorder=3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.012,
                 f"{mean:.4f}\n±{std:.4f}", ha="center", va="bottom", fontsize=8.5, color=INK_PRIMARY)

    ax.axhline(RANDOM_BASELINE, color="#d03b3b", linestyle=":", linewidth=1.2,
                label=f"random chance ({RANDOM_BASELINE})", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Validation loss (epoch 50), mean ± std over 3 seeds", fontsize=10.5, color=INK_SECONDARY)
    ax.set_title("Repeated-seed validation (n=3): mean final validation loss", fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.set_ylim(4.9, 5.45)
    ax.yaxis.grid(True, linestyle=":", color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE_LINE)
    ax.legend(fontsize=9, loc="upper right", frameon=False)

    caption = "Error bars = sample std dev across seeds {0,1,2}. d_model=64 | n_layers=2 | 1000 train / 200 val chunks | 50 epochs | MPS"
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(PLOTS_DIR, "mean_std_bar_comparison.png"), dpi=150, facecolor=SURFACE)
    plt.close()
    print("Wrote mean_std_bar_comparison.png")

    # ── Plot 2: small multiples, all seed curves overlaid per config ───────
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2), sharey=True)
    seed_linestyles = ["-", "--", ":"]

    for ax, (prefix, label, color) in zip(axes, CONFIGS):
        for seed, curve, ls in zip(SEEDS, curves[prefix], seed_linestyles):
            ax.plot(range(1, len(curve) + 1), curve, color=color, linewidth=1.6,
                     linestyle=ls, label=f"seed {seed}", zorder=3)
        ax.axhline(RANDOM_BASELINE, color="#d03b3b", linestyle=":", linewidth=1.0, zorder=2)
        ax.set_title(label, fontsize=10.5, color=INK_PRIMARY)
        ax.set_xlabel("epoch", fontsize=9, color=INK_SECONDARY)
        ax.tick_params(labelsize=8, colors=INK_SECONDARY)
        ax.yaxis.grid(True, linestyle=":", color=GRIDLINE, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(BASELINE_LINE)
        ax.legend(fontsize=7, loc="upper right", frameon=False)

    axes[0].set_ylabel("validation loss", fontsize=9.5, color=INK_SECONDARY)
    fig.suptitle("All 3 seeds per configuration — validation loss curves", fontsize=13, color=INK_PRIMARY)
    caption = "Same model size and data budget across all panels; only the random seed (init/dropout/batch order) differs within each panel."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.06, 1, 0.90])
    plt.savefig(os.path.join(PLOTS_DIR, "all_seeds_val_curves.png"), dpi=150, facecolor=SURFACE)
    plt.close()
    print("Wrote all_seeds_val_curves.png")


if __name__ == "__main__":
    main()
