"""
mps_experiments/standardization_ablation/make_comparison_plots.py
────────────────────────────────────────────────────────────────
Generates the standardization-ablation comparison table and plots by
reading the existing baseline logs (mps_experiments/checkpoints/{gaussian,
cosine}_ce/training_log.csv, untouched) alongside the two new standardized
runs (mps_experiments/standardization_ablation/checkpoints/*/training_log.csv).

Writes only into mps_experiments/standardization_ablation/{results,plots}/.
Does not modify any existing file.

Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments/standardization_ablation/make_comparison_plots.py
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

RANDOM_BASELINE = 5.2983

# dataviz skill palette
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"
COLOR_BASELINE = "#898781"     # muted gray — "old" / non-standardized
COLOR_STANDARDIZED = "#2a78d6"  # categorical slot 1 (blue) — "new" / standardized


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_summary_csv(rows, path):
    header = ["attention_type", "loss_type", "standardize_qk", "final_val_loss",
              "delta_vs_baseline", "pct_change", "epoch_first_below_random_chance",
              "steady_state_s_per_epoch", "total_epochs"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def first_below(rows, thresh=RANDOM_BASELINE):
    for r in rows:
        if float(r["val_loss"]) < thresh:
            return int(r["epoch"])
    return None


def steady_state_avg_time(rows):
    times = [float(r["time_s"]) for r in rows[5:]]
    times = [t for t in times if t < 50]  # exclude CPU/OS scheduling spikes
    return sum(times) / len(times)


def plot_val_curves(base_rows, std_rows, attn_name, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ep_base = [int(r["epoch"]) for r in base_rows]
    val_base = [float(r["val_loss"]) for r in base_rows]
    ep_std = [int(r["epoch"]) for r in std_rows]
    val_std = [float(r["val_loss"]) for r in std_rows]

    ax.plot(ep_base, val_base, color=COLOR_BASELINE, linewidth=2, linestyle="--",
             label="baseline (no standardization)", zorder=3)
    ax.plot(ep_std, val_std, color=COLOR_STANDARDIZED, linewidth=2,
             label="standardize_qk = True", zorder=4)

    ax.axhline(RANDOM_BASELINE, color="#d03b3b", linestyle=":", linewidth=1.2,
                label=f"random chance (log 196 = {RANDOM_BASELINE})", zorder=2)

    ax.set_xlabel("epoch", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylabel("validation loss", fontsize=10.5, color=INK_SECONDARY)
    ax.set_title(f"{attn_name} + CE — Q/K standardization ablation  [MPS · 50 epochs]",
                  fontsize=12, color=INK_PRIMARY, pad=10)
    ax.tick_params(labelsize=9, colors=INK_SECONDARY)
    ax.yaxis.grid(True, linestyle=":", color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE_LINE)
    ax.legend(fontsize=9, loc="upper right", frameon=False)

    caption = "d_model=64 | n_layers=2 | n_heads=4 | batch=8 | 1000 train chunks | 200 val chunks | MAESTRO v3.0.0 | Apple MPS"
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def plot_final_bar_comparison(gauss_base_final, gauss_std_final, cos_base_final, cos_std_final, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ["Gaussian", "Cosine"]
    base_vals = [gauss_base_final, cos_base_final]
    std_vals = [gauss_std_final, cos_std_final]

    import numpy as np
    x = np.arange(len(labels))
    w = 0.32

    bars_base = ax.bar(x - w / 2, base_vals, w, color=COLOR_BASELINE, alpha=0.9,
                         label="baseline (no standardization)", zorder=3)
    bars_std = ax.bar(x + w / 2, std_vals, w, color=COLOR_STANDARDIZED, alpha=0.9,
                        label="standardize_qk = True", zorder=3)

    ax.axhline(RANDOM_BASELINE, color="#d03b3b", linestyle=":", linewidth=1.2,
                label=f"random chance ({RANDOM_BASELINE})", zorder=2)

    for bar in list(bars_base) + list(bars_std):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                 f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9, color=INK_PRIMARY)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color=INK_SECONDARY)
    ax.set_ylabel("Validation loss (epoch 50)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_title("Q/K standardization: final validation loss  [MPS · 50 epochs]",
                  fontsize=12, color=INK_PRIMARY, pad=10)
    ax.set_ylim(5.0, 5.35)
    ax.yaxis.grid(True, linestyle=":", color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE_LINE)
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)

    caption = "Lower is better. Baseline runs from mps_experiments/checkpoints/ (unmodified)."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def main():
    gauss_base = load(os.path.join(PROJECT_ROOT, "mps_experiments/checkpoints/gaussian_ce/training_log.csv"))
    gauss_std = load(os.path.join(RESULTS_DIR, "..", "checkpoints/gaussian_ce_standardized/training_log.csv"))
    cos_base = load(os.path.join(PROJECT_ROOT, "mps_experiments/checkpoints/cosine_ce/training_log.csv"))
    cos_std = load(os.path.join(RESULTS_DIR, "..", "checkpoints/cosine_ce_standardized/training_log.csv"))

    summary_rows = []
    for attn, base, std in [("gaussian", gauss_base, gauss_std), ("cosine", cos_base, cos_std)]:
        base_final = float(base[-1]["val_loss"])
        std_final = float(std[-1]["val_loss"])
        summary_rows.append([
            attn, "ce", False, round(base_final, 4), 0.0, 0.0,
            first_below(base), round(steady_state_avg_time(base), 2), len(base),
        ])
        summary_rows.append([
            attn, "ce", True, round(std_final, 4), round(std_final - base_final, 4),
            round(100 * (std_final - base_final) / base_final, 2),
            first_below(std), round(steady_state_avg_time(std), 2), len(std),
        ])

    write_summary_csv(summary_rows, os.path.join(RESULTS_DIR, "standardization_comparison_summary.csv"))
    print("Wrote standardization_comparison_summary.csv")

    plot_val_curves(gauss_base, gauss_std, "Gaussian", os.path.join(PLOTS_DIR, "gaussian_val_loss_comparison.png"))
    plot_val_curves(cos_base, cos_std, "Cosine", os.path.join(PLOTS_DIR, "cosine_val_loss_comparison.png"))
    print("Wrote gaussian_val_loss_comparison.png, cosine_val_loss_comparison.png")

    plot_final_bar_comparison(
        float(gauss_base[-1]["val_loss"]), float(gauss_std[-1]["val_loss"]),
        float(cos_base[-1]["val_loss"]), float(cos_std[-1]["val_loss"]),
        os.path.join(PLOTS_DIR, "final_val_loss_bar_comparison.png"),
    )
    print("Wrote final_val_loss_bar_comparison.png")


if __name__ == "__main__":
    main()
