"""
mps_experiments_100ep/make_ntl_comparison_plot.py
────────────────────────────────────────────────────
Generates the CE-vs-CE+NTL comparison summary and plot, combining:
  - the raw logged val_loss (each checkpoint's own training objective)
  - the pure-CE re-evaluation (fixed metric, isolates prediction quality)
from results/tie_resolution_summary.csv (CE) and the 6 new NTL runs, plus
results/ntl_pure_ce_decomposition.csv (from decompose_ntl_loss.py).

Does not modify any existing file outside mps_experiments_100ep/results,
mps_experiments_100ep/plots.

Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments_100ep/make_ntl_comparison_plot.py
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

ATTN_COLOR = {"gaussian": "#2a78d6", "cosformer": "#eb6834"}

CE_DIRS = {
    "gaussian": ["gaussian_ce", "gaussian_ce_seed1", "gaussian_ce_seed2"],
    "cosformer": ["cosformer_ce", "cosformer_ce_seed1", "cosformer_ce_seed2"],
}
NTL_DIRS = {
    "gaussian": ["gaussian_ntl_seed0", "gaussian_ntl_seed1", "gaussian_ntl_seed2"],
    "cosformer": ["cosformer_ntl_seed0", "cosformer_ntl_seed1", "cosformer_ntl_seed2"],
}


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def final_val(dirname):
    rows = load(os.path.join(CKPT_DIR, dirname, "training_log.csv"))
    return float(rows[-1]["val_loss"])


def main():
    decomp_rows = load(os.path.join(RESULTS_DIR, "ntl_pure_ce_decomposition.csv"))
    pure_ce_by_ckpt = {r["checkpoint"]: float(r["pure_ce_reevaluated"]) for r in decomp_rows}

    summary_rows = []
    stats = {}  # (attn, loss_type) -> (logged_vals, logged_mean, logged_std, pure_ce_vals, pure_ce_mean, pure_ce_std)

    for attn in ["gaussian", "cosformer"]:
        for loss_type, dirs in [("ce", CE_DIRS[attn]), ("ce+ntl", NTL_DIRS[attn])]:
            logged_vals = [final_val(d) for d in dirs]
            pure_ce_vals = [pure_ce_by_ckpt[d] for d in dirs]
            logged_mean, logged_std = statistics.mean(logged_vals), statistics.stdev(logged_vals)
            pure_mean, pure_std = statistics.mean(pure_ce_vals), statistics.stdev(pure_ce_vals)
            stats[(attn, loss_type)] = (logged_vals, logged_mean, logged_std, pure_ce_vals, pure_mean, pure_std)
            for d, lv, pv in zip(dirs, logged_vals, pure_ce_vals):
                summary_rows.append([attn, loss_type, d, round(lv, 4), round(pv, 4)])

    with open(os.path.join(RESULTS_DIR, "ntl_vs_ce_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attention_type", "loss_type", "checkpoint", "logged_val_loss_own_objective",
                     "pure_ce_reevaluated"])
        w.writerows(summary_rows)
    print("Wrote ntl_vs_ce_summary.csv")

    # ── Plot: pure-CE mean +/- std, CE vs CE+NTL, both attention types ────
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    labels = ["Gaussian\nCE", "Gaussian\nCE+NTL", "CosFormer\nCE", "CosFormer\nCE+NTL"]
    keys = [("gaussian", "ce"), ("gaussian", "ce+ntl"), ("cosformer", "ce"), ("cosformer", "ce+ntl")]
    colors = [ATTN_COLOR["gaussian"], ATTN_COLOR["gaussian"], ATTN_COLOR["cosformer"], ATTN_COLOR["cosformer"]]
    alphas = [0.95, 0.55, 0.95, 0.55]
    hatches = [None, "///", None, "///"]

    x = np.arange(len(labels))
    means = [stats[k][4] for k in keys]
    stds = [stats[k][5] for k in keys]

    for i, (xi, mean, std, color, alpha, hatch) in enumerate(zip(x, means, stds, colors, alphas, hatches)):
        ax.bar(xi, mean, 0.6, yerr=std, capsize=5, color=color, alpha=alpha, hatch=hatch,
               edgecolor="white" if hatch else color, linewidth=0.8,
               error_kw=dict(ecolor=INK_SECONDARY, elinewidth=1.3, capthick=1.3), zorder=3)
        ax.text(xi, mean + std + 0.015, f"{mean:.4f}\n±{std:.4f}", ha="center", va="bottom",
                 fontsize=8.5, color=INK_PRIMARY)
        pure_ce_vals = stats[keys[i]][3]
        jitter = np.linspace(-0.06, 0.06, len(pure_ce_vals))
        ax.scatter([xi + j for j in jitter], pure_ce_vals, color=INK_PRIMARY, s=22, zorder=5,
                    edgecolor="white", linewidth=0.7)

    ax.axhline(RANDOM_BASELINE, color="#d03b3b", linestyle=":", linewidth=1.2,
                label=f"random chance ({RANDOM_BASELINE})", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel("Pure cross-entropy loss (fixed metric, re-evaluated)", fontsize=10, color=INK_SECONDARY)
    ax.set_title("CE vs. CE+NTL training: does NTL change next-token prediction quality?",
                  fontsize=12, color=INK_PRIMARY, pad=12)
    ax.set_ylim(4.75, 5.35)
    ax.yaxis.grid(True, linestyle=":", color=GRIDLINE, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE_LINE)
    ax.legend(fontsize=8.5, loc="upper right", frameon=False)

    caption = ("Solid = CE-trained, hatched = CE+NTL-trained; both re-evaluated under the SAME pure-CE metric (not each\n"
               "checkpoint's own training loss). Black dots = individual seeds. n=3 seeds, 100 epochs, MPS.")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.3, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(os.path.join(PLOTS_DIR, "ntl_vs_ce_pure_ce_comparison.png"), dpi=150, facecolor=SURFACE)
    plt.close()
    print("Wrote ntl_vs_ce_pure_ce_comparison.png")

    return stats


if __name__ == "__main__":
    main()
