"""
make_plots_mps.py
─────────────────
Generate the three paper figures from the 9-run MPS experiment matrix.
Run from the music_transformer/ directory:
  /opt/anaconda3/bin/python mps_experiments/make_plots_mps.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data (MPS, 50 epochs, 1000 train / 200 val chunks) ────────────────────────

attention_types = ["Gaussian", "Cosine", "CosFormer"]

# Final val loss (epoch 50)
ce_vals  = [5.2457, 5.1474, 5.1045]
ntl_vals = [5.1211, 5.1567, 5.1753]
mse_vals = [2.5690, 2.6777, 2.5032]   # different scale

# Avg seconds per epoch (excluding first-epoch MPS warmup)
speed = {
    "Gaussian\nCE":       17.5,
    "Gaussian\nMSE":      16.5,
    "Gaussian\nCE+NTL":   12.9,
    "Cosine\nCE":          8.2,
    "Cosine\nMSE":         7.7,
    "Cosine\nCE+NTL":      6.4,
    "CosFormer\nCE":       6.6,
    "CosFormer\nMSE":      5.2,
    "CosFormer\nCE+NTL":   5.1,
}

RANDOM_BASELINE = 5.2983

# ── Shared style ──────────────────────────────────────────────────────────────

COLORS = {
    "CE":     "#4C72B0",   # blue
    "CE+NTL": "#DD8452",   # orange
    "MSE":    "#55A868",   # green
}
BAR_W = 0.32
DPI   = 150


# ── Plot 1: CE vs CE+NTL grouped bar chart ────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(attention_types))

bars_ce  = ax.bar(x - BAR_W / 2, ce_vals,  BAR_W,
                  label="CE (baseline)", color=COLORS["CE"],     alpha=0.88, zorder=3)
bars_ntl = ax.bar(x + BAR_W / 2, ntl_vals, BAR_W,
                  label="CE + NTL",      color=COLORS["CE+NTL"], alpha=0.88, zorder=3)

# Random-chance baseline
ax.axhline(RANDOM_BASELINE, color="red", linestyle="--", linewidth=1.4,
           label=f"Random chance  (log 196 = {RANDOM_BASELINE})", zorder=4)

# Value labels on top of bars
for bar in list(bars_ce) + list(bars_ntl):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=8.5)

# Annotate best result
ax.annotate("best result\n5.1045",
            xy=(x[2] - BAR_W / 2, ce_vals[2]),
            xytext=(x[2] - BAR_W / 2 - 0.55, ce_vals[2] - 0.025),
            fontsize=8, color="darkblue",
            arrowprops=dict(arrowstyle="->", color="darkblue", lw=1.2))

ax.set_xticks(x)
ax.set_xticklabels(attention_types, fontsize=11)
ax.set_ylabel("Validation Loss (epoch 50)", fontsize=11)
ax.set_title("CE vs CE+NTL Loss across Attention Types on MAESTRO  [MPS · 50 epochs]",
             fontsize=11, pad=10)
ax.set_ylim(5.04, 5.36)
ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="upper right")

# Shaded region below random chance
ax.axhspan(5.04, RANDOM_BASELINE, alpha=0.06, color="green", zorder=0)
ax.text(2.48, RANDOM_BASELINE - 0.012, "all bars beat\nrandom chance",
        fontsize=7.5, color="darkgreen", ha="right", va="top")

caption = ("d_model=64 | n_layers=2 | n_heads=4 | batch=8 | "
           "1000 train chunks | 200 val chunks | 50 epochs | MAESTRO v3.0.0 | Apple MPS")
fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="gray")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("mps_experiments/plots/plot1_ce_vs_ntl.png", dpi=DPI)
plt.close()
print("Saved mps_experiments/plots/plot1_ce_vs_ntl.png")


# ── Plot 2: MSE results only ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))

bars_mse = ax.bar(attention_types, mse_vals, 0.45,
                  color=COLORS["MSE"], alpha=0.88, zorder=3)

for bar in bars_mse:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=10.5, fontweight="bold")

ax.set_ylabel("Validation Loss  (CE_structural + MSE_ordinal) / 2", fontsize=11)
ax.set_title("MSE Loss Results (separate scale)  [MPS · 50 epochs]", fontsize=11, pad=10)
ax.set_ylim(2.40, 2.80)
ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelsize=11)

# Prominent scale warning
ax.text(0.5, 0.92,
        "MSE operates on normalised [0, 1] scale\n"
        "— not comparable to CE values above",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9.5, color="#8B0000",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff0f0",
                  edgecolor="#cc0000", linewidth=1.2))

caption = ("Token values normalised: pitch 0–127→[0,1]  "
           "duration 1–16→[0,1]  velocity 0–31→[0,1]\n"
           "MSE applied to ordinal tokens only; CE used for structural tokens  |  Apple MPS")
fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="gray")

plt.subplots_adjust(top=0.88, bottom=0.12)
plt.savefig("mps_experiments/plots/plot2_mse_separate.png", dpi=DPI)
plt.close()
print("Saved mps_experiments/plots/plot2_mse_separate.png")


# ── Plot 3: Training speed horizontal bar chart ───────────────────────────────

labels = list(speed.keys())
values = list(speed.values())

bar_colors = (
    [COLORS["CE"], COLORS["MSE"], COLORS["CE+NTL"]] * 3
)

fig, ax = plt.subplots(figsize=(8, 6))

y_pos = np.arange(len(labels))
hbars = ax.barh(y_pos, values, 0.6, color=bar_colors, alpha=0.88, zorder=3)

for bar, val in zip(hbars, values):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}s", va="center", fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Seconds per Epoch (average, excl. first-epoch warmup)", fontsize=10)
ax.set_title("Training Speed: Seconds per Epoch  [MPS · Apple Silicon]",
             fontsize=11, pad=10)
ax.set_xlim(0, 24)
ax.xaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.invert_yaxis()

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLORS["CE"],     alpha=0.88, label="CE"),
    Patch(facecolor=COLORS["MSE"],    alpha=0.88, label="MSE"),
    Patch(facecolor=COLORS["CE+NTL"], alpha=0.88, label="CE+NTL"),
]
ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

caption = "All runs: 50 epochs | 1000 train chunks | d_model=64 | n_layers=2 | Apple MPS"
fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color="gray")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("mps_experiments/plots/plot3_speed.png", dpi=DPI)
plt.close()
print("Saved mps_experiments/plots/plot3_speed.png")

print("\nAll 3 MPS plots saved.")
