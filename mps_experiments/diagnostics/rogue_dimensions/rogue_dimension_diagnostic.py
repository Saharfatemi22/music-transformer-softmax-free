"""
mps_experiments/diagnostics/rogue_dimensions/rogue_dimension_diagnostic.py
────────────────────────────────────────────────────────────────────────
Read-only rogue-dimension diagnostic for the 9 trained 50-epoch MPS checkpoints.

Question (Daniel Armani): do a small number of embedding dimensions dominate
variance / cosine similarity / Euclidean distance in THIS project's trained
music-token representations — the way Timkey & van Schijndel (EMNLP 2021)
found for BERT/GPT-2/RoBERTa/XLNet text embeddings — or is that a text-only
phenomenon that does not show up in symbolic-music embeddings?

What this script does:
  1. Loads each mps_experiments/checkpoints/<run>/best.pt   READ-ONLY
     (map_location='cpu', model.eval(), torch.no_grad() everywhere).
  2. Runs a fixed sample of MAESTRO validation tokens through each model
     purely for inference — no optimizer, no backward pass, nothing written
     back to any checkpoint.
  3. Captures, via non-invasive forward hooks (no source files edited):
       - token embedding matrix          (vocab_size, d_model)
       - hidden state after each layer + final pre-head LayerNorm
       - Q and K linear projections, per layer, pre-head-split (d_model)
         AND per-head (d_head) -- attention literally computes its
         distance/cosine score in the d_head subspace, so that's the
         operationally relevant granularity for Gaussian and Cosine attention.
  4. For every (model, representation, layer, granularity) computes:
       - per-dimension variance, and the top-1/3/5 share of total variance
       - per-dimension contribution to cosine similarity (avg over sampled
         token-vector pairs), and its top-1/3/5 share
       - per-dimension contribution to squared Euclidean distance, and its
         top-1/3/5 share
     each compared against the uniform-share baseline (1/d, 3/d, 5/d) that
     would hold if no dimension were special.
  5. Writes CSV tables to results/ and PNG plots to plots/.

Does NOT modify model/, attention/, data/, utils/train.py, configs/, or any
existing checkpoint / result .txt / plot / log file. Safe to re-run.

Run from anywhere:
  /opt/anaconda3/bin/python mps_experiments/diagnostics/rogue_dimensions/rogue_dimension_diagnostic.py
"""

import os
import sys
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Path setup (script is 3 levels below the project root) ────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from model.transformer import build_model          # noqa: E402
from data.midi_dataset import MusicDataset          # noqa: E402

CKPT_DIR = os.path.join(PROJECT_ROOT, "mps_experiments", "checkpoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "mps_experiments", "diagnostics", "rogue_dimensions")
RESULTS_DIR = os.path.join(OUT_DIR, "results")
PLOTS_DIR = os.path.join(OUT_DIR, "plots")

RUNS = [
    "gaussian_ce", "gaussian_mse", "gaussian_ntl",
    "cosine_ce",   "cosine_mse",   "cosine_ntl",
    "cosformer_ce", "cosformer_mse", "cosformer_ntl",
]

N_PAIRS = 3000       # sampled pairs for cosine / Euclidean estimates
SEED = 0
MAX_VAL_CHUNKS = 40  # ~40*512 = 20,480 token vectors sampled per model
SEQ_LEN = 512
BATCH_SIZE = 8

# ── dataviz skill palette (static PNG, light mode) ─────────────────────────────
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

ATTN_COLOR = {
    "gaussian": "#2a78d6",    # categorical slot 1 (blue)
    "cosine": "#1baf7a",      # categorical slot 2 (aqua)
    "cosformer": "#eb6834",   # categorical slot 8 (orange)
}
SEQ_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE_RAMP)


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_model(run_name):
    ckpt_path = os.path.join(CKPT_DIR, run_name, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


# ── Sample validation data (same val split used during training) ──────────────

def load_sample_batches():
    ds = MusicDataset(
        dataset="maestro",
        data_dir=os.path.join(PROJECT_ROOT, "data", "maestro-v3.0.0"),
        seq_len=SEQ_LEN,
        split="val",
        max_chunks=MAX_VAL_CHUNKS,
    )
    xs = [ds[i][0] for i in range(len(ds))]
    batches = [
        torch.stack(xs[i:i + BATCH_SIZE])
        for i in range(0, len(xs), BATCH_SIZE)
    ]
    return batches


# ── Activation capture via forward hooks (no source files modified) ───────────

def collect_activations(model, batches):
    store = {}

    def make_hook(name):
        def hook(module, inp, out):
            store.setdefault(name, []).append(out.detach().reshape(-1, out.shape[-1]))
        return hook

    handles = []
    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(make_hook(f"hidden_layer{i}")))
        handles.append(layer.attn.W_Q.register_forward_hook(make_hook(f"Q_layer{i}")))
        handles.append(layer.attn.W_K.register_forward_hook(make_hook(f"K_layer{i}")))
    handles.append(model.norm.register_forward_hook(make_hook("hidden_final")))

    with torch.no_grad():
        for batch in batches:
            model(batch)

    for h in handles:
        h.remove()

    return {name: torch.cat(tensors, dim=0) for name, tensors in store.items()}


# ── Rogue-dimension metrics ─────────────────────────────────────────────────────

def sample_pairs(n, n_pairs, seed):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    same = i == j
    j[same] = (j[same] + 1) % n
    return i, j


def variance_profile(X):
    """X: (N, d). Returns (var per dim, top1/3/5 share of total variance)."""
    var = X.var(dim=0, unbiased=False).numpy()
    total = var.sum()
    order = np.argsort(-var)
    frac = lambda k: float(var[order[:k]].sum() / total) if total > 0 else float("nan")
    return var, frac(1), frac(3), frac(5)


def cosine_profile(X, n_pairs=N_PAIRS, seed=SEED):
    """Per-dimension contribution to cosine similarity across sampled pairs."""
    n = X.shape[0]
    i, j = sample_pairs(n, n_pairs, seed)
    Xi, Xj = X[i], X[j]
    ni = Xi.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    nj = Xj.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    contrib = (Xi / ni) * (Xj / nj)              # rows sum to cosine similarity
    mean_abs = contrib.abs().mean(dim=0).numpy()
    mean_cos_sim = float(contrib.sum(dim=-1).mean().item())
    total = mean_abs.sum()
    order = np.argsort(-mean_abs)
    frac = lambda k: float(mean_abs[order[:k]].sum() / total) if total > 0 else float("nan")
    return mean_abs, frac(1), frac(3), frac(5), mean_cos_sim


def euclidean_profile(X, n_pairs=N_PAIRS, seed=SEED):
    """Per-dimension contribution to squared Euclidean distance across sampled pairs."""
    n = X.shape[0]
    i, j = sample_pairs(n, n_pairs, seed)
    diff = X[i] - X[j]
    sq = (diff ** 2)
    mean_sq = sq.mean(dim=0).numpy()
    mean_dist = float(sq.sum(dim=-1).sqrt().mean().item())
    total = mean_sq.sum()
    order = np.argsort(-mean_sq)
    frac = lambda k: float(mean_sq[order[:k]].sum() / total) if total > 0 else float("nan")
    return mean_sq, frac(1), frac(3), frac(5), mean_dist


def per_head_average(X, n_heads, d_head):
    """X: (N, d_model) contiguous per-head layout -> list of n_heads tensors (N, d_head)."""
    n = X.shape[0]
    Xh = X.view(n, n_heads, d_head)
    return [Xh[:, h, :] for h in range(n_heads)]


# ── Main per-model analysis ─────────────────────────────────────────────────────

def analyze_model(run_name, model, cfg, batches):
    attn_type, loss_type = run_name.rsplit("_", 1)
    n_heads = cfg.get("n_heads", 4)
    d_model = cfg.get("d_model", 64)
    d_head = d_model // n_heads

    acts = collect_activations(model, batches)

    representations = []  # (repr_name, layer, granularity, tensor_or_list)

    representations.append(("embedding", None, "full", model.embed.weight.detach()))
    representations.append(("hidden_layer0", 0, "full", acts["hidden_layer0"]))
    representations.append(("hidden_layer1", 1, "full", acts["hidden_layer1"]))
    representations.append(("hidden_final", None, "full", acts["hidden_final"]))

    for layer_idx in range(len(model.layers)):
        for qk in ["Q", "K"]:
            key = f"{qk}_layer{layer_idx}"
            full_tensor = acts[key]
            representations.append((qk, layer_idx, "full", full_tensor))
            heads = per_head_average(full_tensor, n_heads, d_head)
            representations.append((qk, layer_idx, "per_head", heads))

    summary_rows = []
    raw_var_rows = []
    raw_cos_rows = []
    raw_euc_rows = []

    for repr_name, layer, granularity, tensor in representations:
        if granularity == "full":
            X = tensor
            d = X.shape[1]
            var, v1, v3, v5 = variance_profile(X)
            cos, c1, c3, c5, mean_cos = cosine_profile(X)
            euc, e1, e3, e5, mean_euc = euclidean_profile(X)

            for dim in range(d):
                raw_var_rows.append([run_name, attn_type, loss_type, repr_name, layer, granularity,
                                      dim, float(var[dim]), float(var[dim] / var.sum())])
                raw_cos_rows.append([run_name, attn_type, loss_type, repr_name, layer, granularity,
                                      dim, float(cos[dim]), float(cos[dim] / cos.sum())])
                raw_euc_rows.append([run_name, attn_type, loss_type, repr_name, layer, granularity,
                                      dim, float(euc[dim]), float(euc[dim] / euc.sum())])

            summary_rows.append([
                run_name, attn_type, loss_type, repr_name, layer, granularity, d,
                v1, v3, v5, round(1 / d, 4), round(3 / d, 4), round(5 / d, 4),
                c1, c3, c5, mean_cos,
                e1, e3, e5, mean_euc,
            ])
        else:  # per_head: average concentration stats across heads
            head_tensors = tensor
            d = head_tensors[0].shape[1]
            v1s, v3s, v5s = [], [], []
            c1s, c3s, c5s, mean_coss = [], [], [], []
            e1s, e3s, e5s, mean_eucs = [], [], [], []
            var_acc = np.zeros(d)
            cos_acc = np.zeros(d)
            euc_acc = np.zeros(d)

            for X in head_tensors:
                var, v1, v3, v5 = variance_profile(X)
                cos, c1, c3, c5, mean_cos = cosine_profile(X)
                euc, e1, e3, e5, mean_euc = euclidean_profile(X)
                v1s.append(v1); v3s.append(v3); v5s.append(v5)
                c1s.append(c1); c3s.append(c3); c5s.append(c5); mean_coss.append(mean_cos)
                e1s.append(e1); e3s.append(e3); e5s.append(e5); mean_eucs.append(mean_euc)
                var_acc += var / var.sum()
                cos_acc += cos / cos.sum()
                euc_acc += euc / euc.sum()

            n_heads_local = len(head_tensors)
            var_acc /= n_heads_local
            cos_acc /= n_heads_local
            euc_acc /= n_heads_local

            for dim in range(d):
                raw_var_rows.append([run_name, attn_type, loss_type, repr_name, layer, granularity,
                                      dim, float("nan"), float(var_acc[dim])])
                raw_cos_rows.append([run_name, attn_type, loss_type, repr_name, layer, granularity,
                                      dim, float("nan"), float(cos_acc[dim])])
                raw_euc_rows.append([run_name, attn_type, loss_type, repr_name, layer, granularity,
                                      dim, float("nan"), float(euc_acc[dim])])

            summary_rows.append([
                run_name, attn_type, loss_type, repr_name, layer, granularity, d,
                float(np.mean(v1s)), float(np.mean(v3s)), float(np.mean(v5s)),
                round(1 / d, 4), round(3 / d, 4), round(5 / d, 4),
                float(np.mean(c1s)), float(np.mean(c3s)), float(np.mean(c5s)), float(np.mean(mean_coss)),
                float(np.mean(e1s)), float(np.mean(e3s)), float(np.mean(e5s)), float(np.mean(mean_eucs)),
            ])

    return summary_rows, raw_var_rows, raw_cos_rows, raw_euc_rows


# ── CSV writers ─────────────────────────────────────────────────────────────────

SUMMARY_HEADER = [
    "model", "attention_type", "loss_type", "representation", "layer", "granularity", "n_dims",
    "var_top1_frac", "var_top3_frac", "var_top5_frac",
    "var_uniform_top1", "var_uniform_top3", "var_uniform_top5",
    "cos_top1_frac", "cos_top3_frac", "cos_top5_frac", "mean_cosine_sim",
    "euc_top1_frac", "euc_top3_frac", "euc_top5_frac", "mean_euclidean_dist",
]
RAW_HEADER = [
    "model", "attention_type", "loss_type", "representation", "layer", "granularity",
    "dim_index", "raw_value", "frac_of_total",
]


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def filter_rows(rows, repr_filter):
    return [r for r in rows if r[3] in repr_filter]


# ── Plotting ────────────────────────────────────────────────────────────────────

HEATMAP_REPR_ORDER = [
    ("embedding", None), ("hidden_layer0", 0), ("hidden_layer1", 1), ("hidden_final", None),
    ("Q", 0), ("K", 0), ("Q", 1), ("K", 1),
]
HEATMAP_LABELS = ["embed", "hidL0", "hidL1", "hidFin", "Q-L0", "K-L0", "Q-L1", "K-L1"]


def build_heatmap_matrix(summary_rows, metric_col_idx):
    matrix = np.full((len(RUNS), len(HEATMAP_REPR_ORDER)), np.nan)
    lookup = {}
    for row in summary_rows:
        model, repr_name, layer, granularity = row[0], row[3], row[4], row[5]
        if granularity != "full":
            continue
        lookup[(model, repr_name, layer)] = row
    for ri, run_name in enumerate(RUNS):
        for ci, (repr_name, layer) in enumerate(HEATMAP_REPR_ORDER):
            row = lookup.get((run_name, repr_name, layer))
            if row is not None:
                matrix[ri, ci] = row[metric_col_idx]
    return matrix


def plot_concentration_heatmap(matrix, title, out_path, value_fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(matrix, cmap=SEQ_CMAP, vmin=0, vmax=np.nanmax(matrix) * 1.05, aspect="auto")

    ax.set_xticks(range(len(HEATMAP_LABELS)))
    ax.set_xticklabels(HEATMAP_LABELS, fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(range(len(RUNS)))
    ax.set_yticklabels(RUNS, fontsize=9, color=INK_SECONDARY)

    for ri in range(matrix.shape[0]):
        for ci in range(matrix.shape[1]):
            val = matrix[ri, ci]
            if np.isnan(val):
                continue
            text_color = INK_PRIMARY if val < 0.6 * np.nanmax(matrix) else "#ffffff"
            ax.text(ci, ri, value_fmt.format(val), ha="center", va="center",
                     fontsize=7.5, color=text_color)

    ax.set_title(title, fontsize=12, color=INK_PRIMARY, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY)
    cbar.outline.set_visible(False)

    caption = "Rows = trained model (attention_loss). Columns = representation. Cell = top-3 share of total (full d_model granularity)."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color=INK_MUTED)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


def plot_per_dim_bars(run_name, raw_var_rows, raw_cos_rows, raw_euc_rows, out_path):
    attn_type = run_name.split("_")[0]
    color = ATTN_COLOR[attn_type]

    reps = [("embedding", None, "full"), ("Q", 0, "full"), ("K", 0, "full")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)

    for ax, (repr_name, layer, gran) in zip(axes, reps):
        rows = [r for r in raw_var_rows
                if r[0] == run_name and r[3] == repr_name and r[4] == layer and r[5] == gran]
        rows.sort(key=lambda r: r[6])
        fracs = [r[8] for r in rows]
        d = len(fracs)
        ax.bar(range(d), fracs, color=color, alpha=0.88, width=0.85, zorder=3)
        ax.axhline(1 / d, color=BASELINE, linestyle="--", linewidth=1.0, zorder=2,
                    label=f"uniform (1/{d})")
        ax.set_title(f"{repr_name}" + (f" L{layer}" if layer is not None else ""),
                      fontsize=10, color=INK_PRIMARY)
        ax.set_xlabel("dimension index", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylabel("fraction of total variance", fontsize=8.5, color=INK_SECONDARY)
        ax.tick_params(labelsize=7.5, colors=INK_SECONDARY)
        ax.yaxis.grid(True, linestyle=":", color=GRIDLINE, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(BASELINE)
        ax.legend(fontsize=7, loc="upper right", frameon=False)

    fig.suptitle(f"Per-dimension variance share — {run_name}", fontsize=12, color=INK_PRIMARY)
    caption = "Dashed line = uniform-share baseline if no dimension dominated. Bars above the line are over-represented."
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7.5, color=INK_MUTED)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading shared validation sample (read-only, inference only)...")
    batches = load_sample_batches()
    n_tokens = sum(b.numel() for b in batches)
    print(f"  {len(batches)} batches, {n_tokens} token vectors total")

    all_summary, all_var, all_cos, all_euc = [], [], [], []

    for run_name in RUNS:
        print(f"Analyzing {run_name} ...")
        model, cfg = load_model(run_name)
        summary_rows, var_rows, cos_rows, euc_rows = analyze_model(run_name, model, cfg, batches)
        all_summary.extend(summary_rows)
        all_var.extend(var_rows)
        all_cos.extend(cos_rows)
        all_euc.extend(euc_rows)

    write_csv(os.path.join(RESULTS_DIR, "rogue_dimension_summary.csv"), SUMMARY_HEADER, all_summary)
    write_csv(os.path.join(RESULTS_DIR, "embedding_variance.csv"), RAW_HEADER,
              filter_rows(all_var, {"embedding"}))
    write_csv(os.path.join(RESULTS_DIR, "hidden_state_variance.csv"), RAW_HEADER,
              filter_rows(all_var, {"hidden_layer0", "hidden_layer1", "hidden_final"}))
    write_csv(os.path.join(RESULTS_DIR, "qk_variance.csv"), RAW_HEADER,
              filter_rows(all_var, {"Q", "K"}))
    write_csv(os.path.join(RESULTS_DIR, "cosine_contribution.csv"), RAW_HEADER, all_cos)
    write_csv(os.path.join(RESULTS_DIR, "euclidean_contribution.csv"), RAW_HEADER, all_euc)
    print(f"Wrote 6 CSV files to {RESULTS_DIR}")

    # Heatmaps: column indices in SUMMARY_HEADER -> var_top3=8, cos_top3=14, euc_top3=18
    var_matrix = build_heatmap_matrix(all_summary, metric_col_idx=8)
    cos_matrix = build_heatmap_matrix(all_summary, metric_col_idx=14)
    euc_matrix = build_heatmap_matrix(all_summary, metric_col_idx=18)

    plot_concentration_heatmap(var_matrix, "Variance concentration (top-3 dims / total, full d_model)",
                                os.path.join(PLOTS_DIR, "variance_concentration_heatmap.png"))
    plot_concentration_heatmap(cos_matrix, "Cosine-similarity contribution concentration (top-3 dims)",
                                os.path.join(PLOTS_DIR, "cosine_concentration_heatmap.png"))
    plot_concentration_heatmap(euc_matrix, "Euclidean-distance contribution concentration (top-3 dims)",
                                os.path.join(PLOTS_DIR, "euclidean_concentration_heatmap.png"))
    print(f"Wrote 3 heatmap plots to {PLOTS_DIR}")

    for run_name in ["gaussian_ce", "cosine_ce", "cosformer_ce"]:
        out_path = os.path.join(PLOTS_DIR, f"per_dim_variance_{run_name}.png")
        plot_per_dim_bars(run_name, all_var, all_cos, all_euc, out_path)
    print(f"Wrote 3 per-dimension bar plots to {PLOTS_DIR}")

    print("\nDone. No existing checkpoint, log, result, or plot file was modified.")


if __name__ == "__main__":
    main()
