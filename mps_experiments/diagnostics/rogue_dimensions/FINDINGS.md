# Rogue-Dimension Diagnostic — Findings

**Question asked (Daniel Armani):** do rogue dimensions appear in symbolic-music Transformer representations, or is this mainly a text-embedding issue?

**Method:** all 9 trained 50-epoch MPS checkpoints (`mps_experiments/checkpoints/<attention>_<loss>/best.pt`) were loaded read-only and run once, in inference mode, over a fixed sample of 20,480 MAESTRO validation tokens (40 chunks × 512 tokens, same val split used in training). Token embeddings, per-layer hidden states, and Q/K projections (both the full 64-dim vector and the 16-dim per-head subspace attention actually computes in) were captured via forward hooks. For each representation we measured, following Timkey & van Schijndel (EMNLP 2021): per-dimension variance, per-dimension contribution to cosine similarity, and per-dimension contribution to squared Euclidean distance — each compared against the uniform-share baseline that would hold if no dimension were special (`1/d`, `3/d`, `5/d` for the top-1/3/5 shares). No checkpoint, log, or existing result file was modified; see `HISTORY.md` for the full audit trail.

---

## Headline answer

**Rogue dimensions in the severe, text-transformer sense do not appear in this project's music representations.** Token embeddings are essentially isotropic and dimensionally uniform. Where concentration does appear — mainly in hidden states and in the Q/K per-head subspace — it is real but mild-to-moderate (roughly 2–5× the uniform baseline), never approaching the extreme single-dimension domination reported for BERT/GPT-2/XLNet (where Timkey & van Schijndel report up to 99% of cosine similarity carried by one dimension out of hundreds — a ~700× departure from uniform for d=768). The largest concentration found anywhere in this study is about **4.8× uniform**, in a 16-dimensional per-head subspace, for one model/layer/head-averaged cell out of the 9×8 grid tested. That is a different regime, not the same phenomenon at smaller scale.

This suggests rogue dimensionality, as originally described, is at least partly a property of the **scale and training regime of large pretrained text encoders** (hundreds of millions of tokens, d=768+, contextual language modeling) rather than an inevitable property of distance/cosine-based attention itself. It does **not** rule out rogue dimensions emerging at larger model/data scale in this project — see "Caveats" below.

---

## Results by representation type

Values below are the **average top-3 concentration ratio to the uniform baseline**, averaged across all 9 trained models (gaussian/cosine/cosformer × ce/mse/ntl). A ratio of 1.0× means dimensions are exactly as uniform as random chance predicts; higher means more concentrated.

| Representation | Variance ratio | Cosine-similarity ratio | Mean cosine similarity (anisotropy) |
|---|---|---|---|
| Token embedding (196×64 matrix) | **1.13×** | **1.18×** | **≈ 0.000** |
| Hidden state, layer 0 | 2.43× | 3.93× | 0.502 |
| Hidden state, layer 1 | 2.41× | 4.12× | 0.619 |
| Hidden state, final norm | 2.30× | 3.97× | 0.602 |
| Q projection (full d=64) | 1.86× | 3.14× | 0.387 |
| K projection (full d=64) | 1.84× | 3.04× | 0.386 |

**Single most concentrated cell found anywhere** (out of 9 models × 8 representations × 3 metrics): `cosformer_ntl`, Q projection, layer 1, **per-head** (d=16) granularity — top-1 dimension carries 30% of the average cosine-similarity signal within its head, against a 6.25% uniform baseline (**4.8×**). The next-highest cells are `gaussian_mse` K/Q layer 1 per-head (3.7×/3.4×) and `cosformer_ce` Q/K layer 1 per-head (3.4×/3.3×). Full numeric detail is in `results/rogue_dimension_summary.csv`.

## Two distinct phenomena, worth separating

Timkey & van Schijndel's paper actually documents two related but distinct pathologies, and this diagnostic can tell them apart:

1. **Anisotropy** — the average cosine similarity between two *random* token vectors is high (representations all point in roughly the same direction in space, wasting the representational volume).
2. **Rogue dimensionality** — that anisotropy (when present) is disproportionately caused by a tiny number of dimensions, rather than being spread across the whole vector.

In this project:
- **Token embeddings show neither.** Mean cosine similarity between random embedding pairs is ≈ 0 (fully isotropic — the 196-token vocabulary occupies its embedding space roughly like a random Gaussian would), and no dimension is over-represented beyond noise.
- **Hidden states show anisotropy but NOT concentrated rogue dimensionality.** Mean cosine similarity between random token hidden states is high — 0.50 to 0.78 depending on layer/model — meaning the model's internal representations do collapse toward a shared direction after a few Transformer blocks (this is a real, known Transformer phenomenon, unrelated to the attention mechanism choice). But that collapse is spread across roughly 15–25 of the 64 dimensions, not concentrated in 1–3 — so it would **not** show up as "rogue dimensions" under Timkey & van Schijndel's original diagnostic, even though the underlying anisotropy is present.
- **Q/K per-head subspaces show the mildest version of both**, but this is exactly the space Gaussian and Cosine attention compute their kernel in, so even a 3–5× concentration there is the most actionable finding of this diagnostic (see recommendation below).

## Attention-type comparison (Daniel's "focus on Gaussian and Cosine")

No systematic difference in concentration was found between Gaussian, Cosine, and CosFormer attention types in this diagnostic — the per-model rows in `results/rogue_dimension_summary.csv` and the three heatmaps (`plots/*_concentration_heatmap.png`) show all 9 models landing in a similar band for every representation. This is a meaningful negative result: **whatever mild dimensional concentration exists is not specific to the distance-based kernels (Gaussian, Cosine) versus the ReLU+cosine-reweight kernel (CosFormer) or versus loss function** — it looks like a generic property of this small Transformer trained on this data, not a symptom of removing softmax.

## Caveats — read before citing this in the paper

- **Model and data scale are small.** 112K-parameter models, d_model=64, trained on 1,000 chunks. The rogue-dimension literature studies models 1000×+ larger trained on much more data. This diagnostic answers "does it appear at the scale we've actually trained so far," not "is it impossible in principle for this architecture at any scale."
- **Single validation sample, single measurement pass.** No confidence intervals or repeated sampling across seeds. The `n_pairs=3000` random-pair sampling for cosine/Euclidean estimates is stable (verified by construction) but was not itself bootstrapped.
- **Q/K captured pre-normalization/pre-ReLU** — i.e., the raw linear-projection output, which is exactly what Gaussian attention's kernel and Cosine attention's `F.normalize()` operate on directly, and what CosFormer's `ReLU()` operates on. This is the correct object to analyze for "does the representation feeding the kernel have rogue dimensions," consistent with how the original paper analyzes raw embeddings feeding cosine similarity.

## Recommendation

Given the mild-but-real concentration found in Q/K per-head subspaces (2.4×–4.8× uniform), the standardization ablation already specced in `CLAUDE.md`'s Phase 2 Plan (Paper 1: per-token mean/std standardization of Q/K before the distance/cosine kernel) is now backed by an actual measurement rather than an assumption — worth implementing and testing whether it changes downstream loss, even though the pre-standardization concentration is far below where it caused problems in text transformers. If it makes no measurable difference (plausible, given how mild the concentration is here), that itself is a citable, paper-worthy negative result answering Daniel's question directly: symbolic music embeddings at this scale do not exhibit textual embedding spaces' rogue-dimension pathology.

## Files

- `results/rogue_dimension_summary.csv` — master table, one row per (model, representation, layer, granularity): all variance/cosine/Euclidean top-1/3/5 concentration ratios and uniform baselines.
- `results/embedding_variance.csv`, `hidden_state_variance.csv`, `qk_variance.csv` — per-dimension raw variance values.
- `results/cosine_contribution.csv`, `euclidean_contribution.csv` — per-dimension raw contribution values, all representations.
- `plots/variance_concentration_heatmap.png`, `cosine_concentration_heatmap.png`, `euclidean_concentration_heatmap.png` — 9 models × 8 representations, top-3 concentration.
- `plots/per_dim_variance_{gaussian,cosine,cosformer}_ce.png` — detailed 64-bar profiles for embedding/Q/K on the three CE baseline models, visually confirming no single-dimension spikes.
- `rogue_dimension_diagnostic.py` — the analysis script; safe to re-run, does not modify anything outside this folder.
