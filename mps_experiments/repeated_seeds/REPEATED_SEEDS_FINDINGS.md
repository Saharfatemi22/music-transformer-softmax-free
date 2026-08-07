# Repeated-Seed Validation — Findings

**Purpose:** every prior comparison in this project (attention types, loss types, the Q/K standardization ablation) rested on a single run per configuration. This experiment re-runs the five most important comparisons with 3 seeds each (0, 1, 2) to check whether the previously reported differences survive ordinary run-to-run noise from weight initialization, dropout, and batch order.

**Method:** added a `--seed` flag to `train.py` (seeds `random`, `numpy`, and `torch`, verified to produce identical model init for the same seed and different init across seeds). Ran 15 training jobs — 5 configurations × 3 seeds — sequentially on MPS, identical settings to every other MPS experiment in this project: 50 epochs, 1000 train chunks, 200 val chunks, `d_model=64, n_layers=2, n_heads=4, d_ff=256, batch_size=8`. Note that the train/val chunk *sets* are identical across seeds (chunk selection is deterministic, capped by `max_chunks`, not randomly sampled) — only weight init, dropout, and training batch order differ between seeds. All 15 runs completed successfully (exit 0), all monotonic with no divergence, and in every run the final (epoch 50) validation loss was also the best validation loss achieved.

---

## Mean and standard deviation (n=3 seeds, final validation loss)

| Configuration | seed 0 | seed 1 | seed 2 | **Mean** | **Std dev** |
|---|---|---|---|---|---|
| Gaussian + CE (baseline) | 5.2156 | 5.2032 | 5.0553 | **5.1580** | **0.0892** |
| Gaussian + CE + standardize_qk | 5.2122 | 5.2020 | 5.0545 | **5.1563** | **0.0882** |
| Cosine + CE (baseline) | 5.2637 | 5.2662 | 5.3169 | **5.2823** | **0.0300** |
| Cosine + CE + standardize_qk | 5.2596 | 5.2758 | 5.3188 | **5.2847** | **0.0306** |
| CosFormer + CE | 5.2019 | 5.2021 | 5.0443 | **5.1494** | **0.0911** |

Random-chance baseline: `log(196) = 5.2983`. Full per-run data: `results/repeated_seeds_summary.csv`. Mean/std table: `results/repeated_seeds_stats.csv`. Plots: `plots/mean_std_bar_comparison.png` (mean ± std, all 5 configs), `plots/all_seeds_val_curves.png` (all 3 seed curves overlaid per configuration).

---

## Headline correction: the standardization "improvement" reported earlier does not hold up

The single-seed ablation (`mps_experiments/standardization_ablation/ABLATION_FINDINGS.md`) reported standardization improving Gaussian by **−1.77%** and Cosine by **−1.08%**. Averaged over 3 seeds, **both effects collapse to noise**:

| Attention | Mean Δ (standardized − baseline) | % change | Seed-to-seed std dev |
|---|---|---|---|
| Gaussian | **−0.0018** | **−0.03%** | 0.088–0.089 |
| Cosine | **+0.0024** | **+0.05%** | 0.030–0.031 |

Both mean effects are more than an order of magnitude smaller than the seed-to-seed standard deviation. **The earlier single-seed result was very likely a lucky/unlucky draw, not a real effect of standardization** — those two original runs (from before the `--seed` flag existed) had no seed control at all, so each was effectively one arbitrary sample from the same noisy distribution measured here.

That said, the **paired, same-seed comparison** is more informative than the aggregate means and shows a genuine (if tiny) pattern:

| Seed | Gaussian: standardized − baseline | Cosine: standardized − baseline |
|---|---|---|
| 0 | −0.0034 | −0.0041 |
| 1 | −0.0012 | +0.0096 |
| 2 | −0.0007 | +0.0019 |

**Gaussian improves with standardization in all 3 seeds**, consistently, though the magnitude (0.0007–0.0034) is roughly two orders of magnitude smaller than originally reported. **Cosine shows no consistent direction** — better in 1 of 3 seeds, worse in 2 of 3 — consistent with Cosine already L2-normalizing Q/K, leaving little for explicit standardization to correct (the same interpretation offered in the original ablation, but now correctly scaled down to a much smaller and directionally inconsistent effect).

**Conclusion on standardization:** does **not** meaningfully improve either attention type at this model/data scale. Gaussian shows a small, consistent-in-direction-but-practically-negligible improvement (~0.001–0.003 in val loss); Cosine shows no reliable effect at all. This should replace, not supplement, the earlier single-seed conclusion in any paper draft.

---

## Does CosFormer + CE remain the best overall?

**By point estimate, yes** — CosFormer has the lowest mean validation loss (5.1494) of all five configurations. **But it is not decisively separated from Gaussian** at this sample size: Gaussian (baseline 5.1580, standardized 5.1563) overlaps CosFormer's mean well within one standard deviation in both directions (CosFormer std 0.091, Gaussian std 0.088–0.089). The three configurations Gaussian-baseline / Gaussian-standardized / CosFormer are statistically indistinguishable at n=3 — see the overlapping error bars in `plots/mean_std_bar_comparison.png`.

**Cosine is clearly and robustly the worst of the three attention types.** Its mean (5.28) sits above the random-chance baseline (5.2983) at the lower end of its own error bar, and the gap to Gaussian/CosFormer (~0.13) is roughly 4× larger than Cosine's own seed-to-seed std (0.03) and larger than Gaussian/CosFormer's stds too — this ranking looks robust even with only 3 seeds, unlike the Gaussian-vs-CosFormer comparison.

**Revised conclusion:** CosFormer + CE remains the best point estimate and is a reasonable default recommendation, but the paper should not claim it is *decisively* better than Gaussian + CE — the honest statement is "CosFormer and Gaussian are statistically tied for best; Cosine is clearly worse." This is a meaningfully weaker (and more defensible) claim than the single-seed study supported.

---

## A secondary observation: seed sensitivity differs by attention type

Seed 2 produced a notably better outcome (~5.05, versus ~5.20–5.22 for seeds 0/1) for **both** Gaussian and CosFormer, but not for Cosine, where seed 2 was actually its *worst* of the three seeds (5.3169). Since the same three seed values were used across all five configurations and the train/val data is identical across seeds (only init/dropout/batch order differ), this suggests Gaussian's and CosFormer's optimization landscapes respond similarly to this particular seed's initialization, while Cosine's landscape responds differently — visible directly in `plots/all_seeds_val_curves.png`, where the Gaussian and CosFormer panels show a clearly separated dotted (seed 2) curve pulling away below the other two, while Cosine's three curves stay close together throughout training with seed 2 trailing rather than leading. This is a descriptive observation, not something this experiment can explain causally (would require more seeds and probably weight-trajectory analysis to say more), but it's worth keeping in mind before treating any single-seed run from this project as representative.

---

## Caveats

- **n=3 is still a small sample.** A standard deviation computed from 3 points is itself uncertain; these numbers narrow the earlier single-run claims from "confident" to "plausible directionally, not decisive," but a paper claiming e.g. "CosFormer is best" should ideally use more seeds (5–10) before stating it as a firm result.
- **CE loss only, one model/data scale.** This does not test whether standardization or the attention-type ranking behaves differently under `mse`/`ce+ntl` loss, at larger model/data scale, or with CosFormer + standardization (not implemented, out of scope per the original standardization spec).
- **Validation loss only**, consistent with how this project has reported results throughout — no held-out test-set numbers yet.

## What was and wasn't touched

**New:** `train.py` gained a `--seed` argument and a `set_seed()` helper (seeds `random`, `numpy`, `torch`) — additive, defaults to seed 0 if unspecified, does not change behavior for any run that already specified deterministic behavior wasn't previously possible (there was no seed control before this, so no prior run's exact numeric output is reproducible or comparable to a "default seed" — this only affects new runs going forward).

**Not touched:** no existing checkpoint, training log, result file, or plot outside `mps_experiments/repeated_seeds/` was modified, including the earlier `mps_experiments/standardization_ablation/` and `mps_experiments/checkpoints/` results that this experiment re-examines.

## Files

- `checkpoints/<config>_seed{0,1,2}/` — `best.pt`, 5 periodic epoch checkpoints, `training_log.csv`, for all 15 runs.
- `logs/<config>_seed{N}.log` — full stdout per run.
- `results/repeated_seeds_summary.csv` — one row per run (15 rows).
- `results/repeated_seeds_stats.csv` — one row per configuration (5 rows): all 3 seed values, mean, std.
- `plots/mean_std_bar_comparison.png` — mean ± std dev, all 5 configurations, with random-chance line.
- `plots/all_seeds_val_curves.png` — all 3 seed curves overlaid, per configuration, small multiples.
- `make_repeated_seeds_report.py` — regenerates the CSVs/plots from the 15 training logs; safe to re-run.
- `run_all.sh` — the orchestration script that ran all 15 jobs sequentially.
