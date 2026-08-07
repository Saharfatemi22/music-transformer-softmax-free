# HISTORY.md — Running Change Log

This file tracks every change made to this repository going forward: what was done, which files were touched, why, and what the recommended next step is. Older project context (setup, bug fixes, the original 18-run experiment matrix) lives in `CLAUDE.md`; this file starts from the rogue-dimension diagnostic work.

---

## 2026-07-03 13:57 — Implemented rogue-dimension diagnostic (Daniel's ask, not yet answered before now)

**What was done:** Built and ran a read-only diagnostic that answers Daniel's explicit question — do rogue dimensions (Timkey & van Schijndel, EMNLP 2021) appear in this project's trained symbolic-music representations, or is that a text-embedding-only phenomenon? Loaded all 9 intact 50-epoch MPS checkpoints, ran a fixed sample of ~20K MAESTRO validation tokens through each in inference mode only (no training, no gradients), captured token embeddings / hidden states / Q&K projections via non-invasive forward hooks, and measured per-dimension variance, cosine-similarity contribution, and Euclidean-distance contribution against a uniform-share baseline.

**Files created:**
- `mps_experiments/diagnostics/rogue_dimensions/rogue_dimension_diagnostic.py` — the analysis script (re-runnable, does not modify anything outside its own output folder)
- `mps_experiments/diagnostics/rogue_dimensions/results/rogue_dimension_summary.csv` — master concentration-ratio table
- `mps_experiments/diagnostics/rogue_dimensions/results/embedding_variance.csv`
- `mps_experiments/diagnostics/rogue_dimensions/results/hidden_state_variance.csv`
- `mps_experiments/diagnostics/rogue_dimensions/results/qk_variance.csv`
- `mps_experiments/diagnostics/rogue_dimensions/results/cosine_contribution.csv`
- `mps_experiments/diagnostics/rogue_dimensions/results/euclidean_contribution.csv`
- `mps_experiments/diagnostics/rogue_dimensions/plots/variance_concentration_heatmap.png`
- `mps_experiments/diagnostics/rogue_dimensions/plots/cosine_concentration_heatmap.png`
- `mps_experiments/diagnostics/rogue_dimensions/plots/euclidean_concentration_heatmap.png`
- `mps_experiments/diagnostics/rogue_dimensions/plots/per_dim_variance_gaussian_ce.png`
- `mps_experiments/diagnostics/rogue_dimensions/plots/per_dim_variance_cosine_ce.png`
- `mps_experiments/diagnostics/rogue_dimensions/plots/per_dim_variance_cosformer_ce.png`
- `mps_experiments/diagnostics/rogue_dimensions/FINDINGS.md` — written interpretation
- `HISTORY.md` — this file

**Files modified:** none. Verified with `find mps_experiments/checkpoints -newer <script>` returning empty — no checkpoint, training log, existing plot, or existing result `.txt`/`.md` file was touched. No training was run.

**Why it was done:** Daniel's explicit guidance was not to assume rogue dimensions exist in music embeddings but to test experimentally and report findings. Before this, the repository had zero implementation of this test — the "Phase 2 Plan" in `CLAUDE.md` specced a standardization *fix* (Paper 1) but no diagnostic to establish whether the fix was even needed. This closes that gap using the 9 checkpoints that already survived the MPS run without needing to retrain anything.

**Headline finding (full detail in `FINDINGS.md`):** Token embeddings are essentially isotropic and dimensionally uniform (~1.1–1.2× the uniform baseline — no rogue dimensions). Hidden states show real anisotropy (mean cosine similarity between random token pairs 0.50–0.78) but it is spread across ~15–25 dimensions, not concentrated in 1–3 — so it does not qualify as "rogue" under the original diagnostic. The Q/K per-head subspace (the literal space Gaussian/Cosine attention compute their kernel in) shows the mildest-but-most-relevant concentration, up to 4.8× uniform in the single most extreme cell (`cosformer_ntl`, Q, layer 1). This is far below the ~700× departures from uniform reported for large text transformers (e.g. XLNet layer 11) — a different regime, not the same phenomenon at smaller scale. No systematic difference was found between Gaussian, Cosine, and CosFormer attention types.

**Next recommended step:** Implement the standardization ablation already specced in `CLAUDE.md`'s Phase 2 Plan (`_standardize()` in `gaussian_attention.py` / `cosine_attention.py`, plus `standardize_qk` config flag) and re-run a small subset of checkpoints with/without it, to check whether the mild concentration measured here has any measurable effect on validation loss — this is now a backed-by-measurement experiment rather than an assumption. Do this before any further long training runs, per the user's instruction. Also still outstanding from the earlier repository review: fix the CPU-phase checkpoint-overwriting issue, add held-out test-set evaluation, and consider multi-seed runs — see the full repository review from earlier in this session for the complete punch list.

---

## 2026-07-03 15:05 — Implemented and ran the Q/K standardization ablation (Gaussian + Cosine, CE only)

**What was done:** Implemented `standardize_qk` as an optional, off-by-default config/CLI flag that per-token mean/std standardizes Q and K (across `d_model`, before the head split) immediately before Gaussian attention's kernel and Cosine attention's existing L2-normalization — exactly the fix specced in `CLAUDE.md`'s Phase 2 Plan (Paper 1, Timkey & van Schijndel). Verified structurally and empirically that `standardize_qk=False` (the default) leaves existing behavior byte-for-byte unchanged, since the new code path is simply not executed. Then ran the two requested experiments — Gaussian+CE+standardize_qk and Cosine+CE+standardize_qk — at the same settings as the existing 50-epoch MPS baselines (1000 train chunks, 200 val chunks, `d_model=64, n_layers=2, n_heads=4, d_ff=256, batch=8`, MPS device) and compared against the existing, untouched baseline checkpoints.

**Files created:**
- `mps_experiments/standardization_ablation/checkpoints/gaussian_ce_standardized/` — `best.pt`, 5 epoch checkpoints, `training_log.csv`
- `mps_experiments/standardization_ablation/checkpoints/cosine_ce_standardized/` — same
- `mps_experiments/standardization_ablation/logs/gaussian_ce_standardized.log`, `logs/cosine_ce_standardized.log`
- `mps_experiments/standardization_ablation/results/gaussian_ce_standardized_results.txt`, `results/cosine_ce_standardized_results.txt`
- `mps_experiments/standardization_ablation/results/standardization_comparison_summary.csv`
- `mps_experiments/standardization_ablation/plots/gaussian_val_loss_comparison.png`, `plots/cosine_val_loss_comparison.png`, `plots/final_val_loss_bar_comparison.png`
- `mps_experiments/standardization_ablation/make_comparison_plots.py`
- `mps_experiments/standardization_ablation/ABLATION_FINDINGS.md` — written interpretation

**Files modified (additive only, default behavior preserved):**
- `attention/gaussian_attention.py`, `attention/cosine_attention.py` — added `standardize_qk` constructor arg (default `False`) and `_standardize()` method
- `attention/__init__.py` — forwards `standardize_qk` to gaussian/cosine constructors (default `False`)
- `model/transformer.py` — `build_model()` reads `cfg.get("standardize_qk", False)`
- `configs/default.yaml` — added `standardize_qk: false`
- `train.py` — added `--standardize_qk` CLI flag, prints setting in run banner

**Files NOT touched:** no existing checkpoint, training log, result `.txt`, or plot was modified. Confirmed via `find mps_experiments/checkpoints -newer <new file>` returning nothing from before this session's edits, and via `mtime`/checksum spot-checks.

**Why it was done:** direct follow-up to the rogue-dimension diagnostic (previous entry), which found mild-but-real dimensional concentration (2–5× uniform) in the Q/K per-head subspace — the literal space Gaussian and Cosine attention compute their kernel in. This ablation tests whether that measured concentration actually costs the model anything, converting the diagnostic's descriptive finding into an interventional one.

**Result:** standardization improved validation loss for both attention types — Gaussian: 5.2457 → 5.1529 (−1.77%), Cosine: 5.1474 → 5.0920 (−1.08%) — with the improvement holding across the entire training curve, not just the final epoch. No meaningful effect on per-epoch wall-clock speed (differences within normal system noise), but a clear improvement in convergence speed: both standardized models crossed below the random-chance baseline (5.2983) roughly 40% fewer epochs in (Gaussian: epoch 26 vs. 42; Cosine: epoch 13 vs. 23). Both runs were fully stable — monotonic loss decrease throughout, no divergence. Full detail, caveats, and interpretation in `mps_experiments/standardization_ablation/ABLATION_FINDINGS.md`.

**Next recommended step:** per the user's instruction, stopping here to report rather than expanding the ablation. If continuing later: extend to `mse`/`ce+ntl` loss types and to CosFormer (not in scope for this fix per the original spec, but worth a control run to confirm no effect there since it doesn't use raw Q/K distance/cosine), and consider repeated seeds to put a confidence interval on the ~1–2% improvement before it goes in the paper as a headline number.

---

## 2026-07-05 19:34 — Repeated-seed validation (n=3) for the 5 most important comparisons — corrects the standardization ablation

**What was done:** added a `--seed` CLI flag to `train.py` (seeds `random`, `numpy`, `torch`; verified to produce identical model init for the same seed and different init across seeds). Ran 15 training jobs — 5 configurations × 3 seeds (0, 1, 2) — sequentially on MPS at the same settings as every other MPS experiment (50 epochs, 1000 train chunks, 200 val chunks, `d_model=64, n_layers=2, n_heads=4, d_ff=256, batch=8`): Gaussian+CE baseline, Gaussian+CE+standardize_qk, Cosine+CE baseline, Cosine+CE+standardize_qk, and CosFormer+CE. Total wall time ~3h35m; all 15 runs completed successfully (exit 0), all monotonic, no divergence.

**Files created:**
- `mps_experiments/repeated_seeds/checkpoints/<config>_seed{0,1,2}/` — 15 full checkpoint directories (`best.pt`, 5 epoch checkpoints, `training_log.csv`)
- `mps_experiments/repeated_seeds/logs/<config>_seed{0,1,2}.log` — 15 full stdout logs
- `mps_experiments/repeated_seeds/results/repeated_seeds_summary.csv` — one row per run
- `mps_experiments/repeated_seeds/results/repeated_seeds_stats.csv` — one row per config: mean/std over 3 seeds
- `mps_experiments/repeated_seeds/plots/mean_std_bar_comparison.png`, `plots/all_seeds_val_curves.png`
- `mps_experiments/repeated_seeds/run_all.sh` — orchestration script (sequential, no `set -e`, continues past any single failed run)
- `mps_experiments/repeated_seeds/make_repeated_seeds_report.py` — generates the stats CSVs and plots from the 15 logs
- `mps_experiments/repeated_seeds/REPEATED_SEEDS_FINDINGS.md` — written interpretation

**Files modified (additive only):** `train.py` — added `--seed` argument and `set_seed()` helper. No prior run used seed control, so this doesn't change any earlier result's validity, only enables new reproducible ones going forward.

**Files NOT touched:** no existing checkpoint, log, result, or plot outside `mps_experiments/repeated_seeds/` was modified — including the `mps_experiments/standardization_ablation/` and `mps_experiments/checkpoints/` results this experiment re-examines.

**Why it was done:** the previous standardization ablation and the original 9-run MPS matrix each rested on exactly one run per configuration, with no seed control at all (the `--seed` flag didn't exist yet). The user asked to check whether the standardization "improvement" and the "CosFormer is best" ranking survive ordinary seed-to-seed noise.

**Result — this is an important correction to the previous session's conclusion:**
- **Mean/std (final val loss, n=3):** Gaussian CE 5.1580±0.0892, Gaussian CE+std 5.1563±0.0882, Cosine CE 5.2823±0.0300, Cosine CE+std 5.2847±0.0306, CosFormer CE 5.1494±0.0911.
- **Standardization does NOT meaningfully help either attention type.** The mean effect (Gaussian −0.03%, Cosine +0.05%) is more than an order of magnitude smaller than the seed-to-seed std dev (0.03–0.09) — the previously reported −1.77%/−1.08% improvements were very likely noise from the two single, uncontrolled runs, not a real effect. Paired same-seed comparison shows Gaussian improves consistently in direction across all 3 seeds but by a tiny, practically negligible margin (0.0007–0.0034); Cosine shows no consistent direction (better in 1 of 3 seeds, worse in 2 of 3).
- **CosFormer+CE remains the best point estimate (5.1494) but is not decisively separated from Gaussian (5.1563–5.1580)** — the three overlap well within one standard deviation of each other at n=3. **Cosine is clearly and robustly the worst attention type** — its gap to Gaussian/CosFormer (~0.13) is several times larger than any of the three configurations' own std dev.
- Secondary observation: seed 2 produced a notably better result for both Gaussian and CosFormer but not for Cosine (its worst seed), suggesting differing seed-sensitivity by attention type — descriptive only, not causally explained by this experiment.

**Next recommended step:** update any paper draft language that cited the single-seed standardization result or an unqualified "CosFormer is best" claim — replace with the corrected, appropriately-hedged statements above. If a firmer ranking between Gaussian and CosFormer is needed for the paper, run more seeds (5–10) for just those two configurations rather than re-running the full matrix. Do not add new methods (loss types, attention types) without being asked, per the user's running instruction across this phase of work.

---

## 2026-07-06 00:05 — 100-epoch check, all 3 attention types, single seed, no standardization

**What was done:** ran Gaussian+CE, Cosine+CE, and CosFormer+CE for 100 epochs each (seed 0, no `standardize_qk`), same settings as every other MPS experiment (1000 train chunks, 200 val chunks, `d_model=64, n_layers=2, n_heads=4, d_ff=256, batch=8`), sequentially in the background. User explicitly asked to hold off on 150 epochs pending these results.

**Files created:**
- `mps_experiments_100ep/checkpoints/{gaussian_ce,cosine_ce,cosformer_ce}/` — `best.pt`, 10 periodic epoch checkpoints, `training_log.csv` each
- `mps_experiments_100ep/logs/{gaussian_ce,cosine_ce,cosformer_ce}.log`
- `mps_experiments_100ep/results/100ep_summary.csv`
- `mps_experiments_100ep/plots/100ep_val_loss_comparison.png`
- `mps_experiments_100ep/make_100ep_plots.py`
- `mps_experiments_100ep/RESULTS_100EP.md` — written interpretation

**Files NOT touched:** nothing under `mps_experiments/` (original matrix, standardization ablation, repeated-seed study) was modified — this new experiment lives entirely in the sibling `mps_experiments_100ep/` directory per the user's instruction.

**Why it was done:** after the repeated-seed study left Gaussian and CosFormer statistically tied at 50 epochs, the user asked whether to extend training; rather than guess scope, asked a clarifying question and the user chose "all 3 attention types, 1 seed, no standardization" as the fastest full check before deciding on further seeds or epochs.

**Result:**
- **Val loss @ epoch 50 → epoch 100:** Gaussian 5.2156 → 5.0361; Cosine 5.2637 → 5.0917; CosFormer 5.2019 → 5.0254.
- **All three curves were still monotonically decreasing at epoch 100** — none has converged at this data budget; more epochs would likely help further.
- **Ranking is consistent with the epoch-50 repeated-seed result**, not contradicted: CosFormer numerically best, Gaussian a close second (gap 0.011 — still far smaller than the ~0.09 seed-to-seed std dev found earlier, so the tie remains unresolved by this single seed), Cosine clearly worst (gap ~0.06–0.066).
- Sanity check: this run's seed-0 epoch-50 values matched the earlier repeated-seed study's seed-0 values exactly, confirming reproducibility of the seeding/data pipeline.
- Gaussian's 77.9-minute runtime included a sustained host-machine scheduling slowdown (epochs 32–53), not a training issue — steady-state timing matched prior Gaussian runs.

**Next recommended step:** per the user's explicit instruction, did not run 150 epochs. If continuing: either extend to 150 epochs (curves show no plateau, likely to help) or run the epoch-100 config with 3 seeds to properly test whether Cosine's narrower gap at epoch 100 (vs. the wider 3-seed gap at epoch 50) is a real training-length effect or just this seed's own favorable draw for Cosine.

---

## 2026-07-06 — Held-out test-set evaluation (first in this project)

**What was done:** evaluated the three 100-epoch checkpoints (`mps_experiments_100ep/checkpoints/{gaussian_ce,cosine_ce,cosformer_ce}/best.pt`) read-only on the full, uncapped MAESTRO test split (177 held-out pieces → 11,451 chunks — never used in training or validation anywhere in this project), using the same loss function and batch size as training/validation for exact comparability. This was chosen ahead of resolving the Gaussian/CosFormer tie or extending to 150 epochs, per user priority — closes the "out-of-sample performance" gap Daniel's original framing asked for, which no prior experiment in this project had touched.

**Files created:**
- `mps_experiments/test_evaluation/evaluate_test_set.py` — the evaluation script (read-only w.r.t. checkpoints; safe to re-run)
- `mps_experiments/test_evaluation/make_test_eval_plot.py`
- `mps_experiments/test_evaluation/logs/test_evaluation.log`
- `mps_experiments/test_evaluation/results/test_set_summary.csv`
- `mps_experiments/test_evaluation/plots/val_vs_test_comparison.png`
- `mps_experiments/test_evaluation/TEST_EVALUATION_FINDINGS.md` — written interpretation

**Files NOT touched:** `mps_experiments_100ep/` checkpoints were only loaded and read (`map_location="cpu"`), never modified. Nothing under `mps_experiments/checkpoints/`, `mps_experiments/standardization_ablation/`, or `mps_experiments/repeated_seeds/` was touched.

**Result:**
- **Val → test loss:** Gaussian 5.0361 → 5.0457 (gap +0.0096); Cosine 5.0917 → 5.0986 (gap +0.0069); CosFormer 5.0254 → 5.0347 (gap +0.0093).
- **No overfitting** — all three gaps are small and tightly clustered, an order of magnitude smaller than the ~0.03–0.09 seed-to-seed noise found in the repeated-seed study, despite the test set being 57× larger and completely disjoint from validation.
- **Attention-type ranking holds exactly out-of-sample**: CosFormer < Gaussian < Cosine, same order and similar gap magnitudes as on validation — corroborates but does not overturn any earlier conclusion, including that the Gaussian/CosFormer tie remains a single-seed open question (this evaluation added data diversity, not seed diversity).

**Next recommended step:** the two options identified before this task remain open — resolve the Gaussian/CosFormer tie with 2 more seeds at 100 epochs (4 runs, reusing existing seed-0 100-epoch runs), or extend to 150 epochs since no curve had plateaued. Test-set evaluation could also be extended to the repeated-seed checkpoints (15 checkpoints in `mps_experiments/repeated_seeds/`) if per-seed test-loss variance becomes useful for the paper, but was not requested and not done here.

---

## 2026-07-07 — Tie resolution: 2 more seeds for Gaussian and CosFormer at 100 epochs

**What was done:** added seeds 1 and 2 for Gaussian+CE and CosFormer+CE at 100 epochs (4 new runs total), reusing the existing seed-0 100-epoch checkpoints. Same settings as every other 100-epoch run (CE loss, no standardization, 1000 train / 200 val chunks, MPS). Cosine was excluded — already conclusively established as worst, nothing to resolve there. This directly followed up on the standing recommendation from the last two turns to settle the Gaussian-vs-CosFormer tie before doing anything else.

**Files created:**
- `mps_experiments_100ep/checkpoints/{gaussian_ce_seed1,gaussian_ce_seed2,cosformer_ce_seed1,cosformer_ce_seed2}/` — full checkpoint directories
- `mps_experiments_100ep/logs/{gaussian_ce_seed1,gaussian_ce_seed2,cosformer_ce_seed1,cosformer_ce_seed2}.log`
- `mps_experiments_100ep/results/tie_resolution_summary.csv`
- `mps_experiments_100ep/plots/tie_resolution_mean_std.png`
- `mps_experiments_100ep/run_seed_extension.sh`, `make_tie_resolution_plot.py`
- `mps_experiments_100ep/TIE_RESOLUTION_FINDINGS.md` — written interpretation

**Files NOT touched:** existing seed-0 checkpoints (`gaussian_ce/`, `cosformer_ce/`, `cosine_ce/`), `results/100ep_summary.csv`, `RESULTS_100EP.md`, and everything under `mps_experiments/`.

**Result — the tie is now resolved as a genuine statistical tie, not unresolved noise:**
- **Mean val loss (n=3 seeds, epoch 100):** Gaussian 4.9701 ± 0.0743; CosFormer 4.9681 ± 0.0755. Difference: −0.0020, about 3% of one standard deviation — as close to a dead heat as this experiment design can produce.
- **Paired same-seed comparison shows no consistent winner**: CosFormer ahead in 2 of 3 seeds, Gaussian ahead in 1 of 3, all margins small — the signature of genuinely equivalent methods, not a real effect obscured by noise (contrast with the standardization ablation's Gaussian result, which *was* directionally consistent across all 3 seeds despite being tiny).
- **Seed 2 was again the best seed for both attention types** — same pattern already seen at 50 epochs with the same seed values, now replicated at 100 epochs, making it very unlikely to be coincidence (though still not causally explained).
- All 6 runs (3 seeds × 2 attention types) were still monotonically decreasing at epoch 100 — none converged.
- Gaussian seed-0's earlier 77.9-minute runtime is now confirmed as an outlier (seed 1: 25.6 min, seed 2: 36.6 min).

**Conclusion for the paper:** this project cannot distinguish Gaussian from CosFormer attention on validation loss at this model/data scale, across 3 seeds, two training lengths (50 and 100 epochs), and the held-out test set. Any single-run "CosFormer wins" framing should be replaced with "Gaussian and CosFormer perform statistically equivalently; Cosine is clearly worse; CosFormer trains faster." This closes the last open ranking question in the project.

**Next recommended step:** the experimental groundwork for the paper's Results section is now largely complete (rogue-dimension question answered, standardization shown to be a null result, out-of-sample generalization confirmed, attention-type ranking fully characterized with proper seed rigor). Remaining lower-priority items: repeated-seed/test-set coverage for MSE/NTL loss types (currently only single-run), and 150 epochs if further absolute-loss improvement is wanted (secondary now that the ranking question is settled). Otherwise, this is a reasonable point to begin drafting the paper's Results section.

---

## 2026-07-07 — CE vs. CE+NTL loss-function comparison, 3 seeds, Gaussian + CosFormer — corrects the original matrix's "directly comparable" claim

**What was done:** brought repeated-seed rigor to Daniel's actual second research axis (loss function comparison), which until now rested entirely on single, non-seed-controlled runs. Ran Gaussian+CE+NTL and CosFormer+CE+NTL for 3 seeds each (6 new runs, 100 epochs, same scale as the existing CE runs, λ=0.5). Then, because the raw CE-vs-NTL gap looked suspiciously uniform (+0.0349 for both attention types, spread of only 0.0016 across all 6 seed pairs — far tighter than any genuine effect seen elsewhere in this project), wrote a decomposition script that re-evaluates all 12 relevant checkpoints (6 CE + 6 NTL) under a single fixed pure-CE metric, isolating whether NTL training changes next-token prediction quality from whether it just changes what number gets logged.

**Files created:**
- `mps_experiments_100ep/checkpoints/{gaussian_ntl_seed0,gaussian_ntl_seed1,gaussian_ntl_seed2,cosformer_ntl_seed0,cosformer_ntl_seed1,cosformer_ntl_seed2}/` — full checkpoint directories
- `mps_experiments_100ep/logs/{...}.log` — 6 new logs
- `mps_experiments_100ep/results/ntl_vs_ce_summary.csv`, `results/ntl_pure_ce_decomposition.csv`
- `mps_experiments_100ep/plots/ntl_vs_ce_pure_ce_comparison.png`
- `mps_experiments_100ep/run_ntl_seed_extension.sh`, `decompose_ntl_loss.py`, `make_ntl_comparison_plot.py`
- `mps_experiments_100ep/NTL_VS_CE_FINDINGS.md` — written interpretation

**Files NOT touched:** all existing CE checkpoints were only loaded read-only for re-evaluation, never modified. `results/100ep_summary.csv`, `results/tie_resolution_summary.csv`, `RESULTS_100EP.md`, `TIE_RESOLUTION_FINDINGS.md`, and everything under `mps_experiments/` were untouched.

**Result — an important correction to the original 9-run matrix's methodology, not just a new data point:**
- **Raw logged val_loss** made CE+NTL look worse by a suspiciously uniform +0.0349 for both attention types — too tight and too consistent to be a real training effect.
- **Decomposition showed why**: the `ce+ntl` loss includes an added `λ·MSE_ordinal` term that plain CE's logged loss doesn't have — the two were never on the same scale, despite the original `*_ntl_results.txt` files (from the first 9-run matrix) explicitly describing them as "directly comparable." That framing was incorrect.
- **Re-evaluating all 12 checkpoints under one fixed pure-CE metric**: Gaussian CE-trained 4.9701±0.0743 vs. NTL-trained 4.9700±0.0745 (diff −0.0001); CosFormer CE-trained 4.9681±0.0755 vs. NTL-trained 4.9680±0.0756 (diff −0.0001). **Statistically identical for both attention types** — NTL is a no-op on next-token prediction quality at this scale, not a harmful regularizer as the raw numbers implied.

**Next recommended step:** this project's stated NTL hypothesis (does it improve numerical closeness of ordinal-token errors) was not directly tested — that requires comparing MSE_ordinal itself between CE-trained and NTL-trained models, not pure CE. Checkpoints already on disk make this a cheap follow-up if the paper wants to pursue the NTL line further. Otherwise, both of this project's research axes (attention type, loss function) now have matching 3-seed rigor, and the paper's Results section can be drafted on a fully consistent evidentiary standard.

---

## 2026-07-08 — Ordinal-token MSE evaluation: does NTL achieve its actual design goal?

**What was done:** direct, cheap follow-up to the pure-CE decomposition — re-evaluated the same 12 existing 100-epoch checkpoints (`mps_experiments_100ep/checkpoints/`: gaussian/cosformer × {ce, ce+ntl} × 3 seeds) on ordinal-token MSE (pitch/duration/velocity numerical closeness), using the exact `_mse_ordinal()` function the NTL loss itself optimizes. This tests NTL's actual stated claim (numerical closeness), as opposed to the earlier pure-CE check (classification quality). Read-only throughout — no retraining, no model modification, all output in a new `mps_experiments/ordinal_mse_evaluation/` folder.

**Files created:**
- `mps_experiments/ordinal_mse_evaluation/evaluate_ordinal_mse.py`, `make_ordinal_mse_plot.py`
- `mps_experiments/ordinal_mse_evaluation/logs/ordinal_mse_evaluation.log`
- `mps_experiments/ordinal_mse_evaluation/results/ordinal_mse_summary.csv`, `results/ordinal_mse_verdict.csv`
- `mps_experiments/ordinal_mse_evaluation/plots/ordinal_mse_comparison.png`
- `mps_experiments/ordinal_mse_evaluation/ORDINAL_MSE_FINDINGS.md` — written interpretation

**Files NOT touched:** all 12 checkpoints in `mps_experiments_100ep/checkpoints/` were loaded read-only only; nothing under `mps_experiments_100ep/` (results, plots, findings docs) was modified.

**Result — complete, consistent negative result, closes the NTL question:**
- **Gaussian**: CE-trained ordinal-MSE 0.070014±0.001501 vs. NTL-trained 0.069900±0.001455 — diff −0.000114 (−0.2%, −0.08x avg std).
- **CosFormer**: CE-trained 0.069945±0.001790 vs. NTL-trained 0.069836±0.001737 — diff −0.000109 (−0.2%, −0.06x avg std).
- **No measurable benefit on NTL's own stated design goal**, for either attention type. Combined with the earlier pure-CE finding, CE+NTL training at this model/data scale changes neither classification quality nor ordinal numerical closeness relative to plain CE — a complete, not partial, negative result across both metrics that matter.
- Named as open (untested) explanations in the findings doc: model capacity too small for the auxiliary term to matter, λ=0.5 possibly miscalibrated (never swept), or 100 epochs insufficient for the two objectives to diverge — worth flagging in the paper's discussion section but not tested further here.

**Next recommended step:** the NTL investigation is now complete across both metrics that matter (classification and numerical closeness) — no further follow-up needed unless the paper wants to sweep λ or test at larger scale, both of which are new-scope decisions rather than gaps in the current evidence. Both of this project's research axes (attention type, loss function) now have full 3-seed rigor with no known open interpretive questions. This is a reasonable point to move to drafting the paper's Results section.

---

## 2026-07-08 (later) — Literature review (25 papers) and first manuscript draft (Introduction)

**What was done:** systematically read all 20 PDFs in `useful files/` and cross-referenced against the 5 PDFs in `daniel files/` (4 confirmed byte-identical duplicates, 1 genuinely new: the Rogue Dimensions paper, which was missing from `useful files`). For each paper, recorded: file, main contribution, relation to the project, target manuscript section, and main-vs-background citation classification, then built a citation map grouping all 25 papers by theme. Caught and corrected a self-introduced error (swapped arXiv IDs between the NTL and Secretly Linear papers) via file-size cross-checking before it reached the manuscript. Then drafted the manuscript's Introduction section, using Daniel's existing title/authors/abstract as the fixed starting point and following his specified 5-part narrative structure (Transformer/softmax limitations → efficient/softmax-free attention review → why symbolic music → research gap → RQs and contributions).

**Files created:**
- `paper/manuscript.md` — title, authors, Daniel's original abstract (verbatim), and the new Introduction section

**Files NOT touched:** no experiment, checkpoint, script, or existing findings document was modified. This is a pure writing/synthesis task, no code run.

**Why it was done:** direct continuation of the file-by-file project walkthrough and paper-by-paper literature review from earlier in this session, per explicit user request to begin drafting the manuscript one section at a time, using only claims supported by the project's actual experiments and the reviewed papers.

**Result:** Introduction drafted (~1,100 words), citing 12 of the 25 reviewed papers by design (the remainder — e.g., Museformer, Compound Word Transformer's full detail, the RPE-for-linear-attention papers, Secretly Linear — were deliberately deferred to Related Work, Methods, or Discussion rather than forced into the Introduction). One citation gap flagged explicitly to the user: the MAESTRO dataset paper (Hawthorne et al., 2019) is referenced by name in the Introduction's Methods-preview sentence but was **not** cited inline, since it was not among the papers read this session and citing it without having read it would violate the "do not invent references" instruction.

**Next recommended step:** awaiting user review/approval of the Introduction before proceeding to the next section (likely Related Work, given the citation map already groups the remaining 13 papers by theme). If approved, the actual MAESTRO paper should be sourced and read before Methods is drafted, since the dataset section will need to cite it properly.

---

## 2026-07-08 (later still) — Style cleanup (em dashes) and Related Work section drafted

**What was done:** (1) removed all 9 em dashes from the Introduction per user request, replacing each with commas, semicolons, colons, parentheses, or a sentence split depending on grammatical context, with no change to meaning or content. (2) Drafted section 2, Related Work, covering all 13 papers not cited in the Introduction (plus deeper treatment of several already introduced), organized into four subsections: Efficient and Softmax-Free Attention (2.1), Symbolic Music Generation with Transformers (2.2), Loss Functions for Ordinal and Numerical Tokens (2.3), and Representational Geometry in Transformers (2.4). Proactively avoided em dashes throughout this new section from the start.

**Files modified:** `paper/manuscript.md` (em dash cleanup in Introduction; Related Work section appended).

**Files NOT touched:** no experiment, checkpoint, or findings document.

**Why it was done:** direct continuation of section-by-section manuscript drafting per user request; em dash removal was an explicit stylistic instruction to reduce "AI-generated" prose signals.

**Result:** all 25 reviewed papers are now cited somewhere in the manuscript (12 in Introduction, the remaining 13 introduced or elaborated in Related Work). No figures or tables were needed for Related Work; user was told this explicitly and informed that Methods and Results will need concrete photo/table placement guidance.

**Next recommended step:** awaiting user review of Related Work. Next section is most likely Methods, which will need: a hyperparameter table (model size, training settings), a dataset/tokenization description, and should note where to cite the actual MAESTRO paper (still not sourced/read). Results will need the project's existing plots (e.g. `mps_experiments_100ep/plots/tie_resolution_mean_std.png`, `mps_experiments_100ep/plots/ntl_vs_ce_pure_ce_comparison.png`, `mps_experiments/test_evaluation/plots/val_vs_test_comparison.png`) and several result tables, which will be provided as copy-pasteable markdown per the user's instruction.

---

## 2026-07-08 (later still, again) — Method section drafted (Section 3)

**What was done:** drafted section 3, Method, covering: 3.1 Task and Data Representation (MAESTRO, REMI tokenization, vocabulary table), 3.2 Model Architecture (decoder-only pre-norm Transformer, hyperparameter table), 3.3 Attention Mechanisms (our specific Gaussian/Cosine/CosFormer implementations, including the explicit methodological caveat that Gaussian attention's Nystrom path is never exercised at seq_len=512 and the standardization option), 3.4 Loss Functions (CE and CE+NTL formulas), 3.5 Rogue-Dimension Diagnostic (methodology, explicitly tied to Timkey & van Schijndel's Eq. 3-4). Included two copy-pasteable markdown tables (vocabulary breakdown, model hyperparameters) per user request. Continued avoiding em dashes throughout.

**Files modified:** `paper/manuscript.md` (Method section appended).

**Files NOT touched:** no experiment, checkpoint, or findings document.

**Why it was done:** direct continuation of section-by-section manuscript drafting. Training/evaluation protocol (seeds, epoch budgets per experimental phase, data splits, hardware) was deliberately deferred to a following "Experimental Setup" section to keep this turn's deliverable a manageable review size, rather than combining architecture description with experimental design in one very long section.

**Result:** Method section is accurate to the actual codebase (cross-checked against `model/transformer.py`, `attention/*.py`, `data/vocabulary.py`, `data/midi_dataset.py`, `utils/losses.py` read earlier in this session) and includes the methodologically important caveat about the Gaussian attention Nystrom path.

**Next recommended step:** awaiting user review. Next section: Experimental Setup (seeds, epoch budgets, train/val/test splits, hardware, evaluation protocol), then Results (where the actual plots and result tables come in).

---

## 2026-07-08 (later still, x3) — Method section revised into fuller academic prose

**What was done:** revised Section 3.3 (Attention Mechanisms) and 3.4 (Loss Functions, renumbered) per user request, expanding the bolded mini-paragraph format into fully developed prose paragraphs. Split the Q/K standardization content out of 3.3 into its own new subsection, 3.4 "Q/K Standardization Ablation," specifically to make structurally unambiguous that it is a separate ablation and not part of the main model definition (per explicit user requirement). Strengthened and made more prominent the existing statement that Gaussian attention uses only the exact quadratic kernel in this work, never the Nystrom/linear SOFT approximation. Subsections renumbered: 3.5 Loss Functions, 3.6 Rogue-Dimension Diagnostic. No new methods, claims, or content were added; this was a prose-quality and structural-clarity revision only.

**Files modified:** `paper/manuscript.md` (Section 3 revised in place).

**Files NOT touched:** no experiment, checkpoint, or findings document.

**Why it was done:** user feedback that Section 3 read as note-like/spec-sheet in places rather than full academic prose, plus two explicit clarity requirements (Gaussian = exact quadratic only, standardization = ablation only) that are now each satisfied by dedicated, unambiguous sentences and, for standardization, an entire dedicated subsection.

**Next recommended step:** awaiting user review of the revised Method section. Per user instruction, Results has not been started.

---

## 2026-07-08 (later still, x4) — Method section: LaTeX formulas added, Section 3.6 filled in

**What was done:** converted all plain-text pseudo-formulas in Section 3 (pre-norm update rule, Gaussian kernel, cosine similarity, CosFormer re-weighting, standardization, NTL expected-value/loss, combined loss) into proper LaTeX math notation (`$...$` inline, `$$...$$` display), directly pasteable into a `.tex` manuscript. Filled in Section 3.6 (Rogue-Dimension Diagnostic), which had been left as a single summary paragraph, with full formulas matching the actual diagnostic script exactly: per-dimension variance, per-dimension cosine-similarity contribution, and per-dimension squared-Euclidean-distance contribution, plus the top-k-vs-uniform-baseline comparison. Caught and explicitly documented a real methodological nuance while writing this: the diagnostic script (`rogue_dimension_diagnostic.py`) averages the **absolute value** of each dimension's per-pair cosine contribution before ranking, not the signed value as in Timkey & van Schijndel's original Equation 4; this is now stated explicitly in the manuscript as a deliberate adaptation, not a literal reproduction, so the Methods section doesn't overclaim exact replication of the source paper's formula.

**Files modified:** `paper/manuscript.md` (Section 3 tables/formulas converted to LaTeX; 3.6 fully written).

**Files NOT touched:** no experiment, checkpoint, or findings document. (Re-confirmed the absolute-value detail against the actual `cosine_profile()` function in `mps_experiments/diagnostics/rogue_dimensions/rogue_dimension_diagnostic.py`, read earlier in this session, rather than assuming.)

**Why it was done:** user wants formulas and tables written so they can be copied and pasted directly (LaTeX was chosen as the default format, since the user's original AskUserQuestion clarification attempt was declined and LaTeX is standard for the target venues), and explicitly asked for Section 3.6 to be completed rather than left as a placeholder-level paragraph.

**Next recommended step:** awaiting user review. Method section (3.1-3.6) is now believed complete pending approval. Next: Experimental Setup, then Results.

---

## 2026-07-08 (later still, x5) — Method section reverted to plain-prose style per user preference

**What was done:** user rejected the LaTeX (`$...$`/`$$...$$`) formatting from the previous revision and asked for the plain-prose, inline-pseudo-code-formula style used earlier in the session instead (e.g. `exp(-||Q_i - K_j||^2 / 2*sqrt(d_head))` rather than proper LaTeX). Reverted Section 3 in `paper/manuscript.md` to that style throughout (3.1-3.5), removed the bold emphasis markers added around mechanism names and key caveats, and rewrote Section 3.6 in the same plain style (previously only existed in LaTeX form). Content and technical claims are unchanged from the LaTeX version, including the absolute-value adaptation note for the cosine-contribution diagnostic; only formatting changed.

**Files modified:** `paper/manuscript.md` (Section 3 reformatted in place).

**Files NOT touched:** no experiment, checkpoint, or findings document.

**Why it was done:** direct user preference after seeing the LaTeX version; user explicitly said "i like this text just fill in the 3.6 part," referring back to the plain-prose draft from two revisions prior.

**Next recommended step:** awaiting user review of the plain-style Method section. Method (3.1-3.6) is complete pending approval. Next: Experimental Setup, then Results. Worth confirming with the user going forward which style (plain-text vs. LaTeX) to default to for any future formulas, to avoid repeating this back-and-forth.

---

## 2026-07-08 (later still, x6) — Experimental Setup section drafted (Section 4)

**What was done:** clarified for the user that Methods contains no result plots/tables (those are all reserved for Results, with an explicit mapping of which existing plot/CSV in `mps_experiments*/` supports which upcoming Results claim). Then drafted Section 4, Experimental Setup: 4.1 Training Procedure (AdamW, LR schedule, batch size, MPS hardware), 4.2 Dataset Splits and Scale (962/137/177 piece counts, 1000/200 chunk caps, uncapped 11,451-chunk test set), 4.3 Repeated-Seed Protocol, 4.4 Experimental Design (a 7-row table mapping each of the 7 sub-experiments to attention types, loss types, seed counts, epoch budgets, and research question), 4.5 Evaluation Metrics and Protocol (including the pure-CE/ordinal-MSE re-evaluation methodology central to the NTL findings). Matched the plain-prose style established in the prior revision (no LaTeX, no bold emphasis markers).

**Files modified:** `paper/manuscript.md` (Section 4 appended).

**Files NOT touched:** no experiment, checkpoint, or findings document.

**Why it was done:** direct continuation of section-by-section manuscript drafting.

**Accuracy note caught while drafting:** the design-overview table required distinguishing that Cosine attention only received the 3-seed treatment at 50 epochs, not at 100 (it was never extended past seed 0 at 100 epochs, since it was already established as weakest at 50 epochs); an earlier draft sentence would have implied 3-seed coverage at both epoch budgets for all three attention types, which was not accurate. Fixed by splitting the table into separate 50-epoch and 100-epoch attention-comparison rows with the seed count spelled out precisely, and stating the scoping decision explicitly in prose (Section 4.4). Also explicitly noted that the rogue-dimension diagnostic used single-seed checkpoints predating the repeated-seed protocol, framed as a scoping decision rather than an oversight.

**Next recommended step:** awaiting user review of Section 4. Sections 1-4 (Introduction, Related Work, Method, Experimental Setup) are now complete pending approval. Next: Results, where the actual plots and result tables come in (mapping given to the user in the prior turn).

---

## 2026-07-08 (later still, x7) — Results section drafted (Section 5), all numbers re-verified against source CSVs

**What was done:** before writing, re-verified every number to be cited by reading directly from the source CSVs rather than relying on memory of the earlier findings docs: `mps_experiments/repeated_seeds/results/repeated_seeds_stats.csv`, `mps_experiments_100ep/results/tie_resolution_summary.csv`, `mps_experiments_100ep/results/100ep_summary.csv`, `mps_experiments/test_evaluation/results/test_set_summary.csv`, `mps_experiments_100ep/results/ntl_pure_ce_decomposition.csv`, `mps_experiments/ordinal_mse_evaluation/results/ordinal_mse_verdict.csv`, and spot-checked `mps_experiments/diagnostics/rogue_dimensions/results/rogue_dimension_summary.csv`. All matched prior memory. Drafted Section 5, Results, in six subsections following the RQ1/RQ2/RQ3 structure: 5.1 Attention-Mechanism Comparison (50 and 100 epoch results, tie, training speed), 5.2 Held-Out Test-Set Evaluation, 5.3 Loss-Function Comparison (raw gap and pure-CE decomposition), 5.4 Ordinal-Token Closeness, 5.5 Rogue-Dimension Diagnostic, 5.6 Q/K Standardization Ablation. Included 10 tables (Tables 4-13) and 7 explicit figure-placement markers (`[Figure N: insert <path> here]`) pointing at existing plots in `mps_experiments*/`. Fixed a stale cross-reference in Section 4.4 (pointed at "Section 5.3" for the rogue-dimension diagnostic; corrected to "Section 5.5" to match the actual Results numbering). Noted that Figure 1 (`mps_experiments/repeated_seeds/plots/mean_std_bar_comparison.png`) serves both 5.1 and 5.6, since it already contains both standardized and unstandardized bars; cross-referenced rather than duplicated.

**Files modified:** `paper/manuscript.md` (Section 5 appended; one cross-reference fix in Section 4.4).

**Files NOT touched:** no experiment, checkpoint, or findings document; all source CSVs were only read.

**Why it was done:** direct continuation of section-by-section manuscript drafting; numbers were re-verified from source rather than transcribed from memory given the stakes of getting a Results section wrong.

**Next recommended step:** awaiting user review of Section 5. All of Sections 1-5 are now drafted. Remaining: Discussion (should cover the missing relative-position-embedding gap, the Gaussian Nystrom-path caveat, the twice-replicated "seed 2 is favorable" observation, and candidate explanations for the NTL null result already outlined in `ORDINAL_MSE_FINDINGS.md`), Limitations, and Conclusion.
