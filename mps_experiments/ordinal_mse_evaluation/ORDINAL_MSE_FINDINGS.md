# Ordinal-Token MSE Evaluation — Does NTL Achieve Its Actual Goal?

**Why this exists:** the pure-CE decomposition (`mps_experiments_100ep/NTL_VS_CE_FINDINGS.md`) showed CE+NTL training doesn't change next-token *classification* quality — but that was never NTL's actual claim. The Number Token Loss papers (Zausinger et al., ICML 2025) argue NTL improves **numerical closeness on ordinal-token errors**: predicting pitch 61 when the answer is 60 should be a smaller mistake than predicting pitch 5, something plain cross-entropy cannot express. This evaluation tests that specific, actual claim directly, using checkpoints already on disk — no retraining.

**Method:** re-evaluated all 12 existing 100-epoch checkpoints (`mps_experiments_100ep/checkpoints/`: gaussian/cosformer × {ce, ce+ntl} × 3 seeds each) on the same 200-chunk validation set used throughout this project, using the exact `_mse_ordinal()` function (`utils/losses.py`) the NTL loss itself optimizes — computing the expected numerical value under the model's predicted distribution vs. the true value, restricted to ordinal tokens (NOTE_ON pitch, DURATION, VELOCITY), regardless of which loss each checkpoint was actually trained with. All model loading was read-only (`map_location="cpu"`, `eval()`/`no_grad()`); nothing was retrained or modified.

---

## Result

| Attention | CE-trained ordinal-MSE | CE+NTL-trained ordinal-MSE | Diff | % change |
|---|---|---|---|---|
| Gaussian | 0.070014 ± 0.001501 | 0.069900 ± 0.001455 | −0.000114 | **−0.2%** |
| CosFormer | 0.069945 ± 0.001790 | 0.069836 ± 0.001737 | −0.000109 | **−0.2%** |

Both differences are tiny (−0.06x to −0.08x of the seed-to-seed standard deviation) — essentially zero, well inside noise. Full per-checkpoint data: `results/ordinal_mse_summary.csv`, `results/ordinal_mse_verdict.csv`. Plot: `plots/ordinal_mse_comparison.png` — all four bars are visually indistinguishable, and individual seed values land in nearly the same positions whether the model was CE-trained or CE+NTL-trained.

## Verdict: NTL does not improve ordinal-token numerical closeness at this scale, either

**No measurable benefit on its own stated design goal.** Combined with the earlier pure-CE result, this gives a complete, consistent picture: CE+NTL training at this model/data scale (112K parameters, 1000 train chunks, 100 epochs) changes **neither** next-token classification quality **nor** ordinal-token numerical closeness, relative to plain CE — for either Gaussian or CosFormer attention. The λ=0.5 regularization term is added to the training loss and does measurably increase the *logged* loss value (as shown in the pure-CE decomposition), but it does not detectably change what the trained model actually does on either the classification or the regression-flavored metric it was meant to influence.

This is a clean, complete negative result — not a partial or ambiguous one. There is no evidence in this project, across two different metrics and two attention types, that Number Token Loss provides any benefit over plain cross-entropy at this scale.

## Interpretation: why might this be happening?

A few candidate explanations, none tested further here but worth naming for the paper's discussion section:

- **Model capacity/scale**: at 112K parameters and this data budget, cross-entropy alone may already be extracting essentially all the learnable signal about ordinal structure that this model architecture can represent — leaving no headroom for an auxiliary regression term to add value. This would be consistent with the "Transformers are Expressive... For Regression?" paper's argument (Nath et al.) that small Transformers struggle with smooth-function approximation *in general*, which could mean the auxiliary MSE term also can't be optimized much further, not just that it's unnecessary.
- **λ may be miscalibrated**: λ=0.5 was used throughout this project without a sweep. It's possible a different weighting would produce a measurable effect in either direction; this project has only tested one value.
- **Training length**: 100 epochs might not be enough for the two loss terms' interaction to diverge from each other; this remains untested at longer horizons.

None of these can be distinguished from the data collected so far — flagging as open questions rather than conclusions.

## Caveats

- **CE loss ordinal-MSE was also evaluated post-hoc** (never optimized for it), same as the pure-CE evaluation of NTL-trained models was post-hoc for that metric — this is intentional and is exactly what makes the comparison fair (both groups evaluated on a metric neither used exclusively as its training signal, except NTL-trained models did partially optimize this one).
- **Single λ value (0.5), single model/data scale, single evaluation set (200 chunks)** — consistent with every other evaluation in this project, but the "no effect" conclusion is specific to this configuration.
- **Pooled across pitch/duration/velocity** — `_mse_ordinal()` combines all three ordinal token types into one normalized MSE; a breakdown by sub-type was not computed and could reveal an effect in one dimension masked by no effect (or an offsetting effect) in another. Not done here; a natural, cheap follow-up.

## What was and wasn't touched

**New:** everything under `mps_experiments/ordinal_mse_evaluation/` (script, logs, results, plots, this file).

**Not touched:** all 12 checkpoints in `mps_experiments_100ep/checkpoints/` were only loaded read-only for evaluation — no retraining, no modification. Nothing under `mps_experiments_100ep/results/`, `mps_experiments_100ep/plots/`, or any `.md` findings file there was touched. Nothing under `mps_experiments/` (other than this new folder) was touched.

## Files

- `evaluate_ordinal_mse.py` — the evaluation script; safe to re-run, read-only w.r.t. all checkpoints.
- `make_ordinal_mse_plot.py` — regenerates the plot from `results/ordinal_mse_summary.csv`.
- `logs/ordinal_mse_evaluation.log` — full stdout from the evaluation run.
- `results/ordinal_mse_summary.csv` — per-checkpoint ordinal-token MSE (12 rows).
- `results/ordinal_mse_verdict.csv` — per-attention-type CE vs. CE+NTL comparison (mean, std, diff, % change).
- `plots/ordinal_mse_comparison.png` — grouped bar chart, CE vs. CE+NTL, both attention types, individual seeds overlaid.
