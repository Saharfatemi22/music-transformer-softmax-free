# Q/K Standardization Ablation — Findings

**Question:** does the mild dimensional concentration measured in the rogue-dimension diagnostic (`mps_experiments/diagnostics/rogue_dimensions/FINDINGS.md`) actually matter for training — does standardizing Q/K before the Gaussian kernel / cosine similarity improve validation loss, training speed, or stability?

**Method:** implemented `standardize_qk` as an optional config/CLI flag (default `false`, existing behavior byte-for-byte unchanged when off). When on, Q and K are per-token mean/std standardized across `d_model` — `(x - mean) / std`, computed per token — immediately after the `W_Q`/`W_K` linear projections and before the head split, exactly as specced in `CLAUDE.md`'s Phase 2 Plan (Paper 1). Ran two new experiments, identical in every other hyperparameter to the existing 50-epoch MPS baselines: **Gaussian + CE + standardize_qk** and **Cosine + CE + standardize_qk**, 1000 train chunks, 200 val chunks, `d_model=64, n_layers=2, n_heads=4, d_ff=256, batch_size=8`, MPS device. Compared directly against the existing (unmodified) baseline checkpoints in `mps_experiments/checkpoints/{gaussian,cosine}_ce/`.

---

## Result

| Attention | Baseline val loss (ep. 50) | Standardized val loss (ep. 50) | Δ | % change | Epoch first below random chance (5.2983) |
|---|---|---|---|---|---|
| Gaussian | 5.2457 | **5.1529** | **−0.0928** | **−1.77%** | baseline: epoch 42 → standardized: **epoch 26** |
| Cosine | 5.1474 | **5.0920** | **−0.0554** | **−1.08%** | baseline: epoch 23 → standardized: **epoch 13** |

Full per-epoch data: `results/gaussian_ce_standardized_results.txt`, `results/cosine_ce_standardized_results.txt`, `results/standardization_comparison_summary.csv`. Plots: `plots/gaussian_val_loss_comparison.png`, `plots/cosine_val_loss_comparison.png`, `plots/final_val_loss_bar_comparison.png`.

## Did standardization improve validation loss?

**Yes, for both attention types.** Gaussian improved by 0.093 (1.8% relative), Cosine by 0.055 (1.1% relative). These are modest but consistent, monotonic improvements — the standardized curve sits below the baseline curve at essentially every epoch in both plots, not just at the final epoch (see the two line-chart comparisons). Gaussian benefited more than Cosine in absolute terms, which is consistent with the rogue-dimension diagnostic: Gaussian's kernel uses raw, unnormalized Q/K distances, so removing a dimension-scale imbalance has more room to help than Cosine, which already L2-normalizes Q/K (partially self-correcting for scale, per-vector, even without explicit standardization).

## Did standardization improve speed?

**No meaningful effect, as expected.** Steady-state epoch time: Gaussian 17.54s → 17.00s, Cosine 8.22s → 7.99s — differences of a few percent, well within normal run-to-run system noise on a shared machine, not a real effect. This is expected: `_standardize()` is a cheap elementwise mean/std/subtract/divide, negligible next to the kernel computation itself. Standardization was never expected to change asymptotic complexity or wall-clock cost, and it didn't.

**Convergence speed did improve, though**, which is a distinct and more interesting effect from raw epoch time: the standardized Gaussian model dropped below the random-chance baseline (5.2983) at epoch 26 versus epoch 42 for the unstandardized baseline; standardized Cosine crossed at epoch 13 versus epoch 23. Both models needed roughly 40% fewer epochs to start beating random chance. That's a training-efficiency finding independent of per-epoch wall-clock cost.

## Was training stable?

**Yes, in both runs.** Train and validation loss decreased monotonically every epoch in both the Gaussian and Cosine standardized runs — no divergence, no instability, no NaN losses. Both logs show the same kind of one-off CPU/OS scheduling time spikes seen throughout this project's other MPS runs (Gaussian epoch 18: 524s instead of ~17s; Cosine epochs 25 and 43: 451s and 202s instead of ~8s) — these are host-machine scheduling artifacts, not training instability; the loss values themselves show no discontinuity at those epochs.

## Interpretation

This is a genuine, if modest, positive result for the standardization ablation from Daniel's Paper 1 — even though the rogue-dimension diagnostic found nothing close to the severe single-dimension domination documented for large text transformers, the mild concentration that *was* measured (roughly 2–5× the uniform baseline, concentrated mainly in the Q/K per-head subspace) was still large enough to cost the model some measurable accuracy and convergence speed. Standardizing it away helped both distance-based attention mechanisms tested. This is consistent with, and adds evidence to, the diagnostic's finding that whatever concentration exists is real but small in magnitude — a small fix produced a small (not dramatic) improvement, which is the expected shape of result if the underlying effect size is itself small.

## Caveats

- **Single run per configuration, no repeated seeds.** The two Δval numbers (−0.093, −0.055) are real, monotonic, and consistent across the entire training curve (not just the final epoch), which is reassuring, but statistical significance across seeds has not been established.
- **CE loss only, at this training scale.** Not yet tested with the `mse` or `ce+ntl` loss types, or with CosFormer (which doesn't use raw Q/K distance/cosine and wasn't in scope for this ablation, per the standardization spec in `CLAUDE.md` and this task's request).
- **No test-set evaluation** — this compares validation loss only, consistent with how the rest of this project's results have been reported so far.

## What was and wasn't touched

**New/changed for this ablation (additive only, default behavior unchanged):**
- `attention/gaussian_attention.py`, `attention/cosine_attention.py` — added `standardize_qk` constructor arg (default `False`) and `_standardize()` method.
- `attention/__init__.py` — forwards `standardize_qk` from config kwargs (default `False`) for gaussian/cosine only.
- `model/transformer.py` — `build_model()` reads `cfg.get("standardize_qk", False)`.
- `configs/default.yaml` — added `standardize_qk: false` (existing default preserved).
- `train.py` — added `--standardize_qk` CLI flag (omit = use config default) and prints the setting in the run banner.
- Verified: with `standardize_qk=False` (the default), the new code path is not executed at all — existing runs are structurally guaranteed unaffected, not just empirically similar.

**Not touched:** no existing checkpoint, training log, result `.txt`, or plot was modified. All new outputs live under `mps_experiments/standardization_ablation/`.

## Files

- `checkpoints/gaussian_ce_standardized/`, `checkpoints/cosine_ce_standardized/` — `best.pt`, 5 periodic epoch checkpoints each, `training_log.csv`.
- `logs/gaussian_ce_standardized.log`, `logs/cosine_ce_standardized.log` — full stdout from each training run.
- `results/gaussian_ce_standardized_results.txt`, `results/cosine_ce_standardized_results.txt` — human-readable per-epoch tables, matching the style of the existing root-level `*_results.txt` files.
- `results/standardization_comparison_summary.csv` — machine-readable comparison table.
- `plots/gaussian_val_loss_comparison.png`, `plots/cosine_val_loss_comparison.png` — baseline vs. standardized loss curves.
- `plots/final_val_loss_bar_comparison.png` — final-epoch bar comparison, both attention types.
- `make_comparison_plots.py` — regenerates the comparison CSV/plots from the two training logs; safe to re-run.
