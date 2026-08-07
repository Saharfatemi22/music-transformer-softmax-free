# Held-Out Test-Set Evaluation — Findings

**Why this exists:** every result reported in this project so far — the original 9-run matrix, the rogue-dimension diagnostic, the standardization ablation, the repeated-seed study, the 100-epoch check — used training/validation loss only. Daniel's original framing was specifically about **out-of-sample performance**, and the MAESTRO test split (177 pieces) had never been touched by any experiment. This closes that gap.

**Method:** loaded the three existing 100-epoch checkpoints (`mps_experiments_100ep/checkpoints/{gaussian_ce,cosine_ce,cosformer_ce}/best.pt`) read-only, in `eval()`/`no_grad()` mode — no training, no gradient updates, nothing written back. Evaluated each on the **full, uncapped** MAESTRO test split: 177 held-out pieces → 11,451 chunks of 512 tokens (vs. the 200-chunk cap used for validation during training) — cheap to do in full since this is inference-only (~50–115 seconds per model on CPU). Used the exact same loss function (`combined_loss`, CE for all three) and batch size (8) as training/validation, for direct comparability with each checkpoint's own recorded `val_loss`.

---

## Results

| Config | Val loss (200 chunks) | Test loss (11,451 chunks) | Gap (test − val) | Val perplexity | Test perplexity |
|---|---|---|---|---|---|
| Gaussian | 5.0361 | **5.0457** | +0.0096 | 153.9 | 155.4 |
| Cosine | 5.0917 | **5.0986** | +0.0069 | 162.7 | 163.8 |
| CosFormer | 5.0254 | **5.0347** | +0.0093 | 152.2 | 153.7 |

Random-chance baseline: `log(196) = 5.2983`. Full data: `results/test_set_summary.csv`. Plot: `plots/val_vs_test_comparison.png`.

## Does the model generalize, or is this overfitting to validation?

**Generalizes well — no meaningful overfitting.** All three gaps are small and tightly clustered (0.007–0.010), on a **57× larger and completely disjoint** evaluation set (11,451 test chunks vs. 200 validation chunks). For context, this gap is an order of magnitude smaller than the ~0.03–0.09 seed-to-seed noise found in the repeated-seed study — the models are not overfit to their 200-chunk validation set in any way that shows up here. This is a reassuring, positive result: the small model/data budget used throughout this project produces representations that hold up on genuinely unseen material, not just the specific chunks used to pick the "best" checkpoint during training.

## Does the attention-type ranking hold out-of-sample?

**Yes, exactly.** Test-loss ranking: CosFormer (5.0347) < Gaussian (5.0457) < Cosine (5.0986) — identical ordering to validation loss, at both the 50-epoch repeated-seed study and the 100-epoch check. The CosFormer-vs-Gaussian gap on test (0.011) is essentially the same magnitude as on validation (0.0107) — consistent with this being the same underlying tie already flagged as unresolved at n=1 seed, not something that resolves itself out-of-sample. Cosine's gap to the other two (~0.05–0.06 on test) is likewise consistent with its validation-loss gap and with the repeated-seed study's conclusion that Cosine is robustly the worst of the three.

**Bottom line:** this test-set evaluation doesn't change any conclusion reached so far — it corroborates them on a much larger, held-out sample, which is exactly what you want a test-set check to do. It closes the "out-of-sample performance" gap in the project's methodology without overturning anything.

## Caveats

- **These are the single-seed (seed 0) 100-epoch checkpoints** — the same statistical caveat about the Gaussian/CosFormer tie applies here as everywhere else; this evaluation doesn't add seed diversity, it adds data diversity (a different, larger, disjoint chunk set).
- **Test set is evaluated once per checkpoint, deterministically** (dropout is off in `eval()` mode, no shuffling) — there is no run-to-run variance to report for a fixed checkpoint; the checkpoint itself is the only source of variance, which is exactly what the repeated-seed study already characterizes.
- **CE loss only** — the MSE/NTL checkpoints from the original 9-run matrix were not re-evaluated here (not requested, and several of those checkpoints were also single, non-seed-controlled runs from before the checkpoint-overwriting issue was fixed).

## What was and wasn't touched

**New:** everything under `mps_experiments/test_evaluation/` (script, log, results CSV, plot, this file).

**Not touched:** `mps_experiments_100ep/` checkpoints were only read (`map_location="cpu"`, no `load_state_dict` mutation persisted anywhere), never written to. No file under `mps_experiments/checkpoints/`, `mps_experiments/standardization_ablation/`, or `mps_experiments/repeated_seeds/` was touched.

## Files

- `evaluate_test_set.py` — the evaluation script; safe to re-run, read-only w.r.t. all checkpoints.
- `make_test_eval_plot.py` — regenerates the plot from `results/test_set_summary.csv`.
- `logs/test_evaluation.log` — full stdout from the evaluation run.
- `results/test_set_summary.csv` — the table above in machine-readable form.
- `plots/val_vs_test_comparison.png` — grouped bar chart, validation vs. test loss, all 3 attention types.
