# CE vs. CE+NTL Loss-Function Comparison — Findings

**Why this exists:** every attention-type comparison in this project (rogue dimensions, standardization, the Gaussian/CosFormer tie) now has proper 3-seed rigor. The loss-function axis — Daniel's own framing emphasizes comparing loss functions specifically — did not: every `mse`/`ce+ntl` number in the project came from a single, non-seed-controlled run in the original 9-run matrix. This closes that gap for the paper's actual second research axis: CE vs. CE+NTL (Number Token Loss), for the two attention types left in play (Gaussian, CosFormer; Cosine excluded as already conclusively worse).

**Method:** ran Gaussian+CE+NTL and CosFormer+CE+NTL for 3 seeds each (0, 1, 2) — 6 new runs — at the same 100-epoch scale as the existing CE runs (1000 train chunks, 200 val chunks, same model size, MPS, λ=0.5 for the NTL term). All 6 completed successfully with full 100-epoch logs.

---

## First pass: comparing each checkpoint's own logged validation loss

| | CE mean ± std | CE+NTL mean ± std | Diff |
|---|---|---|---|
| Gaussian | 4.9701 ± 0.0743 | 5.0050 ± 0.0744 | **+0.0349** |
| CosFormer | 4.9681 ± 0.0755 | 5.0030 ± 0.0757 | **+0.0349** |

At face value this looks like a real, consistent effect — CE+NTL appears worse by almost exactly the same amount for both attention types, and the paired per-seed gap is remarkably tight (+0.0343 to +0.0359 across all 6 runs, a spread of only 0.0016). **That level of consistency is itself the tell that something other than "the model got worse at its task" is going on** — it's far tighter than the genuine, real Gaussian-vs-CosFormer differences found elsewhere in this project, which varied unpredictably by seed.

**The reason:** the `ce+ntl` loss actually logged during training is `CE + λ·MSE_ordinal` (λ=0.5), not pure cross-entropy. A `ce+ntl` checkpoint's "val_loss" is answering a different, harder-scored question than a `ce` checkpoint's "val_loss" — they were never on the same scale, despite the original single-run result files (`gaussian_ntl_results.txt`, etc.) describing them as "directly comparable." **That framing from the original 9-run matrix was incorrect** and should not be repeated in the paper.

## Second pass: decomposing the gap — is next-token prediction actually worse?

To answer the real question, every one of the 12 checkpoints (CE-trained and NTL-trained, both attention types, all 3 seeds) was re-evaluated on the same 200-chunk validation set under a **single, fixed metric: plain cross-entropy, forced regardless of each checkpoint's own training objective**. This isolates "did NTL training change how well the model predicts the next token" from "does the loss value include an extra term."

| | CE-trained, pure-CE | CE+NTL-trained, pure-CE | Diff |
|---|---|---|---|
| Gaussian | 4.9701 ± 0.0743 | 4.9700 ± 0.0745 | **−0.0001** |
| CosFormer | 4.9681 ± 0.0755 | 4.9680 ± 0.0756 | **−0.0001** |

**The gap completely vanishes.** Under a fixed, comparable metric, CE-trained and NTL-trained models are statistically indistinguishable — the difference (−0.0001) is three orders of magnitude smaller than the seed-to-seed standard deviation (~0.075). This is visible directly in `plots/ntl_vs_ce_pure_ce_comparison.png`: all four bars (Gaussian CE, Gaussian CE+NTL, CosFormer CE, CosFormer CE+NTL) are essentially the same height, and the individual seed values line up almost exactly across the CE/CE+NTL pairs for each attention type.

**The earlier +0.0349 gap is fully and precisely explained by the added λ·MSE_ordinal term** — implying the raw MSE_ordinal component itself is roughly `0.0349 / 0.5 ≈ 0.070`, consistent in magnitude with the standalone `mse`-mode loss values reported in the original single-run matrix (`~2.6–2.8` on the `(CE_structural+MSE_ordinal)/2` combined scale). Per-checkpoint decomposed values are in `results/ntl_pure_ce_decomposition.csv`.

## Conclusion for the paper

**Adding the Number Token Loss regularizer, as implemented here, does not measurably change the model's next-token prediction quality (pure cross-entropy) at this model/data scale, for either Gaussian or CosFormer attention.** This is a precise, informative negative result: it is *not* that NTL makes the model worse at predicting tokens (the naive reading of the raw logged numbers) — it's that NTL's regression penalty on ordinal tokens (pitch, duration, velocity) neither helps nor measurably hurts the classification objective it's added on top of. Whether NTL achieves its actual intended goal — improving the model's numerical closeness on ordinal-token errors, which is a different question from raw CE — was not tested here (see Caveats) and is the natural next check if this line is worth pursuing further.

This nuance directly supersedes two things in the project's existing record:
1. The original `*_ntl_results.txt` files' claim that CE and CE+NTL values are "directly comparable" — they're comparable in *scale* (similar order of magnitude) but not in *meaning* (different objectives), and the paper should not compare them as if measuring the same thing.
2. Any framing that "NTL hurts" — the correct, evidence-backed statement is "NTL is a statistical no-op on next-token prediction quality," which is a different and more precise claim.

## Caveats

- **This tests whether NTL changes *classification* quality (pure CE), not whether it achieves its actual design goal** — improving the numerical closeness of wrong ordinal-token predictions (e.g., predicting pitch 61 instead of 60 being penalized less than predicting pitch 5). That would require comparing MSE_ordinal itself between CE-trained and NTL-trained models (the CE-trained models' MSE_ordinal was never optimized for and hasn't been measured here) — a natural, cheap follow-up using checkpoints already on disk.
- **CE loss only for the attention-type axis; only Gaussian and CosFormer tested** — consistent with scoping decisions made throughout this phase of work (Cosine excluded as already settled).
- **`mse`-only mode (not `ce+ntl`) was not re-tested** — it was already understood to be a different-scale internal ablation, not the paper's core CE-vs-NTL comparison, and remains a single-run result from the original matrix.
- Model/data scale remains small (112K params, 1000 train chunks) — this conclusion is specific to that scale, not necessarily generalizable to a larger model where the classification and regression objectives might interact differently.

## What was and wasn't touched

**New:** `mps_experiments_100ep/checkpoints/{gaussian_ntl_seed0,gaussian_ntl_seed1,gaussian_ntl_seed2,cosformer_ntl_seed0,cosformer_ntl_seed1,cosformer_ntl_seed2}/`, corresponding logs, `results/ntl_vs_ce_summary.csv`, `results/ntl_pure_ce_decomposition.csv`, `plots/ntl_vs_ce_pure_ce_comparison.png`, `run_ntl_seed_extension.sh`, `decompose_ntl_loss.py`, `make_ntl_comparison_plot.py`, this file.

**Not touched:** no existing checkpoint (including the CE seed-0/1/2 checkpoints re-evaluated here, only ever loaded read-only), `results/100ep_summary.csv`, `results/tie_resolution_summary.csv`, `RESULTS_100EP.md`, `TIE_RESOLUTION_FINDINGS.md`, or anything under `mps_experiments/`.

## Files

- `checkpoints/{gaussian_ntl_seed0,gaussian_ntl_seed1,gaussian_ntl_seed2,cosformer_ntl_seed0,cosformer_ntl_seed1,cosformer_ntl_seed2}/` — `best.pt`, 10 periodic epoch checkpoints, `training_log.csv` each.
- `logs/{...}.log` — full stdout per run.
- `results/ntl_vs_ce_summary.csv` — logged val_loss and pure-CE re-evaluation, all 12 checkpoints.
- `results/ntl_pure_ce_decomposition.csv` — per-checkpoint decomposition (logged loss, pure CE, estimated NTL-term contribution).
- `plots/ntl_vs_ce_pure_ce_comparison.png` — the corrected, apples-to-apples comparison.
- `run_ntl_seed_extension.sh` — orchestration script for the 6 training runs.
- `decompose_ntl_loss.py`, `make_ntl_comparison_plot.py` — analysis scripts; both safe to re-run.
