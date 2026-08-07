# 100-Epoch Experiments — Results

**Scope:** Gaussian, Cosine, CosFormer, all with CE loss, **no Q/K standardization**, **single seed (0)**, 100 epochs, 1000 train chunks, 200 val chunks, same model size as every other MPS experiment (`d_model=64, n_layers=2, n_heads=4, d_ff=256, batch_size=8`). Chosen as the fastest path to a full 3-attention-type picture at 100 epochs (see prior turn's options); note this is a single-seed check, not a repeated-seed result — treat point differences smaller than the ~0.03–0.09 seed-to-seed std dev found in `mps_experiments/repeated_seeds/` as inconclusive.

Random-chance baseline: `log(196) = 5.2983`.

## Results

| Config | Val loss @ ep.50 | Val loss @ ep.100 | Δ (ep50→ep100) | Still decreasing at ep.100? | Total runtime |
|---|---|---|---|---|---|
| Gaussian + CE | 5.2156 | **5.0361** | −0.1795 | Yes | 77.9 min* |
| Cosine + CE | 5.2637 | **5.0917** | −0.1720 | Yes | 26.5 min |
| CosFormer + CE | 5.2019 | **5.0254** | −0.1765 | Yes | 13.1 min |

\* Gaussian's runtime includes a sustained host-machine slowdown (epochs 32–53 ran at 60–97s/epoch instead of the normal ~20–26s, plus isolated spikes at epochs 8 and 82) — a scheduling/contention artifact on this machine, not a training issue. Steady-state epoch time (~26s) is consistent with prior Gaussian runs; without the anomaly, total runtime would have been roughly 44 minutes.

Full data: `results/100ep_summary.csv`. Plot: `plots/100ep_val_loss_comparison.png`.

## Is more training still helping?

**Yes, clearly, for all three.** Every curve was still monotonically decreasing through the last 10 logged epochs, with no sign of flattening — the last-10-epoch deltas are all still substantial (roughly −0.02 to −0.03 per 10 epochs at the end of the run for each config). None of the three models has converged at 100 epochs on this data budget. This means more epochs (e.g. 150) would very likely continue to reduce validation loss further for all three, if that's the direction the user wants to go next — the user has asked to hold off on 150 epochs for now, so no such run was launched.

## How does the epoch-100 (single-seed) ranking compare to the epoch-50 repeated-seed (n=3) ranking?

| | Epoch-50, n=3 seeds (mean) | Epoch-100, seed 0 only |
|---|---|---|
| Ranking | CosFormer (5.1494) ≈ Gaussian (5.1580) [tied within 1 std] > Cosine (5.2823) [clearly worst] | CosFormer (5.0254) < Gaussian (5.0361) [gap 0.011] < Cosine (5.0917) [gap ~0.06] |

**The pattern is consistent, not contradicted:** CosFormer remains numerically best, Gaussian a close second, and Cosine clearly the worst of the three — at both training lengths. The CosFormer-vs-Gaussian gap at epoch 100 (0.011) is, as before, far smaller than the ~0.09 seed-to-seed standard deviation measured at epoch 50, so **this single seed cannot resolve the CosFormer-vs-Gaussian tie** — it simply shows the same tie persisting at 100 epochs for this one seed, which is itself useful information (the tie isn't an epoch-50-specific artifact) but not new statistical evidence on its own.

Cosine's relative gap to the other two did **narrow somewhat** in this single-seed comparison (from ~0.13 at the 3-seed epoch-50 mean down to ~0.06–0.066 at epoch 100 for this one seed) — but since this is one seed rather than three, this narrowing should not yet be read as "Cosine is catching up with more training"; it's equally consistent with this being seed 0's own relatively favorable draw for Cosine specifically (recall from the repeated-seed study that seed-to-seed variation interacts differently per attention type — Cosine's seed 0 was in fact its *best* of the three seeds tested there). Confirming whether Cosine's relative gap genuinely narrows with more training would need the same 3-seed treatment at 100 epochs.

## Sanity check

This run's seed-0 epoch-50 values (Gaussian 5.2156, Cosine 5.2637, CosFormer 5.2019) match the seed-0 values from the earlier repeated-seed study exactly, confirming the `--seed` flag and data pipeline are behaving consistently and reproducibly across separate invocations.

## What was and wasn't touched

**New:** everything under `mps_experiments_100ep/` (checkpoints, logs, results, plots, this file, `make_100ep_plots.py`).

**Not touched:** no file under `mps_experiments/` (the original 9-run matrix, the standardization ablation, or the repeated-seed study) was modified.

## Files

- `checkpoints/{gaussian_ce,cosine_ce,cosformer_ce}/` — `best.pt`, 10 periodic epoch checkpoints, `training_log.csv` each.
- `logs/{gaussian_ce,cosine_ce,cosformer_ce}.log` — full stdout per run.
- `results/100ep_summary.csv` — the comparison table above in machine-readable form.
- `plots/100ep_val_loss_comparison.png` — all 3 curves, epoch 50 marker, random-chance line.
- `make_100ep_plots.py` — regenerates the CSV/plot from the 3 training logs; safe to re-run.
