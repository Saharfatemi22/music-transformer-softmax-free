# Gaussian vs. CosFormer Tie Resolution — Findings

**Question:** the repeated-seed study at 50 epochs (`mps_experiments/repeated_seeds/`) and the single-seed 100-epoch check (`RESULTS_100EP.md`) both left Gaussian and CosFormer statistically indistinguishable — CosFormer numerically ahead by a margin far smaller than the seed-to-seed noise. Does adding 2 more seeds at the more mature 100-epoch point resolve this, or does the tie hold?

**Method:** reused the existing seed-0 100-epoch checkpoints (`mps_experiments_100ep/checkpoints/{gaussian_ce,cosformer_ce}/`, untouched) and added seeds 1 and 2 for both attention types — 4 new runs, same settings as every other 100-epoch run (CE loss, no standardization, 1000 train chunks, 200 val chunks, `d_model=64, n_layers=2, n_heads=4, d_ff=256, batch_size=8`, MPS). Cosine was deliberately excluded — already conclusively established as the worst of the three, nothing to resolve there.

---

## Result: the tie stands. Decisively.

| Attention | seed 0 | seed 1 | seed 2 | **Mean** | **Std dev** |
|---|---|---|---|---|---|
| Gaussian | 5.0361 | 4.9847 | 4.8896 | **4.9701** | **0.0743** |
| CosFormer | 5.0254 | 4.9963 | 4.8826 | **4.9681** | **0.0755** |

**Mean difference (CosFormer − Gaussian): −0.0020** — about **3% of one standard deviation**. This is as close to a dead heat as this kind of experiment can produce. Full data: `results/tie_resolution_summary.csv`. Plot: `plots/tie_resolution_mean_std.png` (the two bars are visually indistinguishable, with near-identical error bars and overlapping individual seed points).

**Paired, same-seed comparison confirms there is no consistent winner:**

| Seed | Gaussian | CosFormer | CosFormer − Gaussian |
|---|---|---|---|
| 0 | 5.0361 | 5.0254 | −0.0107 (CosFormer ahead) |
| 1 | 4.9847 | 4.9963 | +0.0116 (Gaussian ahead) |
| 2 | 4.8896 | 4.8826 | −0.0070 (CosFormer ahead) |

Two of three seeds favor CosFormer, one favors Gaussian, and all three margins are small. This is the signature of two methods with genuinely equivalent performance at this model/data scale, not one method being better and noise obscuring it — a directionally consistent-but-small effect (like the standardization ablation's Gaussian result) would look different from this.

## A secondary, now twice-replicated observation

**Seed 2 produced the best result for both attention types again** (Gaussian 4.8896, CosFormer 4.8826 — both clearly ahead of seeds 0 and 1), exactly mirroring the pattern already seen in the 50-epoch repeated-seed study, where seed 2 was also the best seed for both Gaussian and CosFormer (though notably the *worst* seed for Cosine there). This is now observed at two different training lengths (50 and 100 epochs) with the same three seed values, which makes it very unlikely to be a coincidence — something about seed 2's specific initialization/batch-order interacts favorably with both Gaussian's and CosFormer's optimization landscape on this data, independent of training length. This remains a descriptive observation, not something this experiment design can explain causally.

## All three seeds still decreasing at epoch 100

Every one of the 6 new/reused runs (Gaussian × 3 seeds, CosFormer × 3 seeds) was still monotonically decreasing through the last 10 epochs — none had converged. This reinforces the earlier finding: absolute loss values here should be read as "where training was after 100 epochs on this data budget," not as each model's ceiling.

## Runtime note

Gaussian seed 0's 77.9-minute runtime (previously flagged as containing a host-scheduling anomaly) is now clearly confirmed as an outlier: seed 1 took 25.6 minutes and seed 2 took 36.6 minutes, both much closer to the expected steady-state cost. Full timing: `results/tie_resolution_summary.csv`.

## Conclusion

**This project cannot currently distinguish Gaussian from CosFormer attention on validation loss at this model/data scale** — 3 seeds at both 50 and 100 epochs, plus the held-out test-set evaluation (which showed the same near-tie), all agree. Any paper claim should state this directly: "Gaussian and CosFormer performed statistically equivalently across all tested conditions; CosFormer's numerical edge in any single run should not be interpreted as evidence of superiority." What *does* remain robust and well-supported across every experiment in this project is that **Cosine attention is clearly and consistently the worst of the three**, and that **Q/K standardization does not meaningfully help any attention type** at this scale.

Resolving the Gaussian/CosFormer tie further (if desired) would require either substantially more seeds (diminishing returns given how close the two already are) or a different signal entirely — e.g., training speed (where the two already differ: CosFormer is consistently faster) or downstream generation quality — rather than more validation-loss seeds.

## What was and wasn't touched

**New:** `mps_experiments_100ep/checkpoints/{gaussian_ce_seed1,gaussian_ce_seed2,cosformer_ce_seed1,cosformer_ce_seed2}/`, corresponding logs, `results/tie_resolution_summary.csv`, `plots/tie_resolution_mean_std.png`, `make_tie_resolution_plot.py`, `run_seed_extension.sh`, this file.

**Not touched:** the existing seed-0 checkpoints (`gaussian_ce/`, `cosformer_ce/`, `cosine_ce/`), `results/100ep_summary.csv`, `RESULTS_100EP.md`, and everything under `mps_experiments/` (original matrix, standardization ablation, repeated-seed study, test evaluation).

## Files

- `checkpoints/{gaussian_ce_seed1,gaussian_ce_seed2,cosformer_ce_seed1,cosformer_ce_seed2}/` — `best.pt`, 10 periodic epoch checkpoints, `training_log.csv` each.
- `logs/{gaussian_ce_seed1,gaussian_ce_seed2,cosformer_ce_seed1,cosformer_ce_seed2}.log`
- `results/tie_resolution_summary.csv` — all 6 runs (3 seeds × 2 attention types), val loss and runtime.
- `plots/tie_resolution_mean_std.png` — mean ± std bar chart with individual seed points overlaid.
- `make_tie_resolution_plot.py` — regenerates the CSV/plot; safe to re-run.
- `run_seed_extension.sh` — the orchestration script that ran the 4 new jobs sequentially.
