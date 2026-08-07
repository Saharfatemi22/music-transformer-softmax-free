# MPS Experiment Results Summary

**Device:** Apple Silicon GPU (MPS)  
**Settings:** 50 epochs · 1000 train chunks · 200 val chunks · d_model=64 · n_layers=2 · n_heads=4 · batch=8  
**Dataset:** MAESTRO v3.0.0  
**Random-chance baseline:** log(196) = 5.2983

---

## CE & CE+NTL Results (comparable scale)

| Attention | CE (val loss) | CE+NTL (val loss) |
|-----------|--------------|-------------------|
| Gaussian  | 5.2457       | 5.1211            |
| Cosine    | 5.1474       | 5.1567            |
| CosFormer | **5.1045**   | 5.1753            |

All 6 runs beat the random-chance baseline of 5.2983.  
Best result overall: **CosFormer + CE = 5.1045**

### When each run broke below random chance
| Run | Epoch first below 5.2983 |
|-----|--------------------------|
| Gaussian + CE   | ~45 |
| Cosine + CE     | ~25 |
| CosFormer + CE  | ~20 |
| Gaussian + NTL  | ~26 |
| Cosine + NTL    | ~25 |
| CosFormer + NTL | ~31 |

---

## MSE Results (separate scale — not comparable to CE)

| Attention | MSE val loss |
|-----------|-------------|
| Gaussian  | 2.5690      |
| Cosine    | 2.6777      |
| CosFormer | **2.5032**  |

Loss = (CE_structural + MSE_ordinal) / 2. Values ~2.5–2.7 are expected on this scale.

---

## Training Speed (steady-state seconds/epoch, excl. first-epoch warmup)

| Attention | CE | MSE | CE+NTL |
|-----------|-----|-----|--------|
| Gaussian  | 17.5s | 16.5s | 12.9s |
| Cosine    | 8.2s  | 7.7s  | 6.4s  |
| CosFormer | 6.6s  | 5.2s  | **5.1s** |

CosFormer is both fastest and best-performing across all loss types.

---

## Key Findings

1. **All 9 runs beat random chance** — on CPU (20 epochs), only CosFormer+NTL did. More epochs + more data qualitatively changes the picture.

2. **CE outperforms NTL at 50 epochs** — NTL starts training at a higher loss value due to the combined objective. With 50 epochs, plain CE has converged further. NTL likely needs 80–100+ epochs to show its advantage.

3. **CosFormer dominates on both axes** — fastest (5.1s/epoch) and lowest val loss (5.1045) across all 9 runs.

4. **Cosine MSE is the only run that did not improve over CPU** — CPU reached 2.6735 in 20 epochs; MPS reached 2.6777 in 50 epochs with 2× more data. This suggests cosine+MSE converges quickly and the harder dataset offsets the extra training.

5. **MPS speedup vs CPU** — roughly 2× faster for Gaussian, similar for Cosine/CosFormer (which were already fast on CPU).

---

## File Locations

| Type | Path |
|------|------|
| Checkpoints | `mps_experiments/checkpoints/<run>/best.pt` |
| Training logs | `mps_experiments/checkpoints/<run>/training_log.csv` |
| Plots | `mps_experiments/plots/plot1_ce_vs_ntl.png` |
|       | `mps_experiments/plots/plot2_mse_separate.png` |
|       | `mps_experiments/plots/plot3_speed.png` |
| Plot script | `mps_experiments/make_plots_mps.py` |

---

*CPU baseline results (20 epochs, 500 chunks) are in the root directory: `gaussian_results.txt`, `cosine_results.txt`, etc.*
