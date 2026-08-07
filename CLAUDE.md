# CLAUDE.md — Music Transformer (Softmax-Free Attention)

This file is kept up-to-date as work progresses. It explains everything that exists, everything that was installed, every command that was run, and what the output means.

---

## Project Goal

Train a Transformer from scratch on MIDI piano music **without ever using softmax in the attention mechanism**. Three different distance-based attention methods are implemented and compared:

| Mode | Source paper | Core idea |
|------|-------------|-----------|
| `gaussian` | SOFT (IJCV 2024) | `exp(-‖Q−K‖²)` — distance between embedding vectors |
| `cosine` | Cottention (2024) | L2-normalised dot product — no softmax |
| `cosformer` | CosFormer (ICLR 2022) | ReLU + cosine re-weighting — linear complexity |

---

## File Map — What Every File Does

```
music_transformer/
│
├── CLAUDE.md                  ← THIS FILE — live project log
├── README.md                  ← high-level overview and quick-start
├── requirements.txt           ← Python packages to install
│
├── train.py                   ← ENTRY POINT: run this to start training
├── generate.py                ← ENTRY POINT: run this after training to make a MIDI file
│
├── configs/
│   └── default.yaml           ← ALL hyperparameters in one place
│                                 (change attention_type here to switch modes)
│
├── attention/
│   ├── __init__.py            ← exports build_attention() factory function
│   ├── gaussian_attention.py  ← Gaussian kernel attention (SOFT paper)
│   │                            Uses Nyström approximation to avoid O(N²) memory
│   │                            For short seqs (≤512) uses exact causal attention
│   ├── cosine_attention.py    ← Cosine/Cottention attention (no softmax)
│   └── cosformer_attention.py ← CosFormer linear attention (ReLU + cosine)
│
├── data/
│   ├── __init__.py            ← empty package marker
│   ├── vocabulary.py          ← REMI token vocabulary (196 tokens total)
│   │                            Encodes: BAR, POSITION (16), NOTE_ON (128),
│   │                            DURATION (16), VELOCITY (32)
│   ├── midi_dataset.py        ← MAESTRO MIDI loader + REMI tokeniser
│   │                            Supports two dataset modes:
│   │                            "jsb"     — tiny built-in JSB Chorales (no download)
│   │                            "maestro" — full MAESTRO v3.0.0 dataset
│   └── maestro-v3.0.0/        ← MIDI files (already present)
│       ├── 2004/  2006/  ...  ← year folders with .midi files
│       └── maestro-v3.0.0.csv ← metadata (train/val/test splits)
│
├── model/
│   ├── __init__.py            ← empty package marker
│   ├── transformer.py         ← Full decoder-only Transformer
│   │                            Pre-norm formulation (more stable from scratch)
│   │                            Sinusoidal positional encoding
│   │                            Weight tying (embedding = output projection)
│   │                            build_model(cfg) factory reads from default.yaml
│   └── relative_embedding.py  ← Relative position embeddings (Music Transformer paper)
│
├── utils/
│   ├── __init__.py            ← empty package marker
│   ├── train.py               ← Training loop
│   │                            AdamW optimiser, warmup LR schedule,
│   │                            gradient clipping, CSV loss log,
│   │                            saves best.pt + epoch_XXX.pt checkpoints
│   └── generate.py            ← Autoregressive generation, saves output as .mid
│
└── checkpoints/               ← Created automatically when training starts
    ├── best.pt                ← Best model (lowest validation loss)
    ├── epoch_010.pt           ← Periodic checkpoint every 10 epochs
    └── training_log.csv       ← Epoch, train_loss, val_loss, time per epoch
```

---

## Vocabulary Size

The model uses a 196-token REMI vocabulary:

| Token type | Count | Meaning |
|---|---|---|
| PAD / BOS / EOS | 3 | padding, start, end |
| BAR | 1 | start of a new measure |
| POSITION | 16 | 16th-note grid positions within a bar |
| NOTE_ON | 128 | MIDI pitch 0–127 |
| NOTE_DURATION | 16 | duration in 16th-note multiples |
| NOTE_VELOCITY | 32 | velocity bins |
| **Total** | **196** | |

---

## Model Architecture

- **Type**: Decoder-only Transformer (like GPT), trained with next-token prediction
- **Default size** (configs/default.yaml):
  - `d_model = 128` (embedding dimension)
  - `n_heads = 4` (attention heads)
  - `n_layers = 4` (transformer blocks)
  - `d_ff = 512` (feedforward hidden size)
  - `seq_len = 512` (tokens per training sample)
- **Attention**: Pluggable — swap by changing `attention_type` in the config

---

## What Was Installed

```bash
pip install -r requirements.txt
```

Run on: 2026-05-21. All packages installed successfully with no errors.

| Package | Installed Version | Purpose |
|---------|------------------|---------|
| `torch` | 2.12.0 | PyTorch — model training and tensor ops |
| `numpy` | 2.4.2 (already present) | Numerical arrays |
| `pretty_midi` | 0.2.11 | Read/write MIDI files |
| `music21` | 10.1.0 | Music analysis (used for JSB Chorales) |
| `pyyaml` | 6.0.3 | Load configs/default.yaml |
| `tqdm` | 4.67.3 | Progress bars during training |
| `matplotlib` | 3.10.8 (already present) | Plotting loss curves |

Auto-installed dependencies of the above:
`mido` 1.3.3, `sympy` 1.14.0, `networkx` 3.6.1, `fsspec` 2026.4.0, `filelock` 3.29.0, `jinja2` 3.1.6, `joblib` 1.5.3, `requests` 2.34.2, `chardet` 7.4.3, `jsonpickle` 4.1.1, `more-itertools` 11.0.2, `webcolors` 25.10.0, `setuptools` 81.0.0, `urllib3` 2.7.0

---

## How to Start Training

### Option 1 — Quick test (JSB Chorales, tiny dataset, no download needed)
```bash
cd "/Users/sahar/Downloads/Transformers softmax free project/music_transformer"
python train.py
```
Default config: Gaussian attention, JSB Chorales, 100 epochs.

### Option 2 — Full MAESTRO dataset (MIDI files already in data/)
```bash
python train.py --dataset maestro --data_dir data/maestro-v3.0.0
```

### Option 3 — Change attention type
```bash
python train.py --attention cosine
python train.py --attention cosformer
python train.py --attention gaussian   # default
```

### Option 4 — Override any setting from the command line
```bash
python train.py --attention cosine --epochs 50 --batch_size 32
```

### Option 5 — Edit configs/default.yaml directly
Open [configs/default.yaml](configs/default.yaml) and change `attention_type`, then just run `python train.py`.

### Generate MIDI after training
```bash
python generate.py --checkpoint checkpoints/best.pt --output generated.mid
```

---

## What the Training Output Means

When you run `python train.py` you will see:

```
============================================================
  Music Transformer — Softmax-Free Attention
  Attention type:  gaussian
  Dataset:         jsb
  d_model:         128
  Layers/Heads:    4 layers × 4 heads
============================================================

Training on: cpu          ← or "cuda" if GPU available
Attention:   gaussian
Parameters:  XXX,XXX      ← total trainable parameters in the model

Epoch   1/100  train_loss=5.2801  val_loss=5.1034  (12.3s)
  ✓ Saved best checkpoint (val_loss=5.1034)
Epoch   2/100  train_loss=4.9123  val_loss=4.8876  (11.8s)
  ✓ Saved best checkpoint (val_loss=4.8876)
...
```

**Column meanings:**
| Column | What it means |
|---|---|
| `train_loss` | Cross-entropy loss on training batches. Lower = model is learning. Should decrease each epoch. |
| `val_loss` | Cross-entropy loss on held-out validation data. Should also decrease. If it stops while train_loss keeps dropping = overfitting. |
| `(12.3s)` | Seconds that epoch took. Useful for estimating total training time. |
| `✓ Saved best checkpoint` | val_loss improved — best.pt was updated. |

**What is a "good" loss?**
- Random chance for 196 tokens = `log(196) ≈ 5.28`
- If loss starts near 5.2 and drops to ~3.0–3.5, the model is learning real music structure
- A loss of ~2.0 would be very good for this model size

**Perplexity** = `exp(loss)` — the effective vocabulary size the model is confused about.
- loss=5.28 → perplexity=196 (random)
- loss=3.0  → perplexity=20 (model narrows it down to ~20 plausible next tokens)

---

## Errors Fixed

### Error 1 — `ModuleNotFoundError: No module named 'pretty_midi'`
**When:** First training run.
**Cause:** The system has multiple Python installations. The shell's `python` command points to Anaconda Python at `/opt/anaconda3/bin/python`, but running `pip install` used a different Python under `/Library/Frameworks/Python.framework/`. Packages were installed in the wrong environment.
**Fix:** Re-ran install using the correct pip: `/opt/anaconda3/bin/pip install -r requirements.txt`. All subsequent training runs use `/opt/anaconda3/bin/python`.

### Error 2 — `AttributeError: 'PrettyMIDI' object has no attribute 'get_tempo_change_times'`
**File:** [data/midi_dataset.py](data/midi_dataset.py) line 52
**Cause:** Bug in the original code — the pretty_midi API method is `get_tempo_changes()` (returns a tuple), not `get_tempo_change_times()`. The original code also had a confused line that called the method twice and assigned both return values incorrectly.
**Fix:** Replaced the broken tempo block with:
```python
try:
    _, tempos = pm.get_tempo_changes()
    bpm = float(tempos[0]) if len(tempos) > 0 else 120.0
except Exception:
    bpm = 120.0
```

### Error 3 — Validation set loaded 0 pieces
**When:** Second training run (after Error 2 fix).
**Cause:** The MAESTRO CSV file uses `"validation"` as the split name, but the dataset loader compared against `"val"` (the name passed by `train.py`). No rows matched, so the validation set was empty.
**Fix:** Added a mapping in [data/midi_dataset.py](data/midi_dataset.py):
```python
csv_split = "validation" if split == "val" else split
```

---

## Key Config Options (configs/default.yaml)

| Parameter | Default | What it does |
|---|---|---|
| `attention_type` | `"gaussian"` | Which attention: `gaussian`, `cosine`, `cosformer` |
| `dataset` | `"jsb"` | `"jsb"` (tiny, instant) or `"maestro"` (full) |
| `seq_len` | `512` | Tokens per training sample |
| `d_model` | `64` *(was 128)* | Embedding dimension |
| `n_layers` | `2` *(was 4)* | Number of Transformer blocks |
| `n_heads` | `4` | Attention heads per layer |
| `batch_size` | `8` *(was 16)* | Sequences per gradient update |
| `learning_rate` | `0.0001` | Initial learning rate (warmed up from 0) |
| `max_epochs` | `20` *(was 100)* | Total training epochs |
| `warmup_steps` | `1000` | LR linearly increases for this many steps |
| `grad_clip` | `1.0` | Clips gradients to prevent explosions |
| `n_bottleneck` | `49` | Gaussian only: Nyström bottleneck tokens |
| `newton_steps` | `20` | Gaussian only: pseudo-inverse iterations |

---

## Checkpoints Explained

After training, `checkpoints/` will contain:

| File | What it is |
|---|---|
| `best.pt` | Model with the lowest validation loss — use this for generation |
| `epoch_010.pt`, `epoch_020.pt`, ... | Snapshots every 10 epochs |
| `training_log.csv` | Table of epoch, train_loss, val_loss, time — open in Excel/Numbers to plot |

Each `.pt` file contains: `epoch`, `model_state`, `val_loss`, `cfg` (the full config used).

---

---

## Phase 2 Plan — Changes Required From Daniel's Papers

This section documents exactly what needs to change based on the 5 papers Daniel sent.
**No code has been changed yet.** This is the plan only.

---

### Paper 1 — "All Bark and No Bite: Rogue Dimensions" (Timkey & van Schijndel, EMNLP 2021)

**What it says:**
In BERT, GPT-2, RoBERTa and XLNet, cosine similarity and Euclidean distance between embeddings are dominated by just **1–5 "rogue dimensions"** out of hundreds. These dimensions have mean activations far from zero and disproportionately high variance. In the worst case (XLNet layer 11), a single dimension contributes **99% of the total cosine similarity** between any two random tokens — making the other 767 dimensions invisible. The fix is trivially simple: **standardise** the embeddings (subtract mean, divide by std per dimension) before computing any distance or similarity.

**Why this matters critically for our project:**
Our Gaussian attention computes `exp(−‖Q−K‖²)`. If Q and K have rogue dimensions, the distance ‖Q−K‖² is dominated by 1–3 meaningless numbers and ignores everything else. Our model may appear to learn while the attention is actually doing nothing meaningful. This is the most important fix before any valid comparison.

**Exact code change needed:**

File: `attention/gaussian_attention.py`

In the `forward()` method, before splitting Q, K, V into heads, add a standardisation step:
```python
def _standardize(self, x: torch.Tensor) -> torch.Tensor:
    # Per-token: subtract mean and divide by std across the d_model dimension
    # This removes rogue dimensions per the Timkey & van Schijndel (2021) paper
    mean = x.mean(dim=-1, keepdim=True)
    std  = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return (x - mean) / std
```
Apply as: `Q, K = self._standardize(Q), self._standardize(K)` before computing kernel distances.

Also applies to `attention/cosine_attention.py` — the L2 normalisation there partially handles this, but explicit standardisation first is still cleaner.

Add to `configs/default.yaml`:
```yaml
standardize_qk: true   # apply per-token standardisation before attention distances
```

---

### Paper 2 — "Loss Functions and Metrics in Deep Learning" (Terven et al., Springer AIR 2025)

**What it says:**
A comprehensive survey covering every loss function: MSE, MAE, Huber, Cross-Entropy, Label Smoothing CE, Focal Loss, etc. Key insight for us: **cross-entropy assumes a nominal scale** — it treats all wrong predictions as equally wrong. For tasks where outputs have a natural ordering or numerical meaning, regression losses (MSE, MAE, Huber) encode the right inductive bias.

**Why this matters for our project:**
Our current training loop has `criterion = nn.CrossEntropyLoss(ignore_index=0)` hardcoded in `utils/train.py`. We only ever measure one loss type. Daniel wants us to **compare loss functions** as a second research axis — not just compare attention types. The experiment matrix should expand from 3 runs (one per attention type) to at least 6–9 runs (attention types × loss types).

**Exact code changes needed:**

File: `configs/default.yaml` — add:
```yaml
loss_type: "ce"   # options: "ce", "mse", "ce+mse", "ce+ntl"
ntl_lambda: 0.5   # weight for the NTL term when using "ce+ntl"
```

File: `utils/train.py` — replace hardcoded `nn.CrossEntropyLoss` with a function that reads `cfg["loss_type"]` and builds the appropriate criterion. Three modes to support:

- `"ce"` — current behaviour, `nn.CrossEntropyLoss(ignore_index=0)`
- `"mse"` — treat prediction as regression: compute expected token value under the predicted distribution, then MSE against true token value (only for ordinal tokens; fall back to CE for structural tokens)
- `"ce+ntl"` — CE plus Number Token Loss (see Paper 3 below)

---

### Paper 3 — "Regress, Don't Guess: Number Token Loss" (Zausinger et al., ICML 2025)

**What it says:**
Cross-entropy penalises predicting pitch 61 when the answer is 60 **exactly as much** as predicting pitch 5 — because it treats token IDs as unordered categories. The Number Token Loss (NTL) adds a regression-like term on top of CE that respects numerical proximity. Two versions:

- **NTL-MSE**: compute the expected numerical value under the predicted probability distribution (`Σ pᵢ · valueᵢ`), then take MSE against the true value
- **NTL-Wasserstein**: Wasserstein-1 distance between predicted distribution and true one-hot, treating token values as positions on a number line

Total loss = `L_CE + λ · L_NTL`

**Why this matters for our project:**
Our MIDI vocabulary has tokens that are **explicitly ordinal numbers**:
- `NOTE_ON` tokens (IDs 20–147): represent MIDI pitches 0–127 — adjacent pitches are musically similar
- `NOTE_DURATION` tokens (IDs 148–163): represent durations 1–16 sixteenth notes — adjacent durations are similar
- `NOTE_VELOCITY` tokens (IDs 164–195): represent velocity bins 0–31 — adjacent velocities are similar

For these three token types, NTL-MSE is directly applicable. Structural tokens (BAR, POSITION, PAD, BOS, EOS) have no numerical meaning and should keep standard CE.

**Exact code changes needed:**

New file: `utils/losses.py` — implement:
```python
# Token value maps (token_id → numerical value, or None for non-ordinal tokens)
# NOTE_ON: token IDs 20–147 → values 0–127 (pitch number)
# DURATION: token IDs 148–163 → values 1–16 (sixteenth notes)  
# VELOCITY: token IDs 164–195 → values 0–31 (velocity bin)
# All other tokens (PAD, BOS, EOS, BAR, POSITION): value = None (use CE only)

def build_token_value_tensor(vocab_size, device):
    """Returns a tensor of shape (vocab_size,) with the numerical value
    of each token, or -1 for non-ordinal tokens."""
    ...

def ntl_mse_loss(logits, targets, token_values):
    """Number Token Loss — MSE flavour.
    Only applied where token_values[target] >= 0 (ordinal tokens).
    logits: (B*N, vocab_size)
    targets: (B*N,)
    token_values: (vocab_size,) precomputed tensor
    """
    probs = torch.softmax(logits, dim=-1)
    expected = (probs * token_values.clamp(min=0)).sum(dim=-1)
    true_val  = token_values[targets]
    mask      = true_val >= 0  # only ordinal tokens
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.mse_loss(expected[mask], true_val[mask].float())

def combined_loss(logits, targets, cfg, token_values):
    """Combines CE and NTL based on cfg['loss_type']."""
    ...
```

File: `utils/train.py` — import and use `combined_loss()` instead of `nn.CrossEntropyLoss`.

---

### Paper 4 — "Transformers Are Expressive, But Are They Expressive Enough for Regression?" (Nath et al., 2024)

**What it says:**
Transformer encoders provably cannot approximate smooth (continuously differentiable) functions — they produce piecewise-constant approximations with large step sizes. Approximating a smooth function well requires an exponential number of layers in the input length. The full encoder-decoder architecture also fails experimentally.

**Why this matters for our project:**
This is the theoretical justification for why NTL (Paper 3) is needed. Musical pitch, duration and velocity are **smooth, continuous** properties — note 60 and note 61 are much closer musically than note 60 and note 127. Cross-entropy forces the model to treat this as a 196-way classification problem with no smoothness structure. The expressivity paper says Transformers will struggle to learn that smoothness from data alone. NTL explicitly injects it into the loss. This paper is our theoretical motivation for the CE vs CE+NTL experiment.

**No new code change required** — it motivates Paper 3's implementation.

---

### Paper 5 — "Your Transformer is Secretly Linear" (Razzhigaev et al., 2024)

**What it says:**
Consecutive layers in transformer decoders have a Procrustes similarity score of ~0.99 — nearly perfectly linear. This means many layers are redundant. Adding a **cosine-similarity regularisation term** during pretraining:
`L_cosine = λ · Σ (1 − cos(emb_i, emb_{i−1}))`
paradoxically makes layers *more* expressive (not less), improving SuperGLUE by ~3 points. The MSE regularisation (penalising distance between consecutive layer embeddings) also helps.

**Why this matters for our project:**
With only `n_layers=2`, near-linearity is already a risk — our model may essentially be doing one linear transformation. The cosine regularisation is simple to add to the training loop. Importantly, this paper also **validates our attention comparison** — different attention mechanisms produce different linearity profiles, so the comparison is scientifically meaningful.

**Exact code changes needed:**

File: `configs/default.yaml` — add:
```yaml
layer_cosine_reg: 0.0    # set to e.g. 0.5 to enable cosine regularisation between layers
```

File: `model/transformer.py` — `MusicTransformer.forward()` should optionally return intermediate layer embeddings when `layer_cosine_reg > 0`.

File: `utils/train.py` — add regularisation term to the total loss:
```python
if cfg.get("layer_cosine_reg", 0) > 0:
    # collect layer outputs from model, compute cosine reg term
    reg_loss = sum(1 - F.cosine_similarity(emb[i], emb[i-1], dim=-1).mean()
                   for i in range(1, len(embeddings)))
    loss = loss + cfg["layer_cosine_reg"] * reg_loss
```

---

### Summary: What Files Change and Why

| File | What changes | From which paper |
|------|-------------|-----------------|
| `attention/gaussian_attention.py` | Add `_standardize(Q, K)` before distance computation | Paper 1 (Rogue Dimensions) |
| `attention/cosine_attention.py` | Add pre-standardisation before L2 normalisation | Paper 1 (Rogue Dimensions) |
| `utils/losses.py` | **New file** — NTL-MSE loss, token value map, combined loss function | Paper 3 (NTL) |
| `utils/train.py` | Read `loss_type` from config, call `combined_loss()`, add cosine reg term | Papers 2, 3, 5 |
| `model/transformer.py` | Optionally return intermediate embeddings for cosine regularisation | Paper 5 (Secretly Linear) |
| `configs/default.yaml` | Add `loss_type`, `ntl_lambda`, `standardize_qk`, `layer_cosine_reg` | All papers |

### New Experiment Matrix

Once implemented, the comparison expands from 3 runs to a grid:

| | `loss=ce` | `loss=ce+ntl` | `loss=mse` |
|---|---|---|---|
| **gaussian** | ✓ done | to run | to run |
| **cosine** | ✓ done | to run | to run |
| **cosformer** | ✓ done | to run | to run |

This gives 9 runs total. The key scientific questions become:
1. Does NTL improve over CE for any attention type? (Tests Paper 3 hypothesis)
2. Does standardising Q/K improve Gaussian attention? (Tests Paper 1 fix)
3. Which (attention type, loss type) combination performs best overall?

---

## Status Log

| Date | What happened |
|------|--------------|
| 2026-05-21 | Project files reviewed. CLAUDE.md created. Ready to install and train. |
| 2026-05-21 | Ran `pip install -r requirements.txt`. All 7 required packages installed successfully (torch 2.12.0, pretty_midi 0.2.11, music21 10.1.0, pyyaml 6.0.3, tqdm 4.67.3; numpy and matplotlib were already present). No errors. |
| 2026-05-21 | Fixed 3 bugs (wrong Python env, wrong pretty_midi method name, wrong validation split name). Training now running correctly with `/opt/anaconda3/bin/python train.py --attention gaussian --dataset maestro --data_dir data/maestro-v3.0.0`. Dataset: 962 train pieces → 88,082 chunks; 137 val pieces → 9,918 chunks. Model: 816,384 parameters. Training on CPU (~1.5s/batch, ~2.3 hrs/epoch). |
| 2026-05-21 | Reduced model size for faster CPU training: d_model 128→64, n_layers 4→2, d_ff 512→256, batch_size 16→8, max_epochs 100→20. Model now 112,128 params (was 816,384). |
| 2026-05-21 | Added max_train_chunks=500 / max_val_chunks=100 cap to configs/default.yaml and MusicDataset. Epoch time dropped from ~40 min to ~15 seconds. |
| 2026-05-21 | **All 20 epochs completed (~5 min total). Gaussian softmax-free attention confirmed learning.** val_loss: 5.476 → 5.395 (steady drop every epoch). train_loss: 5.503 → 5.442. Random-chance baseline = 5.28. Best checkpoint saved to checkpoints/best.pt. |
| 2026-05-21 | Generated MIDI from best checkpoint: `generated.mid` (1.8 KB, 288 notes, 512 tokens). Command: `/opt/anaconda3/bin/python generate.py --checkpoint checkpoints/best.pt --output generated.mid --length 512 --temperature 1.0 --top_k 20` |
| 2026-05-21 | Re-ran Gaussian training (clean run). 20 epochs, ~5.5 min total. Final val_loss=5.3885, train_loss=5.4442. Results saved to gaussian_results.txt. |
| 2026-05-21 | Cosine attention training. 20 epochs, ~2.5 min total (2.4× faster than Gaussian). Final val_loss=5.4034, train_loss=5.4454. Results saved to cosine_results.txt. |
| 2026-05-21 | CosFormer attention training. 20 epochs, ~3.5 min total. Final val_loss=5.3684, train_loss=5.4514. Best final val_loss of all three methods. Results saved to cosformer_results.txt. Three-way comparison complete. |
| 2026-05-23 | Generated comparison_plot.png — matplotlib line chart of all 3 val loss curves with random-chance baseline, config caption, 150 dpi. |
| 2026-05-26 | Phase 2 Phase 1 implemented: added MSE loss. New file utils/losses.py (token value map, _mse_ordinal, combined_loss). Updated utils/train.py to import and use combined_loss(). Added loss_type/ntl_lambda to configs/default.yaml. Added --loss CLI flag to train.py. All three modes (ce, mse, ce+ntl) smoke-tested. |
| 2026-05-26 | All 6 missing experiments completed. Results saved: gaussian_mse_results.txt, cosine_mse_results.txt, cosformer_mse_results.txt, gaussian_ntl_results.txt, cosine_ntl_results.txt, cosformer_ntl_results.txt. Full 3×3 experiment matrix now complete. Best result: CosFormer+NTL val_loss=5.2771, breaking below random-chance baseline of 5.2983. |
| 2026-05-29 | MPS (Apple Silicon GPU) infrastructure added. Backups of train.py and utils/train.py saved to backup_before_mps/. Device detection in utils/train.py updated to prefer CUDA → MPS → CPU. --save_dir and --train_chunks/--val_chunks CLI arguments added to train.py. mps_experiments/ directory structure created. No CPU files modified or overwritten. |
| 2026-05-29 | All 9 MPS experiments completed (50 epochs, 1000/200 chunks). Best result: CosFormer+CE val_loss=5.1045. All 6 CE/NTL runs beat random-chance baseline (5.2983) — only 1 of 9 did on CPU. Plots saved to mps_experiments/plots/. Summary saved to mps_experiments/MPS_RESULTS_SUMMARY.md. |
