# Music Transformer — Comparing Softmax-Free Attention and Loss Functions

Training decoder-only Transformers from scratch on symbolic piano music (MAESTRO)
to compare three softmax-free attention mechanisms and two training loss
functions, evaluating training efficiency and out-of-sample performance.

This is a research project by Sahar Fatemi and Daniel Armani. The full academic
writeup is in [`paper/manuscript.md`](paper/manuscript.md); a detailed,
chronological log of every experiment, bug fix, and finding is in
[`CLAUDE.md`](CLAUDE.md) and [`HISTORY.md`](HISTORY.md).

## Research Goal

Compare three distance/similarity-based attention mechanisms, none of which use
softmax, and two training loss functions, on a from-scratch symbolic music
generation task:

**Attention mechanisms:**
1. **Gaussian** — Gaussian kernel over query-key distance (SOFT paper, Lu et al., IJCV 2024)
2. **Cosine** — L2-normalized cosine similarity (Cottention, Mongaras et al., 2024)
3. **CosFormer** — ReLU non-negativity + cosine positional re-weighting (Qin et al., ICLR 2022)

**Loss functions:**
1. **Cross-entropy (CE)** — standard next-token prediction loss
2. **CE + Number Token Loss (NTL)** — CE plus an auxiliary regression term on
   ordinal tokens (pitch, duration, velocity), following Zausinger et al. (2025)

We also test whether the "rogue dimensions" documented in large pretrained text
models (Timkey and van Schijndel, 2021) appear in these smaller,
from-scratch-trained music models, and whether correcting for them helps.

## Key Findings

- **Gaussian and CosFormer attention are statistically equivalent** on
  validation loss, at both 50 and 100 epochs and across repeated seeds; **Cosine
  attention is consistently and robustly the worst of the three**. CosFormer is
  the fastest to train.
- **Number Token Loss provides no measurable benefit** over plain cross-entropy
  at this model and data scale, on either next-token prediction quality or the
  ordinal numerical-closeness objective it is designed to improve, once
  compared under a fixed, apples-to-apples metric.
- **No severe rogue dimensions were found.** Token embeddings are close to
  isotropic; the mild dimensional concentration found in query/key
  representations does not produce a meaningful improvement when corrected via
  standardization.
- All three attention mechanisms generalize well to a held-out test set, with
  no meaningful overfitting to the validation set.

See [`paper/manuscript.md`](paper/manuscript.md) for the full methodology,
results, and discussion, and `mps_experiments*/` for the underlying data,
plots, and per-experiment findings documents.

## Project Structure

```
music_transformer/
├── attention/                  # The three attention mechanisms
│   ├── gaussian_attention.py   # SOFT-style Gaussian kernel (no softmax)
│   ├── cosine_attention.py     # Cottention-style cosine attention (no softmax)
│   └── cosformer_attention.py  # CosFormer-style linear attention (no softmax)
├── data/
│   ├── midi_dataset.py         # MAESTRO MIDI loader + REMI tokenizer
│   └── vocabulary.py           # REMI token vocabulary (196 tokens)
├── model/
│   └── transformer.py          # Decoder-only Transformer (pre-norm, causal)
├── utils/
│   ├── train.py                # Training loop
│   ├── losses.py                # CE / CE+NTL loss functions
│   └── generate.py             # Autoregressive generation -> MIDI
├── configs/
│   └── default.yaml            # All hyperparameters in one place
├── train.py                    # Entry point: python train.py
├── generate.py                 # Entry point: python generate.py
├── mps_experiments/            # 50-epoch experiments: original matrix, rogue-
│                                # dimension diagnostic, standardization
│                                # ablation, repeated-seed study, test-set eval,
│                                # ordinal-MSE eval (each with its own
│                                # results/, plots/, and FINDINGS.md)
├── mps_experiments_100ep/      # 100-epoch experiments: tie resolution,
│                                # NTL loss-function comparison
├── paper/
│   └── manuscript.md           # Full academic writeup (in progress)
├── CLAUDE.md                   # Detailed project log: setup, architecture,
│                                # every bug fixed, every experiment phase
└── HISTORY.md                  # Chronological change log from the
                                 # rogue-dimension work onward
```

## Dataset — MAESTRO v3.0.0

Download the MIDI-only version (~57 MB):
https://magenta.tensorflow.org/datasets/maestro

After downloading, extract so you have:
```
data/maestro-v3.0.0/
    2004/  2006/  2008/  ...  (folders of .mid files)
    maestro-v3.0.0.csv        (metadata file)
```

Or for quick local testing with JSB Chorales (tiny, no download needed), set
`dataset: "jsb"` in `configs/default.yaml`.

## Quick Start

```bash
pip install -r requirements.txt

# Train with default config (Gaussian attention, cross-entropy loss)
python train.py --dataset maestro --data_dir data/maestro-v3.0.0

# Change attention mechanism
python train.py --attention cosine    # or cosformer, gaussian
python train.py --attention gaussian --loss ce+ntl   # CE + Number Token Loss

# Reproducibility: control the random seed
python train.py --attention cosformer --seed 1

# Optional Q/K standardization ablation (Gaussian/Cosine only)
python train.py --attention gaussian --standardize_qk

# Generate a MIDI file from a trained checkpoint
python generate.py --checkpoint checkpoints/best.pt --output generated.mid
```

Run `python train.py --help` for the full list of CLI overrides (epochs, batch
size, save directory, train/val chunk caps, etc.), or edit
`configs/default.yaml` directly.

## Notes on This Repository

Trained model checkpoints (`.pt` files), the MAESTRO dataset, and third-party
paper PDFs are excluded from version control (see `.gitignore`) to keep the
repository lean; all experiment results are preserved as CSVs, plots, and
markdown findings documents alongside the code that produced them.
