"""
train.py  ← run this to start training
─────────────────────────────────────
Usage:
  python train.py                                         # default config
  python train.py --attention cosine                      # change attention
  python train.py --attention gaussian --dataset maestro --data_dir data/maestro-v3.0.0
  python train.py --config configs/default.yaml --attention cosformer
"""

import argparse
import random
import yaml
import numpy as np
import torch

from data.midi_dataset  import MusicDataset
from model.transformer  import build_model
from utils.train        import train


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Seed all RNGs that affect reproducibility: model init, dropout, data shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--attention",  type=str,  default=None,
                        choices=["gaussian", "cosine", "cosformer"],
                        help="Override attention_type from config")
    parser.add_argument("--loss",       type=str,  default=None,
                        choices=["ce", "mse", "ce+ntl"],
                        help="Override loss_type from config")
    parser.add_argument("--dataset",    type=str,  default=None,
                        choices=["jsb", "maestro"])
    parser.add_argument("--data_dir",   type=str,  default=None)
    parser.add_argument("--epochs",       type=int,  default=None)
    parser.add_argument("--batch_size",   type=int,  default=None)
    parser.add_argument("--save_dir",     type=str,  default=None,
                        help="Override checkpoint save directory")
    parser.add_argument("--train_chunks", type=int,  default=None,
                        help="Override max_train_chunks from config")
    parser.add_argument("--val_chunks",   type=int,  default=None,
                        help="Override max_val_chunks from config")
    parser.add_argument("--standardize_qk", action="store_true", default=None,
                        help="Enable Q/K standardization ablation (rogue-dimension fix, "
                             "gaussian/cosine only). Omit flag to use the config's default (off).")
    parser.add_argument("--seed",       type=int,  default=None,
                        help="Random seed for model init, dropout, and data shuffling "
                             "(default: config's 'seed' key, or 0 if unset)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Command-line overrides
    if args.attention  is not None: cfg["attention_type"] = args.attention
    if args.dataset    is not None: cfg["dataset"]        = args.dataset
    if args.data_dir   is not None: cfg["data_dir"]       = args.data_dir
    if args.epochs     is not None: cfg["max_epochs"]     = args.epochs
    if args.batch_size is not None: cfg["batch_size"]     = args.batch_size
    if args.loss       is not None: cfg["loss_type"]      = args.loss
    if args.save_dir     is not None: cfg["save_dir"]           = args.save_dir
    if args.train_chunks is not None: cfg["max_train_chunks"]   = args.train_chunks
    if args.val_chunks   is not None: cfg["max_val_chunks"]     = args.val_chunks
    if args.standardize_qk is not None: cfg["standardize_qk"]   = args.standardize_qk
    if args.seed        is not None: cfg["seed"]          = args.seed
    cfg.setdefault("seed", 0)

    set_seed(cfg["seed"])

    print("=" * 60)
    print(f"  Music Transformer — Softmax-Free Attention")
    print(f"  Attention type:  {cfg['attention_type']}")
    print(f"  Standardize QK:  {cfg.get('standardize_qk', False)}")
    print(f"  Seed:            {cfg['seed']}")
    print(f"  Dataset:         {cfg['dataset']}")
    print(f"  d_model:         {cfg['d_model']}")
    print(f"  Layers/Heads:    {cfg['n_layers']} layers × {cfg['n_heads']} heads")
    print("=" * 60)

    # Load datasets
    train_ds = MusicDataset(
        dataset    = cfg["dataset"],
        data_dir   = cfg.get("data_dir"),
        seq_len    = cfg["seq_len"],
        split      = "train",
        max_chunks = cfg.get("max_train_chunks")
    )
    val_ds = MusicDataset(
        dataset    = cfg["dataset"],
        data_dir   = cfg.get("data_dir"),
        seq_len    = cfg["seq_len"],
        split      = "val",
        max_chunks = cfg.get("max_val_chunks")
    )

    if len(train_ds) == 0:
        print("ERROR: No training data found. Check your data_dir path.")
        return

    # Build model
    model = build_model(cfg)
    print(f"\nModel parameters: {model.count_parameters():,}")

    # Train
    train(model, train_ds, val_ds, cfg)


if __name__ == "__main__":
    main()
