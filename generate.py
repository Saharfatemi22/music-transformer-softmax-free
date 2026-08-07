"""
generate.py  ← run this after training to produce MIDI
────────────────────────────────────────────────────────
Usage:
  python generate.py --checkpoint checkpoints/best.pt --output my_piece.mid
  python generate.py --checkpoint checkpoints/best.pt --length 1024 --temperature 0.9
"""

import argparse
from utils.generate import load_and_generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--output",       default="generated.mid",
                        help="Output MIDI file path")
    parser.add_argument("--length",       type=int,   default=512,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature",  type=float, default=1.0,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k",        type=int,   default=20,
                        help="Top-k sampling (Paper 5 uses top-k=20)")
    args = parser.parse_args()

    load_and_generate(
        checkpoint_path = args.checkpoint,
        output_path     = args.output,
        gen_length      = args.length,
        temperature     = args.temperature,
        top_k           = args.top_k,
    )


if __name__ == "__main__":
    main()
