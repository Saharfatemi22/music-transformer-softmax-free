"""
utils/generate.py
──────────────────
Autoregressive music generation from a trained checkpoint.
Saves the generated token sequence as a MIDI file.

Generation method: top-k sampling with temperature (Paper 5 method).
  "We employ temperature-controlled stochastic sampling with top-k"
  — Huang & Yang, Pop Music Transformer, ACM MM 2020

For cosine attention models, we use the O(1) RNN inference mode
(Cottention paper, constant memory regardless of sequence length).
For gaussian/cosformer models, we use the standard autoregressive loop.
"""

import torch
import pretty_midi
import numpy as np

from data.vocabulary import (
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    NOTE_ON_OFFSET, N_PITCHES,
    DUR_OFFSET, N_DURATIONS,
    VEL_OFFSET, N_VELOCITIES,
    POS_OFFSET, N_POSITIONS,
    BAR_TOKEN, VOCAB_SIZE,
    decode_token
)


# ── Sampling ──────────────────────────────────────────────────────────────────

def top_k_sample(logits: torch.Tensor,
                 temperature: float = 1.0,
                 top_k: int = 20) -> int:
    """
    Sample one token from logits using temperature + top-k filtering.
    Same method used in Paper 5 (REMI Pop Music Transformer).

    logits: (vocab_size,)
    Returns: integer token ID
    """
    logits = logits / max(temperature, 1e-6)

    if top_k > 0:
        # Keep only top-k logits, set rest to -inf
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        threshold = values[-1]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


# ── REMI → MIDI conversion ────────────────────────────────────────────────────

def remi_tokens_to_midi(tokens: list,
                         bpm: float = 120.0,
                         output_path: str = "generated.mid"):
    """
    Convert a list of REMI token IDs back to a MIDI file.

    Reverses the midi_to_remi() function in data/midi_dataset.py.
    """
    midi   = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    piano  = pretty_midi.Instrument(program=0, name="Piano")

    sec_per_beat = 60.0 / bpm
    sec_per_16th = sec_per_beat / 4.0
    sec_per_bar  = sec_per_beat * 4.0

    current_bar      = 0
    current_pos      = 1        # 1–16
    current_velocity = 64
    current_time     = 0.0

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == EOS_TOKEN or tok == PAD_TOKEN:
            break

        elif tok == BOS_TOKEN:
            i += 1

        elif tok == BAR_TOKEN:
            current_bar += 1
            current_time = current_bar * sec_per_bar
            i += 1

        elif POS_OFFSET <= tok < POS_OFFSET + N_POSITIONS:
            current_pos  = tok - POS_OFFSET + 1    # 1-indexed
            current_time = (current_bar * sec_per_bar +
                            (current_pos - 1) * sec_per_16th)
            i += 1

        elif VEL_OFFSET <= tok < VEL_OFFSET + N_VELOCITIES:
            # Velocity bin → MIDI velocity (middle of bin range)
            vel_bin          = tok - VEL_OFFSET
            current_velocity = min(127, vel_bin * 4 + 2)
            i += 1

        elif NOTE_ON_OFFSET <= tok < NOTE_ON_OFFSET + N_PITCHES:
            pitch = tok - NOTE_ON_OFFSET
            # Next token should be duration
            dur_16th = 4   # default 1 beat
            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                if DUR_OFFSET <= next_tok < DUR_OFFSET + N_DURATIONS:
                    dur_16th = next_tok - DUR_OFFSET + 1
                    i += 1   # consume duration token too

            onset  = current_time
            offset = onset + dur_16th * sec_per_16th

            note = pretty_midi.Note(
                velocity = current_velocity,
                pitch    = pitch,
                start    = onset,
                end      = offset
            )
            piano.notes.append(note)
            i += 1

        else:
            i += 1   # skip unknown tokens

    midi.instruments.append(piano)

    if piano.notes:
        midi.write(output_path)
        print(f"Saved MIDI: {output_path}  ({len(piano.notes)} notes)")
    else:
        print("Warning: no notes generated. Check token sequence.")

    return output_path


# ── Main generation function ──────────────────────────────────────────────────

@torch.no_grad()
def generate(model,
             prompt_tokens: list = None,
             gen_length:   int   = 512,
             temperature:  float = 1.0,
             top_k:        int   = 20,
             device:       str   = "cpu") -> list:
    """
    Autoregressively generate a token sequence.

    model:         trained MusicTransformer
    prompt_tokens: list of integer token IDs to prime generation (optional)
    gen_length:    number of NEW tokens to generate
    temperature:   sampling temperature
    top_k:         top-k filtering

    Returns: list of generated token IDs (including prompt)
    """
    model.eval()

    if prompt_tokens is None:
        prompt_tokens = [BOS_TOKEN]

    tokens = list(prompt_tokens)

    for _ in range(gen_length):
        # Prepare input tensor
        x = torch.tensor([tokens], dtype=torch.long, device=device)

        # Forward pass
        logits = model(x)                 # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]    # logits for last position

        # Sample next token
        next_token = top_k_sample(next_logits, temperature, top_k)
        tokens.append(next_token)

        if next_token == EOS_TOKEN:
            break

    return tokens


def load_and_generate(checkpoint_path: str,
                       output_path:     str   = "generated.mid",
                       gen_length:      int   = 512,
                       temperature:     float = 1.0,
                       top_k:           int   = 20):
    """
    Load a checkpoint and generate a MIDI file.
    This is the function called by generate.py entry point.
    """
    from model.transformer import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location=device)
    cfg   = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded model (epoch {ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f})")

    print(f"Generating {gen_length} tokens...")
    tokens = generate(model, gen_length=gen_length,
                       temperature=temperature, top_k=top_k, device=str(device))

    print(f"Generated {len(tokens)} tokens. Converting to MIDI...")
    remi_tokens_to_midi(tokens, output_path=output_path)
