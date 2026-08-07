"""
data/midi_dataset.py
────────────────────
MIDI → REMI token sequence converter + PyTorch Dataset.

Supports two data sources:
  1. MAESTRO v3.0.0  (recommended, download from magenta.tensorflow.org)
  2. JSB Chorales    (tiny built-in dataset via music21, for quick testing)

REMI representation from Paper 5 (Huang & Yang, ACM MM 2020):
  Each note becomes:  BAR? → POSITION → VELOCITY → NOTE_ON → DURATION
  16th-note time grid, Q = 16 positions per bar (4/4 time).
"""

import os
import math
import random
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from data.vocabulary import (
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    bar_token, position_token, note_on_token, duration_token, velocity_token,
    VOCAB_SIZE
)


# ── REMI encoder ──────────────────────────────────────────────────────────────

def midi_to_remi(midi_path: str, ticks_per_beat: int = 480,
                 positions_per_bar: int = 16) -> List[int]:
    """
    Convert a MIDI file to a REMI token sequence.

    Returns a list of integer token IDs.
    Returns empty list if the file cannot be parsed.
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("Run:  pip install pretty_midi")

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return []

    # Estimate tempo (use first tempo change, or 120 BPM default)
    try:
        _, tempos = pm.get_tempo_changes()
        bpm = float(tempos[0]) if len(tempos) > 0 else 120.0
    except Exception:
        bpm = 120.0

    # Seconds per 16th note
    sec_per_beat     = 60.0 / bpm
    sec_per_16th     = sec_per_beat / 4.0
    sec_per_bar      = sec_per_beat * 4    # assumes 4/4

    # Collect all notes across all instruments
    all_notes = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            all_notes.append({
                "onset":    note.start,
                "duration": note.end - note.start,
                "pitch":    note.pitch,
                "velocity": note.velocity,
            })

    if not all_notes:
        return []

    # Sort by onset time
    all_notes.sort(key=lambda n: n["onset"])

    tokens = [BOS_TOKEN]
    current_bar = -1

    for note in all_notes:
        # Which bar does this note fall in?
        bar_num  = int(note["onset"] / sec_per_bar)
        # Which 16th-note position within the bar? (1–16)
        beat_in_bar = (note["onset"] % sec_per_bar) / sec_per_16th
        pos = int(beat_in_bar) + 1          # 1-indexed
        pos = max(1, min(positions_per_bar, pos))

        # Emit BAR token if we've entered a new bar
        if bar_num != current_bar:
            tokens.append(bar_token())
            current_bar = bar_num

        # POSITION
        tokens.append(position_token(pos))

        # VELOCITY (binned)
        tokens.append(velocity_token(note["velocity"]))

        # NOTE_ON
        tokens.append(note_on_token(note["pitch"]))

        # DURATION in 16th-note multiples (1–16)
        dur_16ths = max(1, round(note["duration"] / sec_per_16th))
        dur_16ths = min(16, dur_16ths)
        tokens.append(duration_token(dur_16ths))

    tokens.append(EOS_TOKEN)
    return tokens


# ── JSB Chorales fallback (no download needed) ────────────────────────────────

def get_jsb_sequences() -> List[List[int]]:
    """
    Returns REMI-like token sequences from J.S. Bach Chorales via music21.
    Used when dataset = "jsb" for quick local testing.
    Paper 4 (Music Transformer) also uses JSB Chorales.
    """
    try:
        from music21 import corpus
    except ImportError:
        raise ImportError("Run:  pip install music21")

    sequences = []
    works = corpus.getComposer("bach")[:20]   # use first 20 for speed

    for work_path in works:
        try:
            score = corpus.parse(work_path)
            tokens = [BOS_TOKEN]
            current_bar = -1

            # Flatten all parts into notes sorted by offset
            notes = []
            for part in score.parts:
                for note in part.flatten().notes:
                    offset_16th = int(note.offset * 4)   # offset in 16th notes
                    bar_num     = offset_16th // 16
                    pos         = (offset_16th % 16) + 1
                    duration_16 = max(1, min(16, int(note.duration.quarterLength * 4)))

                    pitch = note.pitch.midi if hasattr(note, 'pitch') else 60
                    notes.append({
                        "bar": bar_num, "pos": pos,
                        "pitch": pitch, "dur": duration_16, "vel": 80
                    })

            notes.sort(key=lambda n: (n["bar"], n["pos"]))

            for n in notes:
                if n["bar"] != current_bar:
                    tokens.append(bar_token())
                    current_bar = n["bar"]
                tokens.append(position_token(n["pos"]))
                tokens.append(velocity_token(n["vel"]))
                tokens.append(note_on_token(n["pitch"]))
                tokens.append(duration_token(n["dur"]))

            tokens.append(EOS_TOKEN)
            if len(tokens) > 10:
                sequences.append(tokens)
        except Exception:
            continue

    return sequences


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class MusicDataset(Dataset):
    """
    Loads MIDI files, converts to REMI token sequences,
    and returns fixed-length windows for training.

    Each item returned: (input_tokens, target_tokens)
      input:  tokens[0 : seq_len]
      target: tokens[1 : seq_len+1]   (next-token prediction)
    """

    def __init__(self,
                 dataset:   str  = "jsb",
                 data_dir:  Optional[str] = None,
                 seq_len:   int  = 512,
                 split:     str  = "train",    # "train", "val", "test"
                 max_chunks: Optional[int] = None):

        self.seq_len = seq_len
        self.sequences: List[List[int]] = []

        print(f"Loading {dataset} dataset ({split} split)...")

        if dataset == "jsb":
            all_seqs = get_jsb_sequences()
            # Simple 80/10/10 split
            n = len(all_seqs)
            if split == "train":
                seqs = all_seqs[:int(0.8 * n)]
            elif split == "val":
                seqs = all_seqs[int(0.8 * n):int(0.9 * n)]
            else:
                seqs = all_seqs[int(0.9 * n):]
            self.sequences = seqs

        elif dataset == "maestro":
            assert data_dir is not None, "Provide data_dir for MAESTRO"
            import csv

            csv_path = os.path.join(data_dir, "maestro-v3.0.0.csv")
            if not os.path.exists(csv_path):
                # Try v2 naming
                csv_path = os.path.join(data_dir, "maestro-v2.0.0.csv")

            # MAESTRO CSV uses "validation" but callers pass "val"
            csv_split = "validation" if split == "val" else split

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader if r.get("split", "train") == csv_split]

            for row in rows:
                midi_rel = row.get("midi_filename", "")
                midi_path = os.path.join(data_dir, midi_rel)
                if not os.path.exists(midi_path):
                    continue
                tokens = midi_to_remi(midi_path)
                if len(tokens) > seq_len + 1:
                    self.sequences.append(tokens)
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Use 'jsb' or 'maestro'.")

        # Build sliding-window chunks from full sequences
        self.chunks: List[List[int]] = []
        for seq in self.sequences:
            for start in range(0, len(seq) - seq_len, seq_len // 2):
                chunk = seq[start: start + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    self.chunks.append(chunk)

        if max_chunks is not None:
            self.chunks = self.chunks[:max_chunks]

        print(f"  Loaded {len(self.sequences)} pieces → {len(self.chunks)} chunks "
              f"of length {seq_len}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)   # input
        y = torch.tensor(chunk[1:],  dtype=torch.long)   # target (shifted by 1)
        return x, y


if __name__ == "__main__":
    ds = MusicDataset(dataset="jsb", seq_len=256)
    x, y = ds[0]
    print(f"Dataset size: {len(ds)}")
    print(f"x shape: {x.shape},  y shape: {y.shape}")
    print(f"First 10 tokens: {x[:10].tolist()}")
