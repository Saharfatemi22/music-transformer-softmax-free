"""
data/vocabulary.py
──────────────────
REMI token vocabulary.

Based directly on Paper 5:
  "Pop Music Transformer: Beat-based Modeling and Generation
   of Expressive Pop Piano Compositions" (Huang & Yang, ACM MM 2020)

Token types (Table 1 of Paper 5):
  BAR           — marks the start of a new bar
  POSITION      — 1 of 16 positions within a bar (16th-note grid)
  NOTE_ON       — MIDI pitch 0–127
  NOTE_DURATION — duration in 16th-note multiples (1–16)
  NOTE_VELOCITY — quantised velocity (1–32 bins)

For simplicity this prototype uses a reduced vocabulary
without Chord/Tempo events. You can add those later.
"""

# ── Token type offsets ─────────────────────────────────────
# Each token is a single integer ID.
# We lay them out as contiguous ranges.

PAD_TOKEN   = 0    # padding (for batching variable-length sequences)
BOS_TOKEN   = 1    # beginning of sequence
EOS_TOKEN   = 2    # end of sequence

# BAR: 1 token
BAR_OFFSET       = 3
N_BAR            = 1          # just one BAR event
BAR_TOKEN        = BAR_OFFSET  # = 3

# POSITION: 16 tokens (positions 1–16 within a bar)
POS_OFFSET       = BAR_OFFSET + N_BAR    # = 4
N_POSITIONS      = 16

# NOTE_ON: 128 tokens (MIDI pitch 0–127)
NOTE_ON_OFFSET   = POS_OFFSET + N_POSITIONS   # = 20
N_PITCHES        = 128

# NOTE_DURATION: 16 tokens (1–16 sixteenth notes)
DUR_OFFSET       = NOTE_ON_OFFSET + N_PITCHES  # = 148
N_DURATIONS      = 16

# NOTE_VELOCITY: 32 tokens (32 velocity bins)
VEL_OFFSET       = DUR_OFFSET + N_DURATIONS    # = 164
N_VELOCITIES     = 32

# Total vocabulary size
VOCAB_SIZE = VEL_OFFSET + N_VELOCITIES   # = 196


# ── Helper functions ───────────────────────────────────────

def bar_token():
    """Returns the integer ID for a BAR event."""
    return BAR_TOKEN


def position_token(pos: int) -> int:
    """
    pos: 1–16 (position within bar on 16th-note grid)
    Returns the integer token ID.
    """
    assert 1 <= pos <= 16, f"Position must be 1–16, got {pos}"
    return POS_OFFSET + (pos - 1)


def note_on_token(pitch: int) -> int:
    """
    pitch: 0–127 (MIDI pitch number)
    Returns the integer token ID.
    """
    assert 0 <= pitch <= 127, f"Pitch must be 0–127, got {pitch}"
    return NOTE_ON_OFFSET + pitch


def duration_token(dur_16th: int) -> int:
    """
    dur_16th: note duration in 16th-note multiples (1–16)
    Returns the integer token ID.
    """
    dur_16th = max(1, min(16, dur_16th))
    return DUR_OFFSET + (dur_16th - 1)


def velocity_token(velocity: int) -> int:
    """
    velocity: MIDI velocity 0–127 → binned to 0–31
    Returns the integer token ID.
    """
    bin_idx = min(31, velocity // 4)
    return VEL_OFFSET + bin_idx


# ── Decode helpers (token ID → human-readable string) ─────

def decode_token(token_id: int) -> str:
    if token_id == PAD_TOKEN:
        return "PAD"
    if token_id == BOS_TOKEN:
        return "BOS"
    if token_id == EOS_TOKEN:
        return "EOS"
    if token_id == BAR_TOKEN:
        return "BAR"
    if POS_OFFSET <= token_id < POS_OFFSET + N_POSITIONS:
        return f"POS_{token_id - POS_OFFSET + 1}"
    if NOTE_ON_OFFSET <= token_id < NOTE_ON_OFFSET + N_PITCHES:
        return f"NOTE_ON_{token_id - NOTE_ON_OFFSET}"
    if DUR_OFFSET <= token_id < DUR_OFFSET + N_DURATIONS:
        return f"DUR_{token_id - DUR_OFFSET + 1}"
    if VEL_OFFSET <= token_id < VEL_OFFSET + N_VELOCITIES:
        return f"VEL_{token_id - VEL_OFFSET}"
    return f"UNKNOWN_{token_id}"


if __name__ == "__main__":
    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"BAR token:       {bar_token()}")
    print(f"POS 1 token:     {position_token(1)}")
    print(f"NOTE C4 (60):    {note_on_token(60)}")
    print(f"DUR 4 token:     {duration_token(4)}")
    print(f"VEL 80 token:    {velocity_token(80)}")
