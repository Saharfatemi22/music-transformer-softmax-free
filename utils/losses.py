"""
utils/losses.py
────────────────
Loss functions for MusicTransformer.

Three modes (controlled by cfg['loss_type']):

  "ce"      — standard cross-entropy on all tokens (ignore PAD).
              Baseline: identical to the original code.

  "mse"     — MSE on ordinal tokens (pitch, duration, velocity) +
              CE on structural tokens (BAR, POSITION, BOS, EOS).
              Replaces CE with regression for the numerical part of the vocab.

  "ce+ntl"  — CE everywhere + λ·MSE on ordinal tokens.
              Number Token Loss from Zausinger et al. (ICML 2025):
              adds a regression penalty on top of CE without removing it.

Ordinal token ranges in our 196-token REMI vocabulary:
  NOTE_ON   IDs 20–147  → pitch values 0–127  (normalised to [0, 1])
  DURATION  IDs 148–163 → duration values 1–16 (normalised to [0, 1])
  VELOCITY  IDs 164–195 → velocity bins 0–31   (normalised to [0, 1])

All other tokens (PAD=0, BOS, EOS, BAR, POSITION) are structural — they
have no numerical ordering so only CE applies to them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.vocabulary import (
    VOCAB_SIZE,
    NOTE_ON_OFFSET, N_PITCHES,
    DUR_OFFSET,     N_DURATIONS,
    VEL_OFFSET,     N_VELOCITIES,
)


def build_token_value_tensor(device: torch.device) -> torch.Tensor:
    """
    Build a (VOCAB_SIZE,) float tensor mapping each token ID to its
    normalised numerical value in [0, 1], or -1.0 for non-ordinal tokens.

    Called once before training; the result is reused every batch.
    """
    values = torch.full((VOCAB_SIZE,), -1.0)

    # NOTE_ON: pitch 0–127 → [0, 1]
    for i in range(N_PITCHES):
        values[NOTE_ON_OFFSET + i] = i / (N_PITCHES - 1)       # 0/127 … 127/127

    # DURATION: values 1–16 → [0, 1]  (index 0 = duration 1, index 15 = duration 16)
    for i in range(N_DURATIONS):
        values[DUR_OFFSET + i] = i / (N_DURATIONS - 1)         # 0/15 … 15/15

    # VELOCITY: bins 0–31 → [0, 1]
    for i in range(N_VELOCITIES):
        values[VEL_OFFSET + i] = i / (N_VELOCITIES - 1)        # 0/31 … 31/31

    return values.to(device)


def _mse_ordinal(logits_flat: torch.Tensor,
                 targets_flat: torch.Tensor,
                 token_values: torch.Tensor) -> torch.Tensor:
    """
    MSE between the model's expected token value and the true token value.
    Only computed for positions where the target is an ordinal token
    (token_values[target] >= 0).

    logits_flat:  (M, vocab_size)  — M = B*seq_len, already flattened
    targets_flat: (M,)
    token_values: (vocab_size,)    — precomputed normalised values

    Returns a scalar tensor (0.0 if no ordinal tokens present).
    """
    mask = token_values[targets_flat] >= 0.0        # True for ordinal targets
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits_flat.device)

    probs    = torch.softmax(logits_flat[mask], dim=-1)                  # (K, V)
    expected = (probs * token_values.clamp(min=0.0)).sum(dim=-1)         # (K,)
    true_val = token_values[targets_flat[mask]]                          # (K,)

    return F.mse_loss(expected, true_val)


def combined_loss(logits: torch.Tensor,
                  targets: torch.Tensor,
                  cfg: dict,
                  token_values: torch.Tensor) -> torch.Tensor:
    """
    Compute training loss according to cfg['loss_type'].

    logits:       (B, N, vocab_size)
    targets:      (B, N)
    token_values: (vocab_size,) from build_token_value_tensor()

    Returns a scalar loss tensor.
    """
    B_N  = logits.view(-1, logits.size(-1))   # (B*N, vocab_size)
    tgt  = targets.view(-1)                   # (B*N,)

    loss_type = cfg.get("loss_type", "ce")

    # ── "ce": baseline, unchanged ───────────────────────────────────────────
    if loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=0)(B_N, tgt)

    # ── "mse": replace CE with MSE for ordinal tokens ───────────────────────
    elif loss_type == "mse":
        # Structural mask: non-PAD tokens that are NOT ordinal
        ordinal_mask    = token_values[tgt] >= 0.0
        structural_mask = (~ordinal_mask) & (tgt != 0)   # exclude PAD (ID 0)

        mse_part = _mse_ordinal(B_N, tgt, token_values)

        if structural_mask.sum() > 0:
            # Plain CE for structural tokens — ignore_index not needed (PAD excluded)
            ce_part = F.cross_entropy(B_N[structural_mask], tgt[structural_mask])
        else:
            ce_part = torch.tensor(0.0, device=logits.device)

        # Equal-weight average of the two components.
        # Note: CE ≈ 5.0 at the start; MSE on [0,1] ≈ 0.08–0.25.
        # The gap is expected — watch both curves separately when analysing.
        return (ce_part + mse_part) / 2.0

    # ── "ce+ntl": CE everywhere + λ·MSE on ordinal (NTL paper) ─────────────
    elif loss_type == "ce+ntl":
        lam     = cfg.get("ntl_lambda", 0.5)
        ce_loss = nn.CrossEntropyLoss(ignore_index=0)(B_N, tgt)
        ntl     = _mse_ordinal(B_N, tgt, token_values)
        return ce_loss + lam * ntl

    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type!r}. "
            f"Choose from 'ce', 'mse', 'ce+ntl'."
        )
