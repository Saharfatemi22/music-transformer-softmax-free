"""
attention/cosine_attention.py
──────────────────────────────
Cosine (Cottention) attention — NO SOFTMAX.

Directly implements the Cottention paper:
  "Linear Transformers with Cosine Attention"
  Mongaras et al., arXiv 2024

Core idea (Equations 3–4 of the paper):
  CosAttention(Q, K, V) = (1 / s^σ(m)) ⊙ N(Q) · N(K)ᵀ · V

where N(X) = X / ‖X‖₂  (L2 normalisation per row vector).

Key properties (directly from the paper):
  • No softmax — ever.
  • Cosine similarity ∈ [−1, +1], so negative attention is possible.
  • A learned scalar m per head (init=0.5) stabilises training.
  • Can be reformulated as an RNN with O(1) inference memory:
      H_t = H_{t−1} + outer(K_t, V_t)
      O_t = inner(H_t, Q_t)

For training (forward pass), the cumulative-sum formulation is used,
giving O(N·d²) complexity instead of O(N²·d).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineAttention(nn.Module):
    """
    Multi-head cosine attention (Cottention), softmax-free.

    Args:
        d_model:    total model dimension
        n_heads:    number of attention heads
        m_init:     initial value of stability scalar m (default 0.5)
        dropout:    dropout rate on output projection
        causal:     if True, use causal (autoregressive) masking
        standardize_qk: if True, per-token mean/std standardise Q and K
                        (across d_model, before the head split) prior to the
                        existing L2 normalisation. Rogue-dimension ablation
                        from Timkey & van Schijndel (EMNLP 2021) — OFF by
                        default, does not change existing behaviour.
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 m_init:  float = 0.5,
                 dropout: float = 0.1,
                 causal:  bool  = True,
                 standardize_qk: bool = False):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.causal  = causal
        self.standardize_qk = standardize_qk

        self.W_Q  = nn.Linear(d_model, d_model, bias=False)
        self.W_K  = nn.Linear(d_model, d_model, bias=False)
        self.W_V  = nn.Linear(d_model, d_model, bias=False)
        self.W_O  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # Learned stability scalar m, one per head.
        # Paper: "initialized to 0.5 for each attention head"
        # We store the raw parameter; stability divisor = s^sigmoid(m).
        self.m = nn.Parameter(torch.full((n_heads,), m_init))

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-token standardisation across d_model: subtract the mean and
        divide by the std of each token's own vector, before L2 normalisation.
        Removes any single dimension's disproportionate contribution to the
        cosine similarity (rogue-dimension fix, Timkey & van Schijndel 2021).

        x: (B, N, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:    (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        B, N, _ = x.shape

        Q = self.W_Q(x)   # (B, N, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        if self.standardize_qk:
            Q = self._standardize(Q)
            K = self._standardize(K)

        # Split into heads: (B, N, d_model) → (B, H, N, d_head)
        def split_heads(t):
            return t.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        Q = split_heads(Q)   # (B, H, N, d_head)
        K = split_heads(K)
        V = split_heads(V)

        # ── Step 1: L2-normalise Q and K (makes it cosine similarity) ───────
        # From Equation 3: N(X) = X / ‖X‖₂
        Q = F.normalize(Q, p=2, dim=-1)   # (B, H, N, d_head)
        K = F.normalize(K, p=2, dim=-1)

        # ── Step 2: Stability normalisation of V ─────────────────────────────
        # From Equation 4: divide V by s^σ(m)
        # σ(m) ∈ (0,1)  so  s^σ(m) ∈ (1, s)  — model learns how much to divide
        sigma_m = torch.sigmoid(self.m)              # (H,) ∈ (0,1)
        sigma_m = sigma_m.view(1, self.n_heads, 1, 1)  # broadcast shape
        V_stable = V / (N ** sigma_m.clamp(1e-6, 1.0 - 1e-6))

        # ── Step 3: Causal cosine attention via cumulative sum ────────────────
        # From Equation 7 of the paper:
        #   O = [(V̄ ⊙ K̄).cumsum(dim=−3) ⊙ Q̄].sum(dim=−1)
        #
        # This is the linear-memory version — avoids the N×N matrix.
        # Shape gymnastics to broadcast:
        #   V̄: (B, H, N, d_head, 1)
        #   K̄: (B, H, N, 1,     d_head)
        # outer product: (B, H, N, d_head, d_head)
        # cumsum along N: (B, H, N, d_head, d_head)
        # inner with Q̄ (B, H, N, 1, d_head): sum last dim → (B, H, N, d_head)

        V_bar = V_stable.unsqueeze(-1)          # (B, H, N, d_head, 1)
        K_bar = K.unsqueeze(-2)                 # (B, H, N, 1,      d_head)
        Q_bar = Q.unsqueeze(-2)                 # (B, H, N, 1,      d_head)

        # Outer product V ⊗ K: (B, H, N, d_head, d_head)
        outer = V_bar * K_bar

        if self.causal:
            # Cumulative sum = running hidden state H_t
            # H_t = H_{t-1} + outer(K_t, V_t)   ← RNN update from paper Eq 8
            cumsum = outer.cumsum(dim=2)          # (B, H, N, d_head, d_head)
        else:
            # Bidirectional: just sum over all positions
            cumsum = outer.sum(dim=2, keepdim=True).expand_as(outer)

        # O_t = (H_t ⊙ Q̄).sum(−1)  ← RNN readout from paper Eq 8
        out = (cumsum * Q_bar).sum(dim=-1)       # (B, H, N, d_head)

        # ── Step 4: Merge heads and project ──────────────────────────────────
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_model)
        out = self.W_O(out)
        out = self.drop(out)

        return out

    def inference_step(self,
                       x_t: torch.Tensor,
                       H:   torch.Tensor) -> tuple:
        """
        Single-step inference using the RNN formulation (Equation 8).
        O(1) memory — does NOT grow with sequence length.

        Call this when generating music note-by-note.

        Args:
            x_t: current token embedding (batch, 1, d_model)
            H:   hidden state from previous step (batch, n_heads, d_head, d_head)
                 Initialise with zeros: H = torch.zeros(B, H, d_head, d_head)

        Returns:
            out: output for this step (batch, 1, d_model)
            H:   updated hidden state (batch, n_heads, d_head, d_head)
        """
        B = x_t.shape[0]

        Q_t = F.normalize(self.W_Q(x_t).view(B, self.n_heads, 1, self.d_head), p=2, dim=-1)
        K_t = F.normalize(self.W_K(x_t).view(B, self.n_heads, 1, self.d_head), p=2, dim=-1)
        V_t = self.W_V(x_t).view(B, self.n_heads, 1, self.d_head)

        # Update hidden state: H_t = H_{t-1} + outer(K_t, V_t)
        outer = V_t.squeeze(2).unsqueeze(-1) * K_t.squeeze(2).unsqueeze(-2)
        H = H + outer   # (B, H, d_head, d_head)

        # Readout: O_t = (H_t ⊙ Q̄).sum(−1)
        out = (H * Q_t.squeeze(2).unsqueeze(-2)).sum(dim=-1)  # (B, H, d_head)

        out = out.view(B, 1, self.d_model)
        out = self.W_O(out)
        return out, H


if __name__ == "__main__":
    B, N, d_model, n_heads = 2, 64, 128, 4
    attn = CosineAttention(d_model=d_model, n_heads=n_heads, causal=True)
    x = torch.randn(B, N, d_model)
    out = attn(x)
    print(f"CosineAttention: input {x.shape} → output {out.shape}")
    assert out.shape == (B, N, d_model)
    print("  PASSED ✓  (no softmax used)")

    # Test RNN inference step
    d_head = d_model // n_heads
    H = torch.zeros(B, n_heads, d_head, d_head)
    x_t = torch.randn(B, 1, d_model)
    out_t, H_new = attn.inference_step(x_t, H)
    print(f"  Inference step: {out_t.shape},  H: {H_new.shape}  ✓")
