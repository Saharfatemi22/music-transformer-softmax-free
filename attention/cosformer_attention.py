"""
attention/cosformer_attention.py
─────────────────────────────────
CosFormer attention — NO SOFTMAX.

Directly implements:
  "CosFormer: Rethinking Softmax in Attention"
  Qin et al., ICLR 2022

Core idea (Equations 6–9 of the paper):
  Replace softmax with two properties:
    1. Non-negativity:    φ(x) = ReLU(x)    → Q' = ReLU(Q), K' = ReLU(K)
    2. Locality bias:     cos(π(i−j)/2M)    → nearby positions get higher weight

The Ptolemy decomposition makes the cos-reweighting decomposable:
  cos(π(i−j)/2M) = cos(πi/2M)·cos(πj/2M) + sin(πi/2M)·sin(πj/2M)

This means we can AVOID the N×N matrix:
  O = Q_cos (K_cos^T V) + Q_sin (K_sin^T V)
  Complexity: O(Nd²) instead of O(N²d)

Why this is relevant: it shows that removing softmax and using
a position-aware linear operation still achieves state-of-the-art
results (Long-Range Arena benchmark, ICLR 2022).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFormerAttention(nn.Module):
    """
    Multi-head CosFormer attention — softmax-free, O(N) complexity.

    Args:
        d_model:  total model dimension
        n_heads:  number of attention heads
        dropout:  dropout on output projection
        causal:   if True, use causal (autoregressive) cumulative sum
    """

    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 causal:  bool  = True):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.causal  = causal

        self.W_Q  = nn.Linear(d_model, d_model, bias=False)
        self.W_K  = nn.Linear(d_model, d_model, bias=False)
        self.W_V  = nn.Linear(d_model, d_model, bias=False)
        self.W_O  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def _cos_reweight(self,
                      Q: torch.Tensor,
                      K: torch.Tensor,
                      N: int,
                      M: int = None) -> tuple:
        """
        Apply ReLU + cosine re-weighting (Ptolemy decomposition).

        Q, K: (B, H, N, d_head)
        Returns: Q_cos, Q_sin, K_cos, K_sin  — each (B, H, N, d_head)
        """
        if M is None:
            M = N

        # Step 1: Non-negativity via ReLU  (from paper Section 2.4)
        Q = F.relu(Q)
        K = F.relu(K)

        # Step 2: Positional cosine reweighting
        # positions: 1, 2, ..., N
        positions = torch.arange(1, N + 1, dtype=Q.dtype, device=Q.device)
        angles    = math.pi * positions / (2.0 * M)   # πi / (2M)

        cos_w = torch.cos(angles).view(1, 1, N, 1)    # (1, 1, N, 1)
        sin_w = torch.sin(angles).view(1, 1, N, 1)

        Q_cos = Q * cos_w     # (B, H, N, d_head)
        Q_sin = Q * sin_w
        K_cos = K * cos_w
        K_sin = K * sin_w

        return Q_cos, Q_sin, K_cos, K_sin

    def _linear_attention(self,
                          Q_cos, Q_sin,
                          K_cos, K_sin,
                          V: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional linear attention (non-causal).
        Complexity: O(Nd²)

        Uses the matrix-product reordering trick from the paper (Eq 9):
          O = Q_cos (K_cos^T V) + Q_sin (K_sin^T V)
        """
        # (K^T V) first: d×d matrices — tiny!
        KcosTV = torch.matmul(K_cos.transpose(-2, -1), V)   # (B, H, d, d)
        KsinTV = torch.matmul(K_sin.transpose(-2, -1), V)

        # Numerator
        num = torch.matmul(Q_cos, KcosTV) + torch.matmul(Q_sin, KsinTV)

        # Denominator: normalise so rows don't explode
        Kcos_sum = K_cos.sum(dim=-2)                         # (B, H, d)
        Ksin_sum = K_sin.sum(dim=-2)
        denom = (Q_cos * Kcos_sum.unsqueeze(-2)).sum(-1, keepdim=True) + \
                (Q_sin * Ksin_sum.unsqueeze(-2)).sum(-1, keepdim=True)

        return num / denom.clamp(min=1e-6)

    def _causal_linear_attention(self,
                                  Q_cos, Q_sin,
                                  K_cos, K_sin,
                                  V: torch.Tensor) -> torch.Tensor:
        """
        Causal (autoregressive) linear attention using cumulative sum.
        Complexity: O(Nd²), no N×N matrix, no softmax.

        This is the music generation version: each position can only
        attend to itself and earlier positions.
        """
        B, H, N, d = Q_cos.shape

        # For each position t, we need:
        #   num_t   = Q_cos_t · cumsum_{s<=t}(K_cos_s^T V_s) + Q_sin_t · (...)
        # Achieved by cumsum of outer products:

        # K_cos_s^T ⊗ V_s: for each s, this is (d, d)
        # Stack over N:
        KcosV_outer = torch.matmul(
            K_cos.unsqueeze(-1),     # (B, H, N, d, 1)
            V.unsqueeze(-2)          # (B, H, N, 1, d)
        )                            # (B, H, N, d, d)

        KsinV_outer = torch.matmul(
            K_sin.unsqueeze(-1),
            V.unsqueeze(-2)
        )                            # (B, H, N, d, d)

        # Causal cumulative sum
        KcosV_cum = KcosV_outer.cumsum(dim=2)   # (B, H, N, d, d)
        KsinV_cum = KsinV_outer.cumsum(dim=2)

        # Multiply by queries: (B, H, N, 1, d) × (B, H, N, d, d) → (B, H, N, 1, d)
        num = torch.matmul(Q_cos.unsqueeze(-2), KcosV_cum) + \
              torch.matmul(Q_sin.unsqueeze(-2), KsinV_cum)
        num = num.squeeze(-2)          # (B, H, N, d)

        # Denominator (causal version)
        Kcos_cum = K_cos.cumsum(dim=2)   # (B, H, N, d)
        Ksin_cum = K_sin.cumsum(dim=2)
        denom = (Q_cos * Kcos_cum).sum(-1, keepdim=True) + \
                (Q_sin * Ksin_cum).sum(-1, keepdim=True)

        return num / denom.clamp(min=1e-6)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        B, N, _ = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        def split_heads(t):
            return t.view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Apply ReLU + cosine reweighting
        Q_cos, Q_sin, K_cos, K_sin = self._cos_reweight(Q, K, N)

        # Choose causal or bidirectional
        if self.causal:
            out = self._causal_linear_attention(Q_cos, Q_sin, K_cos, K_sin, V)
        else:
            out = self._linear_attention(Q_cos, Q_sin, K_cos, K_sin, V)

        # Merge heads
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_model)
        out = self.W_O(out)
        out = self.drop(out)

        return out


if __name__ == "__main__":
    B, N, d_model, n_heads = 2, 64, 128, 4
    attn = CosFormerAttention(d_model=d_model, n_heads=n_heads, causal=True)
    x = torch.randn(B, N, d_model)
    out = attn(x)
    print(f"CosFormerAttention: input {x.shape} → output {out.shape}")
    assert out.shape == (B, N, d_model)
    print("  PASSED ✓  (no softmax used)")
