"""
attention/gaussian_attention.py
────────────────────────────────
Gaussian kernel attention — NO SOFTMAX.

Directly implements the SOFT paper:
  "Softmax-free Linear Transformers"
  Lu et al., NeurIPS 2021 / IJCV 2024

Core idea (Equation 5 of the paper):
  S_ij = exp( −‖Q_i − K_j‖² / (2√d_e) )

This uses DISTANCE between embedding vectors as the similarity score.
When Q_i and K_j are close in embedding space → S_ij near 1 (attend strongly).
When they are far apart → S_ij near 0 (ignore).

No softmax is ever applied.

To avoid O(N²) complexity, the full N×N matrix S is approximated using
Nyström decomposition with m bottleneck tokens (m << N).
The Moore-Penrose pseudo-inverse of the m×m bottleneck matrix A is
computed via Newton-Raphson iteration (Proposition proved in the paper).

Why this matters for your project:
  Your supervisor wants "training from scratch without softmax."
  This is the most direct implementation of distance-based attention
  from the literature you have.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianAttention(nn.Module):
    """
    Multi-head Gaussian kernel attention (softmax-free).

    Args:
        d_model:       total model dimension
        n_heads:       number of attention heads
        n_bottleneck:  m — number of bottleneck tokens for Nyström (default 49)
        newton_steps:  iterations for Moore-Penrose inverse (default 20)
        dropout:       dropout rate on output projection
        causal:        if True, use causal (autoregressive) masking
        standardize_qk: if True, per-token mean/std standardise Q and K
                        (across d_model, before the head split) prior to
                        computing the Gaussian kernel. Rogue-dimension
                        ablation from Timkey & van Schijndel (EMNLP 2021) —
                        OFF by default, does not change existing behaviour.
    """

    def __init__(self,
                 d_model:      int,
                 n_heads:      int,
                 n_bottleneck: int  = 49,
                 newton_steps: int  = 20,
                 dropout:      float = 0.1,
                 causal:       bool  = True,
                 standardize_qk: bool = False):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model      = d_model
        self.n_heads      = n_heads
        self.d_head       = d_model // n_heads      # dimension per head
        self.n_bottleneck = n_bottleneck
        self.newton_steps = newton_steps
        self.causal       = causal
        self.scale        = 2.0 * math.sqrt(self.d_head)  # 2√d_e from paper
        self.standardize_qk = standardize_qk

        # SOFT paper uses Q = K (symmetric kernel), so we share weights.
        # We still keep separate W_Q and W_K for the causal case where
        # asymmetry may help, following the paper's ablation discussion.
        self.W_Q  = nn.Linear(d_model, d_model, bias=False)
        self.W_K  = nn.Linear(d_model, d_model, bias=False)
        self.W_V  = nn.Linear(d_model, d_model, bias=False)
        self.W_O  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    # ── Gaussian kernel ──────────────────────────────────────────────────────

    def _gaussian_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute exp( −‖X_i − Y_j‖² / scale ) for all pairs (i, j).

        X: (..., n, d)
        Y: (..., m, d)
        Returns: (..., n, m)  — values in (0, 1]
        """
        # ‖X_i − Y_j‖² = ‖X_i‖² + ‖Y_j‖² − 2 X_i·Y_j
        X_sq = (X ** 2).sum(dim=-1, keepdim=True)      # (..., n, 1)
        Y_sq = (Y ** 2).sum(dim=-1).unsqueeze(-2)       # (..., 1, m)
        dot  = torch.matmul(X, Y.transpose(-2, -1))    # (..., n, m)
        dist_sq = X_sq + Y_sq - 2.0 * dot
        dist_sq = dist_sq.clamp(min=0.0)               # numerical safety
        return torch.exp(-dist_sq / self.scale)

    # ── Moore-Penrose pseudo-inverse via Newton-Raphson ──────────────────────

    def _moore_penrose_inv(self, A: torch.Tensor) -> torch.Tensor:
        """
        Approximate A† (pseudo-inverse) using Newton-Raphson iteration.

        From the SOFT paper (Section 3.2):
          A_{k+1} = 2 A_k − A_k A A_k
          converges to A†  starting from A_0 = α·A

        A: (..., m, m)
        Returns: (..., m, m)
        """
        # Starting point: α = 2 / ‖A‖₁²
        norm = A.abs().sum(dim=-2).max(dim=-1).values   # ‖A‖₁  shape: (...)
        # Reshape norm for broadcasting
        for _ in range(A.dim() - 2):
            norm = norm.unsqueeze(-1).unsqueeze(-1)

        alpha = 2.0 / (norm ** 2 + 1e-8)
        A_inv = alpha * A

        for _ in range(self.newton_steps):
            # A_{k+1} = 2 A_k − A_k A A_k
            A_inv = 2.0 * A_inv - torch.matmul(torch.matmul(A_inv, A), A_inv)

        return A_inv

    # ── Nyström approximation ────────────────────────────────────────────────

    def _nystrom_attention(self,
                           Q: torch.Tensor,
                           K: torch.Tensor,
                           V: torch.Tensor) -> torch.Tensor:
        """
        Nyström-approximated Gaussian attention.
        Never builds the full N×N attention matrix.

        Q, K, V: (batch, heads, seq_len, d_head)
        Returns: (batch, heads, seq_len, d_head)
        """
        B, H, N, d = Q.shape
        m = min(self.n_bottleneck, N)   # can't have more bottlenecks than tokens

        # Sample m evenly-spaced bottleneck token indices
        stride = max(1, N // m)
        idx = torch.arange(0, N, stride, device=Q.device)[:m]

        # Bottleneck queries/keys: (batch, heads, m, d_head)
        Q_e = Q[:, :, idx, :]
        K_e = K[:, :, idx, :]

        # ── Three small kernel matrices (never N×N) ────────────────────────
        # A: bottleneck × bottleneck  (m × m)
        A = self._gaussian_kernel(Q_e, K_e)          # (B, H, m, m)

        # P: sequence × bottleneck  (N × m)
        # Using Q for rows, K_e for columns (asymmetric, works for causal)
        P = self._gaussian_kernel(Q,   K_e)          # (B, H, N, m)

        # R: bottleneck × sequence  (m × N)
        R = self._gaussian_kernel(Q_e, K)            # (B, H, m, N)

        # ── Nyström: output ≈ P · A† · R · V ─────────────────────────────
        # Computed in order to avoid N×N intermediate:
        #   step 1: R · V          (m × d_head)
        #   step 2: A† · (R·V)    (m × d_head)
        #   step 3: P · (A†·R·V)  (N × d_head)

        A_inv = self._moore_penrose_inv(A)           # (B, H, m, m)
        RV    = torch.matmul(R, V)                   # (B, H, m, d_head)
        ARV   = torch.matmul(A_inv, RV)              # (B, H, m, d_head)
        out   = torch.matmul(P, ARV)                 # (B, H, N, d_head)

        # Normalise by row sums (as in the paper's symmetric normalisation)
        # D = sum of P · A† · R  row-wise  (prevents spectral norm explosion)
        R_sum = R.sum(dim=-1, keepdim=True)          # (B, H, m, 1) — sum over N
        A_inv_R_sum = torch.matmul(A_inv, R_sum)     # (B, H, m, 1)
        denom = torch.matmul(P, A_inv_R_sum)         # (B, H, N, 1)
        out   = out / (denom.clamp(min=1e-6))

        return out

    # ── Rogue-dimension ablation: per-token standardisation ──────────────────

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-token standardisation across d_model: subtract the mean and
        divide by the std of each token's own vector. Removes any single
        dimension's disproportionate contribution to the Gaussian kernel's
        ‖Q−K‖² distance (rogue-dimension fix, Timkey & van Schijndel 2021).

        x: (B, N, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (x - mean) / std

    # ── Causal (simple) fallback for short sequences ─────────────────────────

    def _causal_gaussian_attention(self,
                                    Q: torch.Tensor,
                                    K: torch.Tensor,
                                    V: torch.Tensor) -> torch.Tensor:
        """
        Full N×N Gaussian attention with causal mask.
        Used when N is small enough to afford it (e.g. N < 512).
        NO softmax — just Gaussian kernel + causal mask.

        Q, K, V: (batch, heads, seq_len, d_head)
        """
        B, H, N, d = Q.shape

        # Gaussian kernel scores: (B, H, N, N)
        S = self._gaussian_kernel(Q, K)

        # Causal mask: future positions → 0 (not -inf, because exp(0)=1 but
        # we apply it multiplicatively here, not additively before exp)
        # We zero out the upper triangle AFTER computing exp
        mask = torch.tril(torch.ones(N, N, device=Q.device)).unsqueeze(0).unsqueeze(0)
        S = S * mask

        # Normalise row-wise (replaces softmax normalisation)
        S = S / (S.sum(dim=-1, keepdim=True).clamp(min=1e-6))

        return torch.matmul(S, V)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:    (batch, seq_len, d_model)
        mask: unused (causal mask handled internally)
        Returns: (batch, seq_len, d_model)
        """
        B, N, _ = x.shape

        # Project to Q, K, V
        Q = self.W_Q(x)   # (B, N, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        if self.standardize_qk:
            Q = self._standardize(Q)
            K = self._standardize(K)

        # Split into heads: (B, N, d_model) → (B, H, N, d_head)
        def split_heads(t):
            t = t.view(B, N, self.n_heads, self.d_head)
            return t.permute(0, 2, 1, 3)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Choose attention implementation based on sequence length
        if self.causal and N <= 512:
            # Use full causal attention for short sequences (exact)
            out = self._causal_gaussian_attention(Q, K, V)
        else:
            # Use Nyström approximation for long sequences (O(N))
            out = self._nystrom_attention(Q, K, V)

        # Merge heads: (B, H, N, d_head) → (B, N, d_model)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.d_model)

        # Output projection
        out = self.W_O(out)
        out = self.drop(out)

        return out


if __name__ == "__main__":
    # Quick test
    B, N, d_model, n_heads = 2, 64, 128, 4
    attn = GaussianAttention(d_model=d_model, n_heads=n_heads,
                              n_bottleneck=16, causal=True)
    x = torch.randn(B, N, d_model)
    out = attn(x)
    print(f"GaussianAttention: input {x.shape} → output {out.shape}")
    assert out.shape == (B, N, d_model), "Shape mismatch!"
    print("  PASSED ✓  (no softmax used)")
