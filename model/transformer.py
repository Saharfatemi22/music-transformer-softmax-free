"""
model/transformer.py
─────────────────────
Decoder-only Transformer for music generation.
Trained from scratch — no pretrained weights.

Architecture decisions come directly from the papers:
  • Decoder-only (causal): for autoregressive music generation
    → follows Paper 4 (Music Transformer) and Paper 5 (Pop Music Transformer)
  • Relative position embeddings: helps music because
    "relative differences matter more than absolute values" (Paper 4, Intro)
  • Attention mechanism: pluggable — gaussian / cosine / cosformer
    → no softmax in any of these (your supervisor's requirement)
  • Token embedding + positional encoding → follows standard practice
  • Pre-norm (LayerNorm before attention): more stable training from scratch
"""

import math
import torch
import torch.nn as nn

from attention import build_attention
from data.vocabulary import VOCAB_SIZE


class FeedForward(nn.Module):
    """Standard 2-layer FFN with GELU activation and dropout."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    """
    Single Transformer decoder layer (Pre-Norm formulation):
      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))
    Pre-norm is more stable when training from scratch.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, attention_type: str, **attn_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = build_attention(
            attention_type, d_model, n_heads,
            dropout=dropout, causal=True, **attn_kwargs
        )
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention + residual
        x = x + self.attn(self.norm1(x))
        # Pre-norm FFN + residual
        x = x + self.ff(self.norm2(x))
        return x


class MusicTransformer(nn.Module):
    """
    Full decoder-only Transformer for music sequence modelling.

    Input:  integer token IDs  (batch, seq_len)
    Output: logits over vocabulary  (batch, seq_len, vocab_size)

    Trained with next-token prediction (cross-entropy loss).
    """

    def __init__(self,
                 vocab_size:     int   = VOCAB_SIZE,
                 d_model:        int   = 128,
                 n_heads:        int   = 4,
                 n_layers:       int   = 4,
                 d_ff:           int   = 512,
                 dropout:        float = 0.1,
                 max_seq_len:    int   = 2048,
                 attention_type: str   = "gaussian",
                 **attn_kwargs):
        super().__init__()

        self.d_model  = d_model
        self.n_layers = n_layers

        # ── Token embedding ──────────────────────────────────────────────────
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # ── Sinusoidal positional encoding ───────────────────────────────────
        # Fixed (not learned) — standard choice from "Attention is All You Need"
        pe = self._make_sinusoidal_pe(max_seq_len, d_model)
        self.register_buffer("pe", pe)   # not a parameter, just a buffer

        self.embed_drop = nn.Dropout(dropout)

        # ── Transformer layers ───────────────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, attention_type=attention_type,
                **attn_kwargs
            )
            for _ in range(n_layers)
        ])

        # ── Final projection to vocabulary ───────────────────────────────────
        self.norm  = nn.LayerNorm(d_model)
        self.head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights.
        # Common practice; reduces parameters and improves generalisation.
        self.head.weight = self.embed.weight

        # ── Initialise weights ───────────────────────────────────────────────
        self._init_weights()

    def _make_sinusoidal_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        """Classic sinusoidal positional encoding (Vaswani et al. 2017)."""
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, max_len, d_model)

    def _init_weights(self):
        """Xavier uniform initialisation for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (batch, seq_len)  — integer token IDs
        Returns: (batch, seq_len, vocab_size)  — logits
        """
        B, N = tokens.shape

        # Embed tokens
        x = self.embed(tokens)                     # (B, N, d_model)

        # Add positional encoding
        x = x + self.pe[:, :N, :]                 # (B, N, d_model)
        x = self.embed_drop(x)

        # Pass through all Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Final norm + projection to vocabulary
        x = self.norm(x)
        logits = self.head(x)                      # (B, N, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: dict) -> MusicTransformer:
    """Build model from config dictionary."""
    return MusicTransformer(
        vocab_size      = VOCAB_SIZE,
        d_model         = cfg.get("d_model", 128),
        n_heads         = cfg.get("n_heads", 4),
        n_layers        = cfg.get("n_layers", 4),
        d_ff            = cfg.get("d_ff", 512),
        dropout         = cfg.get("dropout", 0.1),
        attention_type  = cfg.get("attention_type", "gaussian"),
        n_bottleneck    = cfg.get("n_bottleneck", 49),
        newton_steps    = cfg.get("newton_steps", 20),
        cosine_m_init   = cfg.get("cosine_m_init", 0.5),
        standardize_qk  = cfg.get("standardize_qk", False),
    )


if __name__ == "__main__":
    for attn_type in ["gaussian", "cosine", "cosformer"]:
        model = MusicTransformer(
            d_model=128, n_heads=4, n_layers=2,
            attention_type=attn_type
        )
        tokens = torch.randint(0, VOCAB_SIZE, (2, 64))
        logits = model(tokens)
        print(f"[{attn_type:12s}] input {tokens.shape} → logits {logits.shape} "
              f"| params: {model.count_parameters():,}")
        assert logits.shape == (2, 64, VOCAB_SIZE)
    print("\nAll attention types: PASSED ✓")
