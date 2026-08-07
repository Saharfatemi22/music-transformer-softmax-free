from attention.gaussian_attention   import GaussianAttention
from attention.cosine_attention     import CosineAttention
from attention.cosformer_attention  import CosFormerAttention


def build_attention(attention_type: str, d_model: int, n_heads: int,
                    dropout: float = 0.1, causal: bool = True, **kwargs):
    """
    Factory function — returns the correct attention module.
    Pass attention_type = "gaussian" | "cosine" | "cosformer"
    """
    if attention_type == "gaussian":
        return GaussianAttention(
            d_model=d_model, n_heads=n_heads,
            n_bottleneck=kwargs.get("n_bottleneck", 49),
            newton_steps=kwargs.get("newton_steps", 20),
            dropout=dropout, causal=causal,
            standardize_qk=kwargs.get("standardize_qk", False)
        )
    elif attention_type == "cosine":
        return CosineAttention(
            d_model=d_model, n_heads=n_heads,
            m_init=kwargs.get("cosine_m_init", 0.5),
            dropout=dropout, causal=causal,
            standardize_qk=kwargs.get("standardize_qk", False)
        )
    elif attention_type == "cosformer":
        return CosFormerAttention(
            d_model=d_model, n_heads=n_heads,
            dropout=dropout, causal=causal
        )
    else:
        raise ValueError(
            f"Unknown attention_type: '{attention_type}'. "
            f"Choose from: 'gaussian', 'cosine', 'cosformer'"
        )
