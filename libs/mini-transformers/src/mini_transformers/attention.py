"""Mini transformers attention layers."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot product for attention mechanism.

    Works for batched tensors, self-attention, and cross-attention.

    Self-attention: Q, K, V have shape (batch_size, seq_len, d_k or d_v).
    Cross-attention: Q has shape (batch_size, seq_len_q, d_k),
                     K, V have shape (batch_size, seq_len_k, d_k or d_v).

    Args:
        Q: Queries tensor of shape (..., seq_len, d_k).
        K: Keys tensor of shape (..., seq_len, d_k).
        V: Values tensor of shape (..., seq_len, d_v).
        mask: Optional mask tensor broadcastable to (..., seq_len, seq_len_k).

    Returns:
        out: Output tensor of shape (..., seq_len, d_v).
        attn: Attention weights of shape (..., seq_len, seq_len_k).
    """
    d_k: float = Q.shape[-1]

    # Works for batched tensors
    scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
        d_k
    )  # T x T
    if mask:
        mask = mask.to(dtype=torch.bool)
        scores = scores.masked_fill(mask, 1e-15)
    attn = F.softmax(scores, dim=-1)  # T x T
    out = attn @ V  # T x d_v
    return out, attn


class MultiheadAttention(nn.Module):
    """Multi-head attention layer.

    Args:
        input_dim: Dimension of input features.
        embed_dim: Dimension of embedding space.
        num_heads: Number of attention heads.

    Returns:
        out: Output tensor of shape (batch_size, seq_len, input_dim).
        attention: Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
    """

    def __init__(self, input_dim: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.qkv_projection = nn.Linear(input_dim, 3 * embed_dim)
        self.out_projection = nn.Linear(embed_dim, input_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for multi-head attention."""
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv_projection(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, -1)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        scores, attention = scaled_dot_product(q, k, v, mask)
        scores = scores.permute(0, 2, 1, 3)
        scores = scores.reshape(batch_size, seq_len, self.embed_dim)

        out = self.out_projection(scores)
        return out, attention


if __name__ == "__main__":
    """Minimal test code."""

    # Test scaled dot product attention
    seq_len, d_k, d_v = 10, 2, 4
    batch_size = 32
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_v)
    scores, attention = scaled_dot_product(q, k, v)

    # Test multi-head attention
    multihead = MultiheadAttention(input_dim=10, embed_dim=16, num_heads=4)
    x = torch.randn(batch_size, seq_len, 10)
    out, attention = multihead(x)
