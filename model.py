"""model.py — the transformer itself.

This is the heart of the project. Built smallest-to-largest:

    Head / MultiHeadAttention  <- Phase 2 (this file), THE core idea
    FeedForward                <- Phase 3
    Block (attention + FFN, with residuals + LayerNorm)  <- Phase 3
    GPT (embeddings -> N blocks -> output head + loss)   <- Phase 3

The one rule: attention is written by hand, not imported from
nn.MultiheadAttention. Understanding it is the whole point.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """A single head of self-attention.

    Given a batch of sequences x of shape (B, T, n_embd), each position:
      1. produces a query, key, value vector (learned linear projections),
      2. scores every position against every other via q . k  (scaled),
      3. masks out the future (causal), softmaxes into weights,
      4. returns a weighted average of the values.
    Output shape: (B, T, head_size).
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        # No bias: LayerNorm/embeddings already handle offsets, and Q/K/V are
        # conventionally bias-free. These are the only learned parts of a head.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # A lower-triangular matrix of 1s. tril[i, j] == 1 iff j <= i, i.e.
        # position i is allowed to attend to position j. It's a CONSTANT (not
        # learned), so we register it as a buffer: it moves with .to(device)
        # and is saved with the model, but the optimizer never touches it.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def _weights(self, x: torch.Tensor) -> torch.Tensor:
        """The attention pattern itself: (B, T, T), each row summing to 1.

        Factored out of forward() so the sanity check can inspect the exact
        weights the model uses — no duplicated logic to drift out of sync.
        """
        B, T, C = x.shape
        k = self.key(x)                                  # (B, T, head_size)
        q = self.query(x)                                # (B, T, head_size)

        # Match every query against every key: (B,T,hs) @ (B,hs,T) -> (B,T,T).
        # Scale by 1/sqrt(head_size) so the numbers going into softmax stay
        # tame (large dot products make softmax peaky -> vanishing gradients).
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)

        # Causal mask: a position may not attend to the future. Set those
        # scores to -inf so softmax gives them exactly 0 weight.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        return F.softmax(scores, dim=-1)                 # (B, T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wei = self._weights(x)                           # (B, T, T)
        v = self.value(x)                                # (B, T, head_size)
        return wei @ v                                   # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated then projected.

    Each head is free to specialize (one tracks brackets, another agreement).
    We split the n_embd-wide vector into n_head chunks of size n_embd//n_head,
    attend within each, glue the outputs back to width n_embd, then apply a
    final linear projection to let the heads' outputs mix.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate along the feature dim: n_head * head_size == n_embd.
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.proj(out)                                 # (B, T, n_embd)


# TODO(Phase 3): FeedForward, Block, GPT


if __name__ == "__main__":
    # ---- Phase 2 sanity checks (see the guide's Phase 2 checklist) ----
    torch.manual_seed(0)
    B, T, C, n_head = 4, 8, 32, 4
    x = torch.randn(B, T, C)

    # 1. Shape: (B, T, C) in -> (B, T, C) out.
    mha = MultiHeadAttention(n_embd=C, n_head=n_head, block_size=T)
    out = mha(x)
    assert out.shape == (B, T, C), out.shape
    print(f"shape check      : in {tuple(x.shape)} -> out {tuple(out.shape)}   OK")

    # 2. Attention weights: every row sums to 1, future positions get 0 weight.
    head = Head(n_embd=C, head_size=C // n_head, block_size=T)
    w = head._weights(x)                                  # (B, T, T)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "rows must sum to 1"
    print("softmax check    : every attention row sums to 1   OK")

    future = torch.triu(torch.ones(T, T), diagonal=1).bool()  # strictly upper
    assert (w[0][future] == 0).all(), "a position attended to the future!"
    print("causal check     : weight on future positions is exactly 0   OK")

    print("\n  position 0 weights (can only see itself):")
    print("   ", w[0, 0].round(decimals=2).tolist())
    print("  position 4 weights (sees 0..4, then zeros):")
    print("   ", w[0, 4].round(decimals=2).tolist())

    # 3. Behavioral causality: scrambling FUTURE inputs must not change PAST
    #    outputs. This is a stronger guarantee than just reading the mask.
    t = 3
    x2 = x.clone()
    x2[:, t + 1 :] = torch.randn(B, T - (t + 1), C)      # scramble the future
    out2 = mha(x2)
    assert torch.allclose(out[:, : t + 1], out2[:, : t + 1], atol=1e-6), \
        "past outputs changed when the future changed — the mask is leaking!"
    print(f"\nno-peeking check : scrambling inputs after position {t} left "
          f"outputs 0..{t} unchanged   OK")
