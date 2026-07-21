"""model.py — the transformer itself.

This is the heart of the project. Built smallest-to-largest:

    Head / MultiHeadAttention  <- Phase 2, THE core idea
    FeedForward                <- Phase 3
    Block (attention + FFN, with residuals + LayerNorm)  <- Phase 3
    GPT (embeddings -> N blocks -> output head + loss)   <- Phase 3

The one rule: attention is written by hand, not imported from
nn.MultiheadAttention. Understanding it is the whole point.
"""

import math
from dataclasses import dataclass

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


class FeedForward(nn.Module):
    """Per-position MLP: expand 4x, GELU nonlinearity, shrink back.

    Runs on each position independently (no mixing across positions — that's
    attention's job). The 4x hidden width is the standard transformer ratio.
    """

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention then feed-forward, each wrapped in a
    residual connection with pre-LayerNorm.

        x = x + attn(norm(x))   # gather context across positions
        x = x + ffn(norm(x))    # think about it, per position

    The residual `x +` gives gradients a clean path and lets each sublayer
    learn only a correction. Pre-norm (normalize the INPUT to each sublayer)
    is the modern, more-stable convention.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


@dataclass
class GPTConfig:
    """All the knobs that set the model's size, in one place."""

    vocab_size: int          # number of distinct characters (set from data)
    block_size: int = 64     # context length: how far back attention can see
    n_embd: int = 64         # embedding / residual-stream width
    n_head: int = 4          # attention heads per block
    n_layer: int = 4         # number of transformer blocks stacked


class GPT(nn.Module):
    """The full decoder-only transformer.

        ids -> token embedding + positional embedding
            -> n_layer transformer blocks
            -> final LayerNorm -> linear head -> logits over the vocabulary
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # "what character am I" and "where in the sequence am I", both learned.
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(config.n_embd, config.n_head, config.block_size)
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)                       # final norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)    # -> logits
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # Small normal init (GPT-2 convention, std=0.02) keeps early training
        # stable — better than PyTorch's defaults for this architecture.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """idx: (B, T) ids. Returns (logits, loss). loss is None if no targets."""
        B, T = idx.shape
        tok = self.token_embedding(idx)                              # (B, T, n_embd)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok + pos                                                # broadcast add
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                     # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            # cross_entropy wants (N, C) logits and (N,) targets, so flatten.
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # ---- Phase 2 sanity checks: the attention mechanism ----
    torch.manual_seed(0)
    B, T, C, n_head = 4, 8, 32, 4
    x = torch.randn(B, T, C)

    mha = MultiHeadAttention(n_embd=C, n_head=n_head, block_size=T)
    out = mha(x)
    assert out.shape == (B, T, C), out.shape
    print(f"shape check      : in {tuple(x.shape)} -> out {tuple(out.shape)}   OK")

    head = Head(n_embd=C, head_size=C // n_head, block_size=T)
    w = head._weights(x)                                  # (B, T, T)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "rows must sum to 1"
    print("softmax check    : every attention row sums to 1   OK")

    future = torch.triu(torch.ones(T, T), diagonal=1).bool()
    assert (w[0][future] == 0).all(), "a position attended to the future!"
    print("causal check     : weight on future positions is exactly 0   OK")

    t = 3
    x2 = x.clone()
    x2[:, t + 1 :] = torch.randn(B, T - (t + 1), C)
    out2 = mha(x2)
    assert torch.allclose(out[:, : t + 1], out2[:, : t + 1], atol=1e-6), \
        "past outputs changed when the future changed — the mask is leaking!"
    print(f"no-peeking check : scrambling inputs after position {t} left "
          f"outputs 0..{t} unchanged   OK")

    # ---- Phase 3 sanity check: the full model forward pass ----
    import data

    ds = data.load()
    cfg = GPTConfig(vocab_size=ds.vocab_size, block_size=16, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)
    xb, yb = data.get_batch(ds.train_data, cfg.block_size, batch_size=4)
    logits, loss = model(xb, yb)

    print(f"\nparams           : {model.num_params():,}")
    print(f"logits shape     : {tuple(logits.shape)}   (B, T, vocab_size)")
    print(f"init loss        : {loss.item():.3f}   "
          f"(random-guess baseline ln({ds.vocab_size}) = {math.log(ds.vocab_size):.3f})")
