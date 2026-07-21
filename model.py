"""model.py — the transformer itself.

This is the heart of the project. It will hold, from smallest to largest:

    Head / MultiHeadAttention  <- Phase 2, THE piece the project exists to teach
    FeedForward                <- Phase 3
    Block (attention + FFN, with residuals + LayerNorm)  <- Phase 3
    GPT (embeddings -> N blocks -> output head + loss)   <- Phase 3

The one rule of this project: the attention mechanism is written by hand, not
imported from nn.MultiheadAttention. Understanding it is the whole point.

Phases 2-3 fill this in. For now it is a scaffold describing the plan.
"""

# TODO(Phase 2): Head, MultiHeadAttention (scaled dot-product + causal mask)
# TODO(Phase 3): FeedForward, Block, GPT
