"""sample.py — using the trained model to generate text.

Generation is the payoff. Given a seed string, we repeatedly:

    forward the context -> take the LAST position's logits -> softmax to a
    probability over the next character -> sample one char -> append it -> repeat

Crop the context to block_size so it never exceeds what the model was trained
on. Watching gibberish slowly turn into text-that-looks-like-your-corpus is the
moment the whole project pays off.

Phase 5 fills this in. For now it is a scaffold describing the plan.
"""

# TODO(Phase 5): load checkpoint, generate(seed, n_tokens)
