"""data.py — the model's food supply.

Turns a raw text file into batches of (input, target) tensors the model can
train on. For a character-level language model the whole pipeline is:

    raw text  ->  char<->id lookup tables  ->  one long tensor of ids
              ->  get_batch(): random chunks of (x, y) where y is x shifted
                  left by one (the "correct next character" at each position)

Phase 1 fills this in. For now it is a scaffold describing the plan.
"""

# TODO(Phase 1): load text, build stoi/itos, encode(), decode(), get_batch()
