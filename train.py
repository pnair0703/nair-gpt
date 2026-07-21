"""train.py — the loop that makes the model less wrong.

The training loop is plumbing, but it's the plumbing that turns a randomly
initialized network into a language model:

    for each step:
        x, y = get_batch('train')     # fetch a batch
        logits, loss = model(x, y)    # forward pass + cross-entropy loss
        loss.backward()               # backprop: how should each knob change?
        optimizer.step()              # nudge every knob a little
        optimizer.zero_grad()         # reset for next step

We print train/val loss periodically to watch it learn. The rule of thumb:
start tiny and prove the wiring overfits a small slice before scaling up.

Phase 4 fills this in. For now it is a scaffold describing the plan.
"""

# TODO(Phase 4): config, model init, Adam optimizer, training loop, eval
