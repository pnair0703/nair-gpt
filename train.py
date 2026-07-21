"""train.py — the loop that makes the model less wrong.

Usage:  python train.py [preset]      # preset defaults to "medium"

The loop itself is five lines (fetch batch -> forward+loss -> zero grads ->
backward -> step). Everything else is measurement and bookkeeping: we print
train/val loss periodically and save checkpoints (including milestone
snapshots so we can later show the gibberish -> text progression).
"""

import json
import os
import sys
import time

import torch

import data
from model import GPT, GPTConfig

# Model + training knobs, grouped as presets. Start small to prove the wiring
# learns, then scale up for better text — exactly the guide's advice.
PRESETS = {
    "tiny":   dict(n_layer=2, n_head=2, n_embd=64,  block_size=32,  batch_size=16, max_iters=1000, lr=1e-3, dropout=0.0),
    "small":  dict(n_layer=4, n_head=4, n_embd=128, block_size=64,  batch_size=32, max_iters=3000, lr=1e-3, dropout=0.1),
    "medium": dict(n_layer=6, n_head=6, n_embd=192, block_size=128, batch_size=64, max_iters=5000, lr=3e-4, dropout=0.1),
    "big":    dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, max_iters=3600, lr=3e-4, dropout=0.2),
}

EVAL_INTERVAL = 250   # how often to measure train/val loss
EVAL_ITERS = 50       # batches averaged per loss estimate
SEED = 1337


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, ds, block_size, batch_size, device):
    """Average loss over several batches of train AND val (no grad, eval mode).

    Averaging smooths out the noise of any single random batch, and the
    train-vs-val gap is our overfitting gauge.
    """
    model.eval()
    out = {}
    for split, d in (("train", ds.train_data), ("val", ds.val_data)):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = data.get_batch(d, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_ckpt(path, model, cfg, ds, history, preset):
    """Save everything sample.py needs to regenerate text without the corpus."""
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "stoi": ds.stoi,
            "itos": ds.itos,
            "history": history,
            "preset": preset,
        },
        path,
    )


def main():
    preset_name = sys.argv[1] if len(sys.argv) > 1 else "medium"
    p = PRESETS[preset_name]
    device = os.environ.get("DEVICE") or pick_device()
    torch.manual_seed(SEED)

    ds = data.load()
    cfg = GPTConfig(
        vocab_size=ds.vocab_size,
        block_size=p["block_size"],
        n_embd=p["n_embd"],
        n_head=p["n_head"],
        n_layer=p["n_layer"],
        dropout=p["dropout"],
    )
    model = GPT(cfg).to(device)
    print(f"preset={preset_name}  device={device}  params={model.num_params():,}  dropout={p['dropout']}")

    # weight_decay adds a second anti-overfitting pressure alongside dropout.
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=0.1)

    # Milestone snapshots for the README's gibberish -> text story.
    milestones = sorted({0, p["max_iters"] // 4, p["max_iters"] // 2, p["max_iters"]})

    best_val = float("inf")   # early stopping: track the best val loss...
    since_improved = 0        # ...and how many evals since it last improved.
    patience = 6              # stop if val doesn't improve for this many evals

    history = []
    t0 = time.time()
    for it in range(p["max_iters"] + 1):
        if it % EVAL_INTERVAL == 0 or it == p["max_iters"]:
            losses = estimate_loss(model, ds, p["block_size"], p["batch_size"], device)
            history.append((it, losses["train"], losses["val"]))
            flag = ""
            if losses["val"] < best_val:
                # New best-GENERALIZING model — this is what ckpt.pt keeps,
                # not the final (possibly overfit) one.
                best_val = losses["val"]
                since_improved = 0
                save_ckpt("ckpt.pt", model, cfg, ds, history, preset_name)
                flag = "  <- best (saved)"
            else:
                since_improved += 1
            print(
                f"step {it:5d} | train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | {time.time() - t0:.0f}s{flag}"
            )
            if since_improved >= patience:
                print(f"early stop: val hasn't improved in {patience} evals (best {best_val:.4f})")
                break

        if it in milestones:
            os.makedirs("checkpoints", exist_ok=True)
            save_ckpt(f"checkpoints/ckpt_{it}.pt", model, cfg, ds, history, preset_name)

        if it == p["max_iters"]:
            break

        # ---- the five lines that do all the learning ----
        x, y = data.get_batch(ds.train_data, p["block_size"], p["batch_size"], device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the full loss history (every eval) for the README loss curve.
    with open("history.json", "w") as f:
        json.dump(history, f)
    print(f"done in {time.time() - t0:.0f}s — best val {best_val:.4f}, saved ckpt.pt + history.json")


if __name__ == "__main__":
    main()
