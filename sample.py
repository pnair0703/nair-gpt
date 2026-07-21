"""sample.py — using the trained model to generate text.

Usage:  python sample.py [checkpoint.pt]     # defaults to ckpt.pt

Generation is the "predict the next character" game run in a loop, feeding
each sampled character back in as new context.
"""

import sys

import torch

from model import GPT  # noqa: F401  (GPTConfig is unpickled from the checkpoint)


@torch.no_grad()
def generate(model, stoi, itos, seed="\n", n_tokens=500, temperature=1.0, device="cpu"):
    """Autoregressively sample n_tokens characters, starting from `seed`."""
    model.eval()
    block_size = model.config.block_size

    ids = [stoi[c] for c in seed] if seed else [stoi["\n"]]
    context = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T)
    out = list(ids)

    for _ in range(n_tokens):
        # Never feed more than block_size positions (positional embedding limit).
        ctx = context[:, -block_size:]
        logits, _ = model(ctx)                 # (1, T, vocab_size)
        logits = logits[:, -1, :] / temperature  # only the LAST position: (1, vocab)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # sample: (1, 1)
        context = torch.cat([context, next_id], dim=1)     # append, then loop
        out.append(next_id.item())

    return "".join(itos[i] for i in out)


def load_model(path, device):
    # weights_only=False because the checkpoint also stores our GPTConfig object
    # (which we created, so it is safe to unpickle).
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["stoi"], ckpt["itos"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "ckpt.pt"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, stoi, itos = load_model(path, device)
    text = generate(model, stoi, itos, seed="\n", n_tokens=500, temperature=0.8, device=device)
    print(text)


if __name__ == "__main__":
    main()
