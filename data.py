"""data.py — the model's food supply.

Turns a raw text file into batches of (input, target) tensors for training a
character-level language model. Pipeline:

    raw text  ->  char<->id lookup (the "tokenizer")
              ->  one long tensor of ids
              ->  train/val split
              ->  get_batch(): random chunks of (x, y) where y is x shifted
                  left by one — the "correct next character" at every position.
"""

from dataclasses import dataclass

import torch


@dataclass
class CharData:
    """Everything downstream code needs: the tokenizer + the split tensors."""

    chars: list[str]          # the sorted vocabulary, index == id
    stoi: dict[str, int]      # string -> id  ("h" -> 7)
    itos: dict[int, str]      # id -> string  (7 -> "h")
    train_data: torch.Tensor  # 1-D LongTensor of ids (first ~90%)
    val_data: torch.Tensor    # 1-D LongTensor of ids (last ~10%)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        """"hi" -> [7, 12]"""
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        """[7, 12] -> "hi" (accepts ints or a 1-D tensor)"""
        return "".join(self.itos[int(i)] for i in ids)


def load(path: str = "input.txt", train_frac: float = 0.9) -> CharData:
    """Read the corpus, build the tokenizer, encode everything, split."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    # The tokenizer for a char-level model is just: sort the unique characters
    # and let each character's position be its integer id.
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encode the ENTIRE corpus into one long 1-D tensor of ids.
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # Hold out the tail as validation so we can tell learning from memorizing.
    n = int(train_frac * len(data))
    return CharData(chars, stoi, itos, data[:n], data[n:])


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grab `batch_size` random chunks of length `block_size`.

    Returns x, y each of shape (batch_size, block_size), where y is x shifted
    left by one character — so y[b, t] is the target ("next char") for x[b, t].
    """
    # Random starting offsets; -block_size so a chunk never runs off the end.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    # Sanity check the whole pipeline.
    ds = load()
    print(f"corpus chars : {len(ds.train_data) + len(ds.val_data):,}")
    print(f"vocab_size   : {ds.vocab_size}")
    print(f"vocabulary   : {''.join(ds.chars)!r}")

    # Round-trip check: encode then decode should be the identity.
    sample = "Hello, world!"
    assert ds.decode(ds.encode(sample)) == sample, "encode/decode mismatch!"
    print(f"roundtrip    : {sample!r} -> {ds.encode(sample)} -> ok")

    # Show one small batch and the x/y shift, position by position.
    torch.manual_seed(0)
    xb, yb = get_batch(ds.train_data, block_size=8, batch_size=4)
    print(f"\nbatch x shape: {tuple(xb.shape)}  y shape: {tuple(yb.shape)}")
    print("first row, showing 'given x[:t+1], predict y[t]':")
    for t in range(xb.shape[1]):
        context = ds.decode(xb[0, : t + 1])
        target = ds.decode([yb[0, t]])
        print(f"  given {context!r:<12} -> predict {target!r}")
