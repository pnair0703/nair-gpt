# nair-gpt

A decoder-only transformer (a tiny GPT) built **from scratch** in PyTorch, and
trained as a character-level language model that generates text one character at
a time. Multi-head self-attention, causal masking, and positional encoding are
all implemented by hand — no `nn.MultiheadAttention`, no `nn.TransformerDecoderLayer`.

The point of this repo is *understanding*. The commit history walks through the
build phase by phase, and this README tells the story of what got built and how
well it learned.

## The one game a language model plays

> Given the text so far, guess the next character.

Do that well enough, over and over — feeding each guess back in — and the model
writes coherent text. A transformer is a clever machine for that guessing game,
and its central trick is **attention**: to guess the next character, every
position gets to look back at every earlier position and decide which ones matter.

## Architecture (at a glance)

```
characters -> token embedding + positional embedding
           -> N x Transformer Block:
                x = x + MultiHeadAttention(LayerNorm(x))   # mix across positions
                x = x + FeedForward(LayerNorm(x))          # think per position
           -> final LayerNorm -> Linear -> logits over vocabulary
```

## Files

| File | Job |
|------|-----|
| `data.py` | text -> char/id lookup -> batches of `(x, y)` |
| `model.py` | the transformer (attention, blocks, the GPT) |
| `train.py` | the training loop |
| `sample.py` | generate text from a trained model |

## Results

_Filled in at Phase 6 — do not invent these numbers._

- **Parameters:** _TBD_
- **Final train / val loss:** _TBD_
- **Corpus:** _TBD_
- **Config:** _TBD_

### Generation over training (gibberish -> text)

_Samples at a few checkpoints go here — this progression is the whole story._

```
step 0:     TBD
step ...:   TBD
final:      TBD
```

## Run it yourself

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python data.py     # sanity-check the data pipeline
python train.py    # train (writes a checkpoint)
python sample.py   # generate text
```
