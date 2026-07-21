# nair-gpt

A decoder-only transformer (a tiny GPT) built from scratch in PyTorch, and
trained as a character-level language model that generates text one character at
a time. Multi-head self-attention, causal masking, and positional encoding are
all implemented by hand 
## Architecture 

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

_Filled in at Phase 6 

- **Parameters:** _TBD_
- **Final train / val loss:** _TBD_
- **Corpus:** _TBD_
- **Config:** _TBD_

### Generation over training


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
