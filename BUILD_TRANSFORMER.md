# Build a Transformer From Scratch — Build Guide

**Goal:** implement a decoder-only transformer (a tiny GPT) in PyTorch, from scratch, and train it as a character-level language model that generates text. The point of this project is **understanding**, not the artifact — you must be able to explain every piece in an interview.

**The one rule:** you write the attention mechanism and the transformer block yourself. Do NOT `import nn.MultiheadAttention` or `nn.TransformerDecoderLayer`. Those are the "black box" versions and they defeat the entire purpose. You may import basic primitives: `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `F.softmax`, `F.cross_entropy`, the optimizer.

**How to use this with Claude Code:** work through the phases in order. For each phase, the "You implement" parts are yours to write first — struggle with them a little, that's where the learning is. Ask Claude Code to review what you wrote, explain errors, or scaffold the "boilerplate" parts (data loading, training loop plumbing). Don't have it write the attention block for you.

---

## The dumb version of what we're building

A language model plays one game: **given the text so far, guess the next character.** That's it. "T-H-E-space-C-A-" → probably "T". Do that well enough, over and over, and it writes coherent text by guessing one character at a time, feeding its own guess back in.

A transformer is a particular clever way to play that game. Its trick is called **attention**: to guess the next character, every position in the text gets to "look at" every earlier position and decide which ones matter. When you're about to write a closing bracket, the position that holds the opening bracket is the one worth attending to. Attention is the machinery that lets each position pull in the context it needs.

The whole model is: turn characters into vectors → let them attend to each other (several times, in stacked layers) → at the end, each position outputs a guess for the next character. Training nudges all the knobs so the guesses get better.

---

## The pieces, in plain language

Think of it as an assembly line. Text goes in the left, next-character predictions come out the right.

1. **Tokenizer (characters → numbers).** Computers don't eat letters. Build a lookup: every unique character in your text gets an integer id. "hello" → `[7, 4, 11, 11, 14]`. That's the whole "tokenizer" for a char-level model — no fancy subword stuff.

2. **Token embedding (numbers → vectors).** An integer id carries no meaning (id 7 isn't "bigger" than id 4 in any useful sense). So each id looks up a learned vector — a list of, say, 64 numbers — that *does* carry meaning the model shapes during training. This is `nn.Embedding`.

3. **Positional embedding (where am I?).** Attention, by itself, has no sense of order — it sees a bag of vectors. But "dog bites man" ≠ "man bites dog." So we add a second learned vector that encodes *position* (1st slot, 2nd slot, ...). Now each vector knows both what it is and where it sits.

4. **Self-attention (the actual idea).** For each position, produce three vectors:
   - a **Query** ("what am I looking for?")
   - a **Key** ("what do I offer?")
   - a **Value** ("what I'll hand over if you attend to me")

   Each position's Query is compared against every position's Key (a dot product). High match = "this position is relevant to me." Softmax those match-scores into weights that sum to 1, then take a weighted average of the Values. The result: every position has pulled in a blend of information from the positions it cares about.

   Two details that are interview gold:
   - **Scale by √(head dim).** The dot products grow larger as the vectors get longer, which shoves softmax into a corner where gradients vanish. Dividing by √(dk) keeps them tame.
   - **Causal mask.** Position 5 is only allowed to look at positions 1–5, never 6+. If it could see the future, predicting "the next character" would be cheating — the answer would be right there. We enforce this by setting the not-allowed match-scores to −∞ before the softmax, so they get weight 0.

5. **Multi-head attention.** Do the attention thing several times in parallel with separate Q/K/V projections ("heads"), each free to specialize — one head might track brackets, another might track subject-verb agreement. Concatenate their outputs. Mechanically: split the vector into `n_heads` chunks, attend within each, glue back together.

6. **Feed-forward network.** After attention mixes information *across* positions, a small 2-layer MLP processes each position *independently* — expand to a bigger width, apply a nonlinearity (GELU/ReLU), shrink back. This is where a lot of the model's "thinking" capacity lives.

7. **Residual connections + LayerNorm.** Deep stacks are hard to train. Two standard fixes:
   - **Residual:** `x = x + block(x)` — the block only has to learn a *correction* to its input, and gradients get a clean path back. 
   - **LayerNorm:** re-center and re-scale each vector so numbers don't drift into unstable ranges.
   
   A "transformer block" = attention (with residual+norm) then feed-forward (with residual+norm). Stack N of these.

8. **Output head (vectors → guesses).** A final `nn.Linear` maps each position's vector to one score per possible character (the "logits"). Softmax turns scores into probabilities. The highest is the model's bet for the next character.

9. **Loss (how wrong were we?).** Cross-entropy compares the predicted distribution against the actual next character. Low loss = confident and correct. Training = nudge every knob to lower this, via backprop + an optimizer (Adam).

10. **Generation (using the thing).** Feed a seed string, get the next-char distribution, sample one character, append it, repeat. Watch gibberish slowly turn into text-that-looks-like-your-corpus as training progresses. This is the payoff moment.

---

## Phases

### Phase 0 — setup
- `pip install torch`
- One folder. Files: `data.py`, `model.py`, `train.py`, `sample.py` (or one `gpt.py` to start — split later).
- Grab a corpus: TinyShakespeare (~1MB `input.txt`) is the classic. Or any plain-text file you like — your own writing is a nice personal touch.
- **You implement:** nothing yet.
- **Claude Code can:** scaffold empty files, download the corpus.

### Phase 1 — data
- Read the text, build `char → id` and `id → char` dicts, encode the whole corpus to one long tensor of ids.
- Write `get_batch()`: randomly grab B chunks of length T, return `x` (the chunk) and `y` (the chunk shifted one character left — the "correct next char" at each position).
- Split off the last ~10% as validation.
- **You implement:** the encode/decode and `get_batch`. This is where "predict the next char" becomes concrete — writing the x/y shift yourself makes the whole objective click.
- **Claude Code can:** review, catch off-by-one errors.

### Phase 2 — the attention block  ← THE HEART, WRITE THIS YOURSELF
- A `Head` (or fused multi-head `SelfAttention`) module: Q/K/V linear projections, scaled dot-product, causal mask, softmax, weighted sum of V.
- Then multi-head: run several in parallel, concatenate, project out.
- **You implement:** all of it, by hand. This is the module the whole project exists to teach you. If you write nothing else yourself, write this.
- **Claude Code can:** explain a shape-mismatch error, confirm your mask is right — but should NOT hand you the block. Ask it to *check*, not to *write*.
- Sanity check: feed a random `(B, T, C)` tensor in, confirm you get `(B, T, C)` out, confirm row weights sum to 1, confirm position t has zero weight on t+1.

### Phase 3 — the block and the full model
- `FeedForward`: Linear → GELU → Linear.
- `Block`: `x = x + attn(norm(x))`, then `x = x + ffn(norm(x))` (pre-norm style).
- `GPT`: token embedding + positional embedding → N blocks → final norm → output Linear. Add the forward pass that also computes cross-entropy loss when given targets.
- **You implement:** the block wiring and the model's forward pass — you want to be able to trace a tensor start to finish.
- **Claude Code can:** scaffold the module boilerplate, wire up config.

### Phase 4 — training loop
- Adam optimizer, loop: `get_batch` → forward → loss → `backward` → `step` → `zero_grad`. Print train/val loss every N steps.
- Tiny config first (2 layers, 2 heads, 64-dim, block size 64) — must overfit a tiny slice, proving the wiring works. Then scale up.
- **You implement:** you can write this or let Claude Code scaffold it — it's plumbing, not concept. Understand it, don't sweat writing every line.
- **Claude Code can:** write most of this; you review.

### Phase 5 — generation
- `generate(seed, n)`: loop n times — forward, take last position's logits, softmax, `torch.multinomial` to sample, append, repeat. Crop context to block size.
- **You implement:** the sampling loop — it's short and it's satisfying to write the part that actually produces text.
- **Claude Code can:** review.

### Phase 6 — measure and write it up
- Report: param count, final train/val loss, a sample of generated text at a few training checkpoints (gibberish → words → sentence-shaped).
- A `README.md` with the architecture, a loss curve, and generation samples. This README is what a recruiter clicks — make it show the progression.

---

## The bullet this becomes

> Implemented a decoder-only transformer from scratch in PyTorch — multi-head self-attention, causal masking, positional encoding — trained as a character-level language model ([N]M params, [X] val loss) that generates coherent text

Fill the brackets from Phase 6. Don't invent them.

---

## Interview questions this project must let you answer

If you can't answer these after building it, you built it too passively — go back and make sure you understand, not just that it runs.

- Walk me through self-attention. What are Q, K, V?
- Why divide by √(dk)?
- What's the causal mask and why do we need it for a language model?
- Why multiple heads instead of one big attention?
- What do the residual connections and LayerNorm do for training?
- Where does the model mix information across positions vs. process each position alone? (attention vs. FFN)
- How does this relate to the model you QLoRA-fine-tuned? (You adapted a pretrained transformer; now you understand the thing underneath the adapters.)

---

## Guardrails

- Don't import prebuilt attention/transformer layers. The whole value is in Phase 2 being yours.
- Get the tiny config overfitting a small slice *before* scaling — it's the fastest way to catch wiring bugs.
- Keep it CPU-runnable at tiny size; move to a free Colab T4 only when you want it to generate good text.
- The deliverable is a README that shows the gibberish→text progression. That progression is the story; a bare loss number isn't.
