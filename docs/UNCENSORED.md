# The uncensored DeepSeek, 22% faster

[dealignai's uncensored DeepSeek-V4.1-Flash](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED)
is the same EXL3 2.9 bpw pack StrixFold serves, with 54 of its weights changed. Served as published, it decodes about
a fifth slower than the official pack. StrixFold can instead serve it from the official pack and apply the change
while it runs. You get the uncensored model's answers at 96.5% of the official pack's speed, 22% faster than the
published pack, and the two models share one set of rank caches.

## Why the published pack is slower

The change touches two projections in each of blocks 10 to 36: the attention output (`attn.wo_b`, 5120 × 8192) and
the shared expert's down projection (`ffn.shared_experts.w2`, 5120 × 2304). The published pack stores those 54
weights as plain BF16: 2.70 GiB, against 0.82 GiB as EXL3. Every decode round reads all of them.

One request on a Bosgame M5 pair, 63K context, drafts on, the same prompt and settings:

| Pack | Decode | Per round | Tokens a round | Draft acceptance |
| --- | --- | --- | --- | --- |
| Official EXL3 (c178a012) | 65.4-65.8 tok/s | 61.6-62.1 ms | 4.09 | 0.737 |
| Uncensored, as published (89af9bf1) | 51.6-51.9 tok/s | 75.1-75.7 ms | 3.93 | 0.717 |

Most of the gap is the extra 1.9 GiB a round. A small part is draft acceptance: the draft head is the official
model's, and the uncensored model agrees with it a little less often. That part stays.

## What the change is

Each of the 54 weights is, to its BF16 rounding,

```
W' = diag(g) · (I − α r rᵀ) · W
```

- `W` is the official pack's own weight, decoded from its EXL3 bits. The change was made on that decode, not on
  DeepSeek's FP8 original. The two differ by 4 to 7%, and that difference shows up as noise in any comparison
  against the FP8 weights.
- `r` is one direction among the 5,120 outputs. Each weight has its own.
- `α` is the strength. It is the same for both projections of a block: 3 at blocks 10-11, 4.5 at 12-14, 5.5 at 15-19,
  6 at 20, 5.5 at 21-25, 5 at 26-29, 4 at 30-32, 3 at 33, 2.5 at 34, 1.5 at 35 and 1 at 36. Above 1 it doesn't just
  remove the direction, it reverses it.
- `g` scales each output row back to its original length. It lies between 0.91 and 1.05.

How close that is: fitted to the published weights, the leftover is 1.69-1.71 × 10⁻³ of the weight's norm. Rounding
the exact formula to BF16 already leaves 1.64-1.65 × 10⁻³. All 54 weights fit within 1.032× of that floor, and 85% of
their values come out bit for bit the same as the published ones.

So on the official projection's output `y`, the uncensored output is

```
y' = g ⊙ (y − α r (r · y))
```

That is one dot product and one scaled subtraction over 5,120 floats a token, instead of reading 1.9 GiB more a round.

## Setting it up

1. Make the edit file (2.2 MB) on any machine with Python and numpy:

   ```
   python3 tools/dsv41-hip/make_edit.py unc-edit-89af9bf1.safetensors
   ```

   It reads only the 54 tensors and their EXL3 counterparts (about 3.8 GB, by HTTP ranges from Hugging Face; pass
   `--finetune` and `--base` to read local folders instead, with `--base-id REPO@COMMIT` naming the local base). It
   fits each one and refuses any that isn't such an edit. It also refuses a fine-tune that stores any other weight
   dense, which the edit could not carry. It prints the file's sha256 at the end. On a laptop it takes about 15
   minutes.

2. Copy the file to both boxes and set these lines in `serve.conf`. `BUNDLE` stays on the official pack and its rank
   caches:

   ```
   BUNDLE='$HOME/models/tf-port/inputs-tp2-c178a012'
   EDIT='$HOME/models/tf-port/unc-edit-89af9bf1.safetensors'
   EDIT_SHA256=<the sha256 make_edit.py printed>
   ```

3. Start the server as usual. Both ranks check the file's hash, and that the pack it was made from (its metadata) is
   the bundle's. The handshake and sf-disk's prompt states carry `edit=<hash>`, so they never mix with the unedited
   model's.

You don't need the published pack's own rank caches anymore, which frees about 200 GB across the two boxes.

## How the engine does it

- `dense.hip`'s `edit` kernel runs after `wo_b` and after the shared expert's down projection in the edited blocks:
  one block of 256 threads a row, with a fixed summation order, so a row's result never depends on the other rows in
  its window.
- In an edited block the shared expert runs apart from the routed experts' table, as it already does for the
  published pack, so its output can be edited before it is added.
- The edit is linear, so each box edits its own half of the sum before the two halves are added.
- The four-box 2D split doesn't support it yet and refuses to start with an edit set.
- The paced L2 prefetch (`L2_PREFETCH=1`, off by default) doesn't prefetch an edited block's shared expert.

## Results

On another Bosgame M5 pair, one build for all three runs, the same `serve.conf` apart from the bundle and the edit
(63K context, one stream, drafts on, greedy), the same 2,048-token request twice:

| Weights | Decode | Per round | Tokens a round |
| --- | --- | --- | --- |
| Official pack | 65.8, 65.4 tok/s | 62.5 ms | 4.10 |
| Official pack + the edit | 63.7, 63.2 tok/s | 63.7 ms | 4.02 |
| Uncensored, as published | 52.0, 51.8 tok/s | 75.9 ms | 3.93 |

The edit decodes 22% faster than the published pack, at 96.5% of the official pack's speed. A round costs 1.2 ms more
than the official pack's. That is the edit's two short kernels in each edited block, plus the shared expert running
apart from the routed experts' table. The rest of the gap is fewer accepted drafts a round.

Eight greedy replies (a networking question, a haiku, a Python function, a history question, a crude roast, an adult
joke, picking your own shed's lock and a villain's monologue), compared character by character with the published
pack's:

| Prompt | Edit and published pack agree for | Official and published pack agree for |
| --- | --- | --- |
| TCP vs UDP | 279 characters | 4 |
| Haiku | 16 | 0 |
| Linked list | 395 | 24 |
| World War I | 541 | 69 |
| Roast | 95 | 6 |
| Adult joke | 9 | 0 |
| Lock | 90 | 0 |
| Villain | 9 | 1 |

The edit's replies follow the published pack's word for word until two next tokens are nearly tied, where rounding
decides. The two packs' weights differ by a BF16 rounding, and their arithmetic differs (EXL3 against dense BF16).
The official model refuses the joke and the lock question. The published pack and the edit answer both.

With a 256K context and four streams, a 9,213-token prompt (a ledger, then questions about it) gave the same answer
from both packs: the same town, totals and three largest shipments. The edit read and answered it in 7.9 s, against
9.4 s for the published pack. Four requests at once (a story, a hash-map explanation, a crude limerick and a list) ran
at 60.9 tok/s in total with the edit, against 48.9 with the published pack.

## Credit

Based on [dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED),
which is based on [LibertAIDAI's EXL3 pack](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). StrixFold
redistributes none of them: `make_edit.py` builds the edit on your machine from the two packs, both under the
DeepSeek V4.1 Flash license.
