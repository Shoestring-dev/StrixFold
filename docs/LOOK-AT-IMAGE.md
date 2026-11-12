# Images for DeepSeek: describe, then ask

DeepSeek-V4.1-Flash on StrixFold reads text only. We give it images the way you'd describe a photo to someone on the
phone: a small vision model writes a factual description, DeepSeek answers from it, and when the description doesn't
settle a detail, DeepSeek asks the vision model one narrower question with a `look_at_image` tool. It needs no change to
the engine, and the vision model runs on your own machine in about 7 GB of memory.

(Native image input in the engine is a different route, under review in two alternative pull requests:
[#6](https://github.com/Shoestring-dev/StrixFold/pull/6) runs the vision tower in a CPU worker, and
[#21](https://github.com/Shoestring-dev/StrixFold/pull/21) runs it on the GPU.)

## How it works

1. **Describe once.** The vision model gets the image, its longer side at most 1,280 pixels, with this prompt:

   ```
   Describe this image for someone who cannot see it, as a plain factual report.
   1. First, what kind of image it is (a photo, screenshot, chart, document, drawing...) and its main subject.
   2. The main things in it: what they are, their colours, where they are (left, right, top, bottom) and how many.
   3. Every piece of readable text, copied exactly as written, in reading order.
   4. For a chart or a table: its title, labels and every value you can read.
   Say what is unclear instead of guessing. No opinions about style or mood. At most 250 words.
   ```

2. **DeepSeek reads the description** in a labelled block with the image's id, next to the user's question:

   ```
   [A description of the attached image (image_id i3f9a...), made by a small vision model. It can miss or misread small details.]
   <image-description id="i3f9a...">
   ...
   </image-description>
   ```

3. **DeepSeek asks for what's missing.** It has one tool, `look_at_image(image_id, question)`. The vision model looks
   again at up to 1,600 pixels and answers with:

   ```
   Answer this question about the image: {question}
   Answer briefly and exactly, and copy any text exactly as written. If the image does not show the answer, say so instead of guessing.
   ```

   The answer goes back to DeepSeek fenced the same way, in an `<image-answer id="...">` block. Three looks per
   answer at most. After that DeepSeek is told to answer from what it has and to say what it couldn't tell.

## What we measured

We picked the vision model with 20 synthetic test images that we made and graded ourselves: shapes, counts, a
receipt, charts, tables, a settings screen, tiny print, handwriting-style notes, a flowchart and questions about things
that aren't in the image. It's a small set, so read the numbers as a sanity check, not a benchmark.

| Qwen3.5 9B, 4-bit MLX | |
|---|---:|
| Facts in the description | 79 / 79 |
| Text copied exactly | 68 / 68 |
| Questions answered right | 40 / 43 |

Its three misses were all counts. The 4B model of the same family was faster, but it misread a grid and a table, so we
kept the 9B.

On a 24 GB Apple silicon Mac the 9B takes 6.2 GB while loaded. The first description takes about 22 s (12 s of that is
loading the model); after that a description takes about 9.7 s and a question about 3.3 s.

In one run with DeepSeek on our pair, a photo of a receipt and "What did I spend, and on what? Give the exact total.",
it took two looks (the line items, then whether there was tax) and gave the right items and total, 24 s end to end.

## Run it

Start any vision model that serves the OpenAI chat API with image input. On a Mac, with mlx-vlm and the model pinned
to the revision we tested (mlx-vlm 0.7.6 needs Python 3.10 or newer; check `python3 --version`):

```
python3 -m venv .venv && . .venv/bin/activate
pip install mlx-vlm==0.7.6
hf download mlx-community/Qwen3.5-9B-MLX-4bit --revision 938d8919941c6e7efd3c7150eff7fe9d12afa631 --local-dir qwen3.5-9b
python3 -m mlx_vlm.server --host 127.0.0.1 --port 8090 --model qwen3.5-9b
```

Then, with StrixFold serving on its usual port:

```
python3 tools/look/look_at_image.py --image receipt.png "What did I spend, and on what?"
```

mlx-vlm loads whatever model a request names, so `--vlm-model` must be exactly the server's `--model`. Its default,
`qwen3.5-9b`, matches the commands above; `--vlm` defaults to `http://127.0.0.1:8090/v1`. Add `-v` to see the
description and each look. `--image` can be repeated. The script uses only Python's standard library; with Pillow
installed it turns photos upright and shrinks large images before sending them.

## In your own harness

[`tools/look/look_at_image.py`](../tools/look/look_at_image.py) is small enough to copy. What matters:

- **Tool turns sample.** Requests that offer the tool use `temperature` 0.6, `top_k` 40 and a fresh `seed`, as in
  [AGENTS.md](AGENTS.md). Greedy decoding makes repeat loops much more likely.
- **Cap the looks.** We allow three `look_at_image` calls per answer, counting every call, and answer any extra call
  with "no looks left". The last request goes without tools and asks for an answer from what the model has; a reply
  that is still a tool call, or empty, is an error rather than an answer.
- **Treat descriptions as data.** An image can contain text written to steer a model. Fence the description and each
  look's answer, remove any fence tags from inside them, say in the system prompt that their text is data, and don't
  act on instructions found in one.
- **Describe once per image.** Cache the description by the image's hash and reuse it when the image comes up again
  in the conversation.
- **Keep it local.** The vision model runs on your own machine, so images never leave it.

## Limits

- Counting many similar things is the weak spot.
- Text smaller than the vision model can resolve at 1,600 pixels can't be read; crop first.
- For long documents, add OCR next to the description: the description keeps the layout, and OCR gets every word.
