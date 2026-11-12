# Running agents on StrixFold: tool calls and repeat loops

StrixFold serves the OpenAI chat API, so any agent harness that speaks it can drive DeepSeek-V4.1-Flash on the pair. One
problem showed up in our own harness after a few days of agent work; we haven't seen it since the fixes below.

## The symptom

The agent writes the same few steps over and over, each with the same tool call:

```
Let me search for the config file.
Let me check the logs.
Let me search for the config file in the project.
Let me check the recent logs.
Let me search for the config file.
...
```

It runs until the harness's step or call limit, then ends on one more "Let me…" line instead of an answer.

## What caused it

- **Greedy decoding.** Our harness sent `temperature: 0` on every request. With the same context shape each step, the
  model picks the same plan and the same calls again, and nothing breaks the cycle.
- **No repeat detection.** A call identical to an earlier one ran again and returned the same result, and nothing told
  the model it was repeating itself.
- **The last request's wording.** When the limits were reached, the harness took the tools away and asked for an
  answer, and the model wrote its next plan line.

## What fixed it

1. **Sample turns that use tools.** Send `temperature: 0.6` with `top_k` and a fresh `seed` on each request:

   ```json
   {"model": "dsv41", "temperature": 0.6, "top_k": 40, "seed": 918273645, "tools": [...], "messages": [...]}
   ```

   - The engine samples only with `top_k` set, from 1 to 56. A temperature above 0 without it is refused with HTTP 400.
   - Without `seed` the server derives the seed from the prompt, so the same prompt returns the same text at any
     temperature. A new seed per request keeps steps from repeating word for word.
   - Plain chat without tools can stay at `temperature: 0`.

   For a harness that sends a temperature but no `top_k` or `seed`, and that you can't change, two settings in
   [`tools/dsv41-hip/serve.conf`](../tools/dsv41-hip/serve.conf) do it on the server:

   - `TOP_K_MAX=56`: a sampled request without `top_k` (or with 0) gets 20, and one above 56 gets 56, instead of a 400.
   - `SEED_RANDOM=1`: a sampled request without `seed` gets a fresh one.

   A `top_k` or `seed` the request sends itself is kept either way.

   What it costs, measured on our pair (drafting on, thinking off, 300-token replies, two runs each):

   | | temperature 0 | temperature 0.6, top_k 40 |
   |---|---:|---:|
   | Code | 59.6 tok/s | 59.2 tok/s |
   | Prose | 45.1 tok/s | 41.3 tok/s |

   The code prompt was "Write a Python function that parses an ISO 8601 duration like P3DT4H5M into seconds, with
   docstring and three tests." and the prose prompt "Explain to a new homeowner how a heat pump works and when it is
   cheaper than a gas furnace." Sampling lowers how often drafted tokens are accepted, so the cost grows with the
   temperature.

2. **Catch repeats in the harness.** Compare each call with the earlier ones in the same answer: the tool and its
   arguments (case and spacing aside), and its result. A call whose arguments and result both match an earlier one
   is a repeat. Give the model a short line instead of the result again, such as "This call repeats one you already
   made, and its result is the same as before. Use that result, try something different, or answer." After three
   repeats, end tool use for that answer and ask for the answer. Comparing results as well keeps legitimate reruns,
   like the tests after an edit, from counting.

3. **Word the last request plainly.** When the limits are reached, say that nothing else will run and ask for what it
   found and what it didn't find or finish, not for the next step.

## Long thinking

Thinking is off in the shipped configuration (`THINKING=0` in serve.conf). With it on, a separate loop can show up in
very long reasoning (over about 25K tokens of thinking): the model repeats the same stretch of tokens inside its
thinking. On main the server's `--loop-guard` only catches cycles of up to 8 tokens, and serve.conf has no setting
that turns it on. Two open pull requests change that:
[#16](https://github.com/Shoestring-dev/StrixFold/pull/16) looks for cycles of up to 64 tokens, and
[#17](https://github.com/Shoestring-dev/StrixFold/pull/17) adds `LOOP_GUARD=1` to serve.conf.
