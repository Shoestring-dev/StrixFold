# Running agents on StrixFold: tool calls and repeat loops

StrixFold serves the OpenAI chat API, so any agent harness that speaks it can drive DeepSeek-V4.1-Flash on the pair. One
problem showed up in our own harness after a few days of agent work, and the fixes below removed it.

## The symptom

The agent writes the same few steps over and over, each with the same tool call:

```
Let me search for Telegram references.
Let me check their Telegram bot.
Let me search for Telegram integration.
Let me check their bot and payment flow.
Let me search for Telegram references.
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
   - Without `seed` the engine uses a fixed one, so the same prompt returns the same text at any temperature. A new
     seed per request keeps steps from repeating word for word.
   - Plain chat without tools can stay at `temperature: 0`.

   What it costs, measured on our pair (drafting on, 300-token replies, two runs each):

   | | temperature 0 | temperature 0.6, top_k 40 |
   |---|---:|---:|
   | Code | 59.6 tok/s | 59.2 tok/s |
   | Prose | 45.1 tok/s | 41.3 tok/s |

   Sampling lowers how often drafted tokens are accepted, so the cost grows with the temperature. A contributor saw
   about 75 to 45 tok/s at temperature 1.0.

2. **Catch repeats in the harness.** Compare each call with the earlier ones in the same answer: the tool and its
   arguments (case and spacing aside), and its result. A call whose arguments and result both match an earlier one
   is a repeat. Give the model a short line instead of the result again, such as "This call repeats one you already
   made, and its result is the same as before. Use that result, try something different, or answer." After three
   repeats, end tool use for that answer and ask for the answer. Comparing results as well keeps legitimate reruns,
   like the tests after an edit, from counting.

3. **Word the last request plainly.** When the limits are reached, say that nothing else will run and ask for what it
   found and what it didn't find or finish, not for the next step.

## Long thinking

A separate loop can show up in very long reasoning (over about 25K tokens of thinking). A contributor fixed it by
raising the server's loop-guard size and sampling; see the discussion in the project's Discord #contributors channel.
