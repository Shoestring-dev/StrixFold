# l-link: teacher-forced split check, client side (runs on node0 next to rank 0). Each prompt is one max_tokens=1
# request; with TF_DS_WINDOW_LOGITS set, rank 0 dumps the replay window's logits for it (model.zig dumpWindow).
# Prompts: prose (Pride and Prejudice body), code (DeepSeek inference source), README prose; 2K-8K tokens.
import json, sys, time, urllib.request
URL = "http://127.0.0.1:18080/v1/chat/completions"
T = "~/tfp/ref/data/texts/"
prose = open(T + "prose_pg1342_body.txt").read()
# the inference source names the image placeholder, which the server refuses in message text
code = open(T + "code_ds41_inference.txt").read().replace("<｜deepseek_image｜>", "<image>")
readme = open(T + "readme_model.txt").read().replace("<｜deepseek_image｜>", "<image>")
P = [("prose", prose, 0, 11000), ("prose", prose, 100000, 22000), ("prose", prose, 200000, 33000), ("prose", prose, 300000, 45000),
     ("prose", prose, 400000, 11000), ("prose", prose, 500000, 22000), ("prose", prose, 600000, 45000), ("prose", prose, 650000, 33000),
     ("prose", prose, 120000, 15000), ("prose", prose, 250000, 27000), ("prose", prose, 350000, 40000), ("prose", prose, 450000, 8000),
     ("code", code, 0, 8000), ("code", code, 30000, 16000), ("code", code, 50000, 30000), ("code", code, 80000, 20000),
     ("readme", readme, 0, 11000), ("readme", readme, 20000, 30000)]
# set a (default): the 12 prose prompts; set b: the code and README ones; all: both (a first)
SET = sys.argv[1] if len(sys.argv) > 1 else "a"
P = [x for x in P if x[0] == "prose"] + [x for x in P if x[0] != "prose"] if SET == "all" else [x for x in P if (x[0] == "prose") == (SET == "a")]
for i, (kind, src, off, n) in enumerate(P):
    body = {"model": "dsv41", "messages": [{"role": "user", "content": src[off:off + n]}], "temperature": 0, "max_tokens": 1}
    t0 = time.monotonic()
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    print(json.dumps({"i": i, "kind": kind, "off": off, "chars": n, "prompt_tokens": d["usage"]["prompt_tokens"], "wall_s": round(time.monotonic() - t0, 3)}), flush=True)
