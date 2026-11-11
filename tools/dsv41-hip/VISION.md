# Optional DeepSeek V4.1 image input on HIP

The DeepSeek V4.1 HIP family can accept image message parts through its normal
lane and draft path. Enable the image modality explicitly. The encoder uses
the checkpoint's vision transformer, aligner, image delimiters and routing
biases to produce BF16 image rows.

## Authenticate image inputs

`vision-checkpoint.json` pins the public LibertAI EXL3 checkpoint revision,
source index, selected shards, extracted encoder and routing biases. Extract
from a checkpoint already downloaded using the standard input workflow.

```sh
python3 tools/dsv41-hip/prepare_vision.py \
  --model /path/to/checkpoint \
  --out /path/to/vision.safetensors \
  --bias-out /path/to/model-kit/vision/gate_bias_vl.safetensors
```

The extractor verifies the index and source shard hashes before copying any
tensors, then verifies the generated files. The encoder rechecks its weights
before use. Place the routing biases in the model kit used by each runtime
process that loads image-capable model weights.

The optional worker requires CPU Torch `2.9.0+cpu`, NumPy `2.3.5` and Pillow
`12.3.0`. `requirements-vision.lock.txt` provides hashes for the Linux x86_64
Python 3.13 wheels. Use a separate environment for these optional dependencies.
The runtime that encodes images needs the encoder weights and environment.

`verify_vision.py` authenticates the routing biases. Supplying `--python`,
`--weights` and `--weights-sha256` also checks the encoder and package versions.
Use the corresponding hashes from `vision-checkpoint.json`.

## Runtime options

Configure the native serving process with these environment variables. Paths
refer to artifacts selected by the operator's existing launch configuration.

| Variable | Meaning |
|---|---|
| `TF_DS_MODALITIES=text_image` | Enable image inputs |
| `TF_DS_VISION_PYTHON` | Absolute path to the isolated Python executable |
| `TF_DS_VISION_WEIGHTS` | Absolute path to the authenticated vision weights |
| `TF_DS_VISION_SHA256` | Encoder weights SHA256 from the input manifest |
| `TF_DS_VISION_CACHE_MIB` | Retained image-row budget, 0–512 MiB, default 0 |
| `TF_DS_VISION_THREADS` | Optional Torch thread count, otherwise Torch chooses |
| `TF_DS_VISION_TIMEOUT` | Worker timeout in seconds, default 240 |

The standard serving scripts accept the corresponding `VISION_*` names in
`serve.conf`, including `VISION_BIAS_SHA256` for the routing-bias file. They
also accept exported `TF_DS_VISION_*` settings when a config entry is empty.
The launcher forwards these settings through its environment allowlist, adds
the selected Python directory to `PATH`, and verifies image inputs before
startup. The deployment helper includes image routing biases when present in
the selected model kit.

The worker starts on demand and exits before language-model prefill. Its input
is a bounded picture grid and BF16 patches. Each image has at most 9,216 vision
patches and 1,024 image-span tokens. The owning request can cancel and reap its
encoder. One parent-side deadline covers child startup, input writes and output
reads. The child also installs its timeout before importing Torch. Text requests use the existing language-model execution path.

PNG preparation uses the native image decoder. Other formats use the existing
Pillow helper, so make the selected Python environment available on the
serving process's `PATH`.

## Optional HIP attention

Build the code object for the architecture selected by the serving kit.

```sh
bash tools/dsv41-hip/build_vision_worker.sh ROCM_ROOT GPU_ARCH OUTPUT_DIR
```

Set `TF_DS_VISION_HIP_CODE` to the absolute code-object path and
`TF_DS_VISION_HIP_SHA256` to its reviewed SHA256. The code object is verified
before loading. `TF_HIP_DEVICE` selects the device through the existing HIP
runtime setting. Attention uses FP32 HIP BLAS operations and bounded softmax
tiles, with BF16 output. The remaining vision operations use the CPU worker.
`TF_DS_VISION_RESERVE_MIB` optionally reserves additional RAM above the
estimated image workspace. Its default is zero.

The CPU path uses the reference math attention backend. The optional HIP path
changes the reduction order and requires independent numerical qualification.
Output checks and release gates apply to both paths.

## Image API and reuse

Chat requests accept PNG/JPEG base64 data URLs in `image_url` parts, including
images in tool results. Image identities follow their content through template rendering, including
templates that reorder tool results into tool-call order.
Tokenization accounts for the expanded image rows.

A nonzero image-row cache budget also enables image-aware prefix retention
when the native prompt cache is enabled. The prefix key includes image content
and span boundaries. Snapshots are kept between image spans. The model receives
the original image tokens and rows. The caches clear on a runtime restart.

## Verification

Run the synthetic image extraction and API tests with the exact Zig version
specified by the repository.

```sh
python3 -m unittest discover -s tools/dsv41-hip -p 'test_*vision*.py' -v
ZIG="$ZIG" bash zig/tests/server/zig_test.sh
python3 tools/zig/lean_check.py
```

The five HTTP cases use the fake server fixture. Set
`TF_VISION_TEST_TOKENIZER` to the selected tokenizer/template directory and
`TF_VISION_TEST_SERVER` to its built `fake_serve` executable. These cases test
rendering, image token spans and streaming. Real model qualification uses
`tools/dsv41-hip/qualify.sh`, including its pinned logits and decode receipts.
Any speed claim also needs the measurements specified in `docs/RESULTS.md`.

Reference source is DeepSeek `inference/vision.py` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, Git blob
`77af0bdef9f4649edebac2876cd1df50a147ce98`. Its MIT license is retained in
`assets/DEEPSEEK_VISION_LICENSE`.
