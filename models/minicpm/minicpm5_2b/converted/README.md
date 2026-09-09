# MiniCPM5-2B Conversion

Model recipes, instructions, scripts, and utilities for converting MiniCPM5-2B to LiteRT-LM.

These scripts convert [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) (dense `LlamaForCausalLM`, Apache-2.0, 2.5B parameters, hybrid thinking) into `.litertlm` bundles for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime that run on **both** the CPU and the GPU backend, and verify the result with an 8-question gate, a metadata check and a thinking-mode check. The published artifacts are `MiniCPM5-2B_int4.litertlm` (1.55 GB) and `MiniCPM5-2B_int8.litertlm` (2.60 GB) at [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B).

The model is a 42-layer dense transformer (hidden 2048, GQA 16:2, head_dim 128, intermediate 6144, vocab 130,560, untied embeddings, rope_theta 5e6, no rope scaling — 2.52 B parameters, 1.98 B non-embedding). The architecture needs nothing from the exporter. What this conversion documents is **how a bundle can pass on one backend and fail on the other**, and the two post-export steps that close that gap — each found by gating on both backends, each measured before it went into the recipe.

## Build

| recipe | quantization | post-export step | file | use |
|---|---|---|---|---|
| `int4` | blockwise-32 int4 + OCTAV clipping on linears, int8 embedding (`BOCTAV4`, registered by the build script) | zero-scale fix (in place) | 1.55 GB | the phone file: fastest GPU decode everywhere measured; direct answers or short reasoning |
| `int8` | dynamic per-channel int8 on linears + embedding (`dynamic_wi8_afp32`) | fp32 activations declared (repack) | 2.60 GB | reasoning that has to complete; desktop / Android (2.33 GB main section is above the iOS single-section mmap ceiling) |

Both bundles: the checkpoint's `chat_template.jinja` embedded byte for byte (`use_jinja_template=True`), `thought` channel declared, `start_token` `<s>`, stop ids 1 (`</s>`) and 130073 (`<|im_end|>`), 4096-token cache, six prefill signatures (1024, 256, 64, 16, 4, 1), embedding table in its own section (`externalize_embedder=True`).

## Environment

```bash
pip install litert-torch==0.9.3 litert-converter==0.4.0 ai-edge-quantizer==0.9.0 \
    litert-lm-builder==0.16.1 transformers==5.14.1
pip install litert-lm   # >= 0.16 (thought channel + ThinkingConfig); verification here on 0.17.0
```

The published bundles were built on exactly this released stack — no patched checkout.

## Run

```bash
python build_minicpm5_2b.py --recipe int4 --out out_int4      # ~20 min on an M4 Max (quantization ~5 min of it)
python build_minicpm5_2b.py --recipe int8 --out out_int8      # ~10 min

python verify_minicpm5_2b.py out_int4/MiniCPM5-2B_int4.litertlm --backend cpu    # 8-question gate
python verify_minicpm5_2b.py out_int4/MiniCPM5-2B_int4.litertlm --backend gpu    # same gate, GPU — run both
python verify_minicpm5_2b.py out_int4/MiniCPM5-2B_int4.litertlm --check-metadata # template bytes, start/stop tokens, thought channel, sections, zero scales
python verify_minicpm5_2b.py out_int8/MiniCPM5-2B_int8.litertlm --check-metadata --expect-activation fp32
python verify_minicpm5_2b.py out_int8/MiniCPM5-2B_int8.litertlm --backend gpu --thinking-ab   # the question that decided the fp32 declaration
```

The gate runs each question through `litert-lm run` (greedy, fresh session per question). The model thinks by default, so the verify script scores only the text after the thought channel — reasoning text routinely contains the expected string while checking alternatives, which would fake a pass.

## What the export alone gets wrong, and how each fix was established

### 1. int4 loads on the GPU and refuses on the CPU — zero block scales

First gate on the raw int4 export: GPU 8/8; CPU did not reach the first token:

```
unsupported scale value (0.000000) in channel 15616 for INT4 tensor 372
Failed to allocate tensors
```

Read from the flatbuffer: decoder layer 0's `gate_proj` and `up_proj` each carry 832 zero blocks out of 393,216 — 832 = 13 rows × 64 blocks, i.e. **13 all-zero MLP neurons in layer 0**, on the same rows in both projections. No other tensor in the 42 layers, and not the int8 embedder, has a zero scale. Blockwise quantization emits scale 0 for an all-zero block; XNNPACK rejects a zero scale, the GPU delegate does not check.

`fix_zero_scales_inplace.py` replaces each zero scale by the tensor's smallest nonzero scale, patched at the scale words' absolute offsets inside the bundle. The quantized values in a zero block are 0, so the dequantized weights are unchanged; 3,328 bytes change (2 × 1,664 fp16 scales), file size and every other section byte stay identical. The fixed file then gates 8/8 on CPU and 8/8 on GPU. Per-channel int8 has no zero scales on this checkpoint, so the int8 file needs no fix.

Why in place rather than a repack: a repack through litert-lm-builder with one TFLite section would drop the externalized embedder section.

### 2. int8 passes on the CPU and runs away on the GPU — fp16 activations

The raw int8 export gates 8/8 on CPU and **7/8 on GPU**: on the rhyme question the model emits 2,021 thought tokens and never closes `</think>`, so no answer arrives. Reproduced in isolation on the same file (`--thinking-ab` runs it). The GPU executor keeps activations in fp16 by default; over 42 layers that is enough to steer a long reasoning chain away from its termination.

`set_activation_type.py --type fp32` unpacks the bundle, adds `prefer_activation_type = "fp32"` under the TFLiteModel section of `model.toml`, and packs it again; the packed bundle carries it as a section metadata item (`litert-lm unpack` shows it as an `additional_metadata` entry, the peek output as `Key: prefer_activation_type`) — every section's bytes are identical, only that item is new. With it the rhyme closes in 452 tokens and the GPU gate is 8/8. The cost on an M4 Max is ~18 % prefill and ~14 % decode (measured on identical int4 weights with and without the declaration: 1699→1398 prefill, 92.8→80.2 decode tok/s).

The declaration is **not** a blanket fix. On the int4 file the fp16 default passes the gate, and declaring fp32 reproduces the CPU reference's behaviour — thinking chains that do not terminate within 3,584 tokens (see §Thinking mode). Measure per file; the verify script's `--expect-activation` asserts what the bundle declares so the choice is recorded, not assumed.

### 3. Template packaging: verbatim Jinja, and why `<s>` is right

The structured-prefix path (`use_jinja_template=False`) cannot carry the template's `enable_thinking` branch or the tool-calling format, so this recipe embeds the checkpoint's `chat_template.jinja` verbatim — the same packaging as `litert-community/MiniCPM5-1B`. litert-torch declares the `thought` channel from the `<think>` literal in the template; `--check-metadata` reads it back out of the bundle rather than assuming it.

The template starts with `{{ bos_token }}`. At runtime that renders empty and the engine prepends the metadata `start_token`, so the model sees exactly one `<s>` — the same stream `apply_chat_template` produces upstream (`add_bos_token: false`, post-processor adds `<s>`). `start_token` therefore stays `<s>` here; on a checkpoint whose template renders its own BOS literal the opposite holds (see the Hy-MT2 recipe).

The exporter also writes 14 string stop tokens of the form `X<|im_end|>\n` next to the two ids. They come from litert-torch 0.9.3's export builder, not from the tokenizer (no vocabulary piece contains `<|im_end|>`), and are harmless — ids 1 and 130073 do the stopping.

### 4. Weight-only packaging loses the GPU backend

The same repo carries two weight-only exports of this model (`minicpm_wi8_afp32.litertlm`, `minicpm_wi4c_wi8_afp32.litertlm`: every linear stored as `DEQUANTIZE` → fp32 `FULLY_CONNECTED`, the int4 one with channelwise int4 plus a Hadamard rotation). Both answer on the CPU backend. On the GPU backend (litert-lm 0.17.0, Mac) both fail at engine creation: the delegate rejects the dequantized 2048×2048 weight — `Shape mismatch: {bhwc, {2048, 1, 1, 2048}} vs {bhwc, {1, 1, 2048, 2048}}` → `Failed to modify graph with delegate`. The int4 one has a second blocker before that: `Following operations are not supported by GPU delegate: CUSTOM aeq.hadamard_rotation` (two per layer; 216 of 2,121 ops would delegate). The recipe above uses *dynamic* recipes for that reason — the delegate consumes the quantized weights directly, with no `DEQUANTIZE` and no custom op, and the Galaxy S26 delegates 1873/1873 nodes. The verify script reports the weight-only int4 file as `--check-metadata` 7/8 (only `max_num_tokens` differs) and CPU gate 7/8 with one degenerate row (the "0.9 or 0.11" reasoning loops for 22k characters and never answers).

## Thinking mode

Thinking is the model's default: with no `ThinkingConfig` it reasons before every answer. `enable_thinking=false` (ThinkingConfig or the conversation's extra context — both reach the template) switches it to direct answers; `true` pre-fills `<think>\n`. All three modes hold over three-turn conversations on both files.

| int4 / int8, thinking off, GSM8K n=100, greedy, max 2048 new tokens (the protocol OpenBMB's cards use) | GSM8K |
|---|---|
| bf16 PyTorch (MPS), upstream template | 92 % |
| int8, CPU | 91 % |
| int4, CPU / GPU fp16 default / GPU fp32 declared | 86 % / 87 % / 88 % |

int8 is at parity (7 of its 9 misses are the bf16 model's own). int4 costs ~5 points; the activation dtype costs nothing measurable with thinking off.

With thinking **on** (ten GSM8K questions, 3,584-token budget), the picture changes: the bf16 model closes its reasoning on 9/10 with ~3,000-character chains and int8 on CPU reproduces that question for question (9/10, ~3,200 characters), while **int4 closes 0/10 on CPU** (~13,700 characters — it keeps re-checking until the budget runs out; the answer is usually right inside the thought text but is never emitted). The int4 GPU path with fp16 activations happened to close 10/10 with ~8,000-character chains; with fp32 declared it matches the CPU reference (0/10). So the thinking-length inflation is int4 quantization damage, and the fp16 rounding masks half of it — a reason to keep the fp16 default on int4, not a property to rely on. A block-128 int4 variant was built and rejected: −8 GSM8K points (79 vs 87), 7/8 on the CPU gate, and the chains do not get shorter.

## Verification record

All numbers on the published bundles; litert-lm 0.17.0 unless noted.

- **8-question gate** (Mac M4 Max): int4 8/8 CPU, 8/8 GPU; int8 (fp32 declared) 8/8 CPU, 8/8 GPU. Raw int8 (fp16 default) 7/8 GPU with the rhyme question degenerate — the one-variable case behind §2.
- **Galaxy S26** (SM-S942Q, Adreno; v0.16.0 kit `litert_lm_advanced_main`): both files generate on GPU and CPU with full OpenCL delegation — 1873/1873 nodes on every prefill signature and decode, 0 rejected ops. int4 GPU 401–411 prefill / 16.1–18.6 decode tok/s (TTFT 0.56 s, peak RSS 1.14 GB); int8 GPU 150–160 / 10.9–12.8 (peak 1.10 GB).
- **iPhone 17 Pro**: int4 7/8 on Metal GPU (init 5.7 s) and 7/8 on CPU (init 2.2 s); the one miss on each leg is the rhyme line inside a composite prompt, answered "green" — the same answer the bf16 model gives with thinking off.
- **Mac M4 Max** (`litert-lm benchmark -p 256 -d 256 --runs 3 --cache no --max-num-tokens 1024`, quiet machine): int4 GPU 1699 prefill / 92.8 decode tok/s, CPU 149 / 31.1; int8 (fp32) GPU 1405 / 74.7, CPU 161 / 30.0.
- **Re-run with the scripts in this directory (2026-09-09, litert-lm 0.17.0, Mac M4 Max, published files downloaded from the Hub, sha256 verified)**: `--check-metadata` 9/9 on both files (`--recipe int4 --expect-activation fp16`, `--recipe int8 --expect-activation fp32`); gate int4 CPU 8/8, int4 GPU 8/8, int8 CPU 8/8, int8 GPU 8/8, no degenerate row (thought lengths 126–752 characters, except the int8 GPU rhyme at 1,726); `--thinking-ab --backend gpu` on int8: thought closed after 1,726 characters, answer "blue".
