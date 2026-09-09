# MiniCPM5-2B

Model recipes, conversion scripts, instructions, and utilities for [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) — OpenBMB's 2.5B-parameter dense hybrid-reasoning model (42 layers, hidden 2048, GQA 16:2, 130,560-token vocabulary, one checkpoint that answers directly or reasons inside `<think>…</think>` depending on `enable_thinking`).

* [`converted/`](converted/) — HF checkpoint → `.litertlm` for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime: the build script, the two post-export fixes, and a verify script with an 8-question gate, a metadata check and a thinking-mode check.

Published bundles built with this recipe: [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B) (`MiniCPM5-2B_int4.litertlm`, `MiniCPM5-2B_int8.litertlm`).

## Why this recipe exists

The architecture is a plain `LlamaForCausalLM`; the stock exporter handles it. What decides whether the bundle runs on **both** backends is two things the export alone does not give you, each visible only on one backend:

| symptom | backend that shows it | cause | fix in this recipe |
|---|---|---|---|
| int4 bundle fails at load: `unsupported scale value (0.000000) … for INT4 tensor` | **CPU** (XNNPACK); the GPU loads the same file silently | decoder layer 0's MLP has 13 all-zero rows → blockwise quantization emits scale 0 for their blocks | `fix_zero_scales_inplace.py` sets those scales to an epsilon in place (dequantized weights unchanged; 3,328 bytes of a 1.55 GB file) |
| int8 bundle never closes `</think>` on a short question (2000+ tokens, empty answer) | **GPU** (fp16 activations by default); CPU answers | fp16 activation rounding over 42 layers steers the reasoning chain | `set_activation_type.py --type fp32` declares fp32 activations in the bundle (a repack, weights untouched); the question closes in ~450 tokens |

A bundle that passes on one backend can fail on the other, and the failure modes are different. Gate on both.

## Measured (Mac M4 Max, litert-lm 0.17.0; Galaxy S26; iPhone 17 Pro)

| file | 8-question gate CPU / GPU | GSM8K n=100, thinking off (bf16 reference 92 %) | Galaxy S26 GPU (OpenCL) decode | M4 Max GPU (Metal) decode |
|---|---|---|---|---|
| `MiniCPM5-2B_int4.litertlm` (blockwise-32 int4 + OCTAV, int8 embedding; zero-scale fix) | 8/8 · 8/8 | 86 % (CPU) · 87 % (GPU) | 16.1–18.6 tok/s, 1873/1873 nodes delegated | 92.8 tok/s |
| `MiniCPM5-2B_int8.litertlm` (dynamic int8; fp32 activations declared) | 8/8 · 8/8 | 91 % (CPU) | 10.9–12.8 tok/s | 74.7 tok/s |

Thinking mode is where int4 shows its damage: with thinking on and a 3584-token budget, int8 closes its reasoning on 9/10 GSM8K questions with ~3,000-character chains (the bf16 model does the same), int4 closes 0/10 on CPU (~13,700-character chains). The card and [`converted/README.md`](converted/README.md) carry the full tables and the conditions behind every number.
