# Z-Image-Turbo conversion

Exports the Z-Image-Turbo pipeline (Qwen3-4B text encoder → S3-DiT → VAE) to LiteRT
INTEGER-int8 graphs and the precomputed host inputs the app loads. Every heavy stage
becomes an int8 graph; the tokenizer, RoPE, classifier-free-guidance loop and
scheduler stay on the host (reproduced here so the device port is a 1:1 translation).

## Environment

A PyTorch + `litert-torch` conversion env with `diffusers` (Z-Image pipeline),
`ai-edge-litert`, and — for the VAE export — the `litert_gpu_toolkit` GPU-compat
patches from the [LiteRT-Models](https://github.com/john-rocky/LiteRT-Models) repo
(`vae_deploy.py` / `probe_vae.py` import `litert_gpu_toolkit.patches`; put that package
on `PYTHONPATH`).

## Steps

```bash
python chunked_export.py --export   # z_refx / z_refc + zc_main0..5 + zc_final (5-layer chunks)
python prep2.py --export            # z_embx / z_embc + refiners (pad mask moved to the host)
python vae_deploy.py                # zvae_int8_256.tflite (VAE decoder)
python qwen_enc.py                  # text-encoder graph (penultimate hidden)
python gen_prep.py                  # gen_bins/ host inputs + the fp32 reference image
python gen_verify.py                # confirms the exact device loop (corr ~0.99 vs the pipeline)
```

Then stage the graphs and `gen_bins/` with `../android/install_to_device.sh`.

## Why the DiT is split

The monolithic S3-DiT is one ~6 GB int8 graph that exceeds LiteRT's file-load limit and
a phone's GPU budget. It is split into graphs that each compile fully on the GPU delegate
and load one at a time (peak = a single sub-1 GB graph), so it fits an 8 GB phone:

- `z_embx` / `z_refx` — image patch embedder + noise refiner.
- `z_embc` / `z_refc` — caption embedder + context refiner (step-independent).
- `zc_main0..5` — 5 S3-DiT layers each (30 total).
- `zc_final` — final adaLN + projection.

The refined image/context tokens meet as one unified hidden state `[1,288,3840]` on the
host; the composition is bit-exact to the monolithic DiT.

## Device-only fixes (invisible to the desktop op-checker)

- **`bc coord for BATCH axis` compile wall (C19 sibling).** A MUL right after the embed
  FC (the learned pad-token substitution) makes ML Drift assign the token dim to the
  batch axis. Fix: keep the pad-token mask and the x/c concat on the host.
- **fp16 NaN in the adaLN path.** The modulated refiner/main layers overflow fp16; the
  app runs the graphs with `GpuOptions(precision = FP32)`.
- **OpenCL context leak.** Reuse one `Environment` across every `CompiledModel.create`.

## Files

- `build_zimage.py`, `deploy_dit.py`, `full_dit.py`, `check_ops.py` — the GPU-clean S3-DiT
  block reimplementation shared by the exporters.
- `chunked_export.py`, `prep2.py` — the chunk exporters.
- `vae_deploy.py`, `probe_vae.py` — VAE export + probe.
- `qwen_enc.py` — text-encoder export.
- `gen_prep.py`, `gen_verify.py` — host-input precompute + exact-loop verification.
