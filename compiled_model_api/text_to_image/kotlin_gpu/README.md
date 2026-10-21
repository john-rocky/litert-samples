# Z-Image-Turbo — on-device text-to-image (LiteRT CompiledModel GPU)

Alibaba Tongyi-MAI **Z-Image-Turbo** (6B, Apache-2.0) — a Single-Stream Diffusion
Transformer (S3-DiT) — generating images fully on the phone GPU through LiteRT
`CompiledModel`. Via chunked sequential residency it is verified generating a real
image end-to-end on a commodity 8 GB phone (Pixel 8a, Mali GPU).

![generated on a Pixel 8a](docs/pixel8a_generated.png)

*Generated entirely on a Pixel 8a Mali GPU, prompt "a red apple on a wooden table,
studio lighting", 8 flow-matching steps.*

## Pipeline

Qwen3-4B text encoder → S3-DiT denoiser (8 steps, FlowMatchEuler with classifier-free
guidance) → VAE decoder. Every heavy stage is a LiteRT **INTEGER-int8** graph (int×int
compute — the path the GPU delegate runs; weight-only-FLOAT hangs on the delegate).

## Chunked sequential residency

The monolithic S3-DiT is one ~6 GB int8 graph, over LiteRT's file-load limit and a
phone's GPU budget. It is split into graphs that each compile fully on the ML Drift GPU
delegate and load one at a time, so the peak footprint is a single graph:

| Graph | Role | int8 size |
|-------|------|-----------|
| `z_embx` / `z_refx` | image embed + noise refiner | 0.3 / 363 MB |
| `z_embc` / `z_refc` | caption embed + context refiner | 10 / 355 MB |
| `zc_main0..5` | 5 S3-DiT layers each (30 total) | 866 MB × 6 |
| `zc_final` | final adaLN + projection | 1.2 MB |
| `zvae` | VAE decoder | 50 MB |

The refined image/context tokens meet as one unified hidden state `[1,288,3840]` passed
between chunks; the composition is bit-exact to the monolithic DiT, and the on-device
image matches the fp32 reference at PSNR 40.4 dB (corr 0.9994).

## Device constraints solved (GPU-delegate-only, invisible to the desktop op-checker)

- **`bc coord for BATCH axis` compile wall.** A MUL right after the embed FC (the pad-token
  substitution) makes ML Drift assign the token dim to the batch axis and the OpenCL
  kernel fails to generate. Fix: move the pad-token mask (`x*(1-pad) + pad_token*pad`) and
  the x/c concat to the **host** — the graphs are then a bare FC or a refiner stack.
- **fp16 NaN in the adaLN path.** The modulated refiner/main layers overflow fp16 to
  all-NaN; the context refiner (no modulation) is clean. Fix: `GpuOptions(precision = FP32)`.
- **Cumulative OpenCL context leak.** A null `Environment` per `CompiledModel.create` aborts
  the process after ~20 FP32 compiles. Fix: share one `Environment`; close each input/output
  `TensorBuffer` per run.

## Host-side glue (the Python in `conversion/` is authoritative)

1. Tokenize + embed the prompt, run the **encoder graph**, slice to the valid length →
   `cap_feats`.
2. Precompute the fixed inputs: 3-axis **RoPE** `cos/sin`, per-step **adaLN**, FlowMatchEuler
   **sigmas**, the initial latent, and the patchify / unpatchify index permutations.
3. Per step (latent space): patchify → run the chunked DiT for **cond** and **uncond** →
   `noise_pred = -(pos + guidance*(pos - neg))` (the Z-Image CFG, negated; batch 2 whenever
   `guidance > 0`) → `latent += dsigma * noise_pred`. The image branch (`embx → refx`) is
   shared; the context branch (`embc → refc`) is step-independent.
4. Decode the final latent with the **VAE graph** → RGB.

## Build & run

1. Convert the graphs — see `conversion/README.md`.
2. Stage them to the device: `android/install_to_device.sh <dir-with-graphs-and-gen_bins>`
   (they are pushed, never bundled; the first launch before staging shows "model not found").
3. Build + run: `cd android && ./gradlew :app:installDebug`, then tap **Generate**. The full
   8-step loop runs on the GPU (~32 min on a Pixel 8a, FP32-compute, recompile-per-step).

`minSdk 26`, `arm64-v8a`, LiteRT `CompiledModel` GPU.

### Kotlin

- `ZImageGenerator.kt` — the full generation loop (chunked DiT cond+uncond, Z-Image CFG,
  FlowMatchEuler, unpatchify, VAE decode → bitmap).
- `ChunkRunner.kt` — loads one graph on the GPU (FP32, shared `Environment`, per-run buffer
  free) and returns its output.
- MVVM: `MainViewModel` drives generation on a confined worker; `ZImageScreen` shows the
  prompt, a Generate button, progress, and the image.

## Quality

int8 (INTEGER-compute) renders a faithful image: the on-device output matches the fp32
reference at PSNR 40.4 dB (corr 0.9994), visually indistinguishable. int4 is garbage
(PSNR 18), so int8 is the floor.

Two host-loop details each cost a visible per-patch mesh if missed: the cond and uncond
prompts differ in length, so each branch needs its own context RoPE (cc/cs) and cap-pad
mask (the image branch is shared, same latent); and the latent must be denormalized
before the VAE (latents / 0.3611 + 0.1159).
