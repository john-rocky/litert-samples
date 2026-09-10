# Model recipes

Each directory under `models/<family>/<model>/` holds the scripts that convert one model to LiteRT or LiteRT-LM, the checks that verify the result, and a README that records the toolchain, the steps and the results. The published weights each recipe produces are linked from the table below.

## Using the models

- **`.litertlm` bundles (LLMs)** run through the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) engine. From a terminal: `pip install litert-lm`, then `litert-lm run model.litertlm --prompt "..."`. In an app: [PhotoTalk](../samples/litert/phototalk_sample_app/) and the [Qualcomm LLM chatbot](../samples/litert/qualcomm/llm_chatbot_npu/) under `samples/litert/`, and the samples under [`samples/litert_lm/`](../samples/litert_lm/).
- **`.tflite` graphs (vision, audio, diffusion)** run through the LiteRT CompiledModel API; the apps under [`samples/litert/`](../samples/litert/) show the host code in Kotlin, Swift, Python, C++ and others. Two recipes here have their own app: [image generation](../samples/litert/image_generation/) for Bonsai Image 4B and [text to speech](../samples/litert/text_to_speech_lm/) for Qwen3-TTS.
- **Tensor API graphs** are authored directly in C++ with the LiteRT Tensor API, with no converter. Each `tensor_api/` directory below is a complete program, and [`samples/tensor_api_playground/`](../samples/tensor_api_playground/) runs the API in the browser.
- [LiteRT-CLI](https://github.com/google-ai-edge/LiteRT-CLI) (`litert`) downloads, converts, quantizes, runs and benchmarks models from one command.

## Where to find examples

Runnable apps live under [`samples/`](../samples/): [`samples/litert/`](../samples/litert/) (CompiledModel API), [`samples/litert_interpreter/`](../samples/litert_interpreter/) (Interpreter API), [`samples/litert_lm/`](../samples/litert_lm/) (LiteRT-LM), [`samples/end_to_end/`](../samples/end_to_end/) and [`samples/web_demos/`](../samples/web_demos/). The interactive demos are at [google-ai-edge.github.io/litert-samples](https://google-ai-edge.github.io/litert-samples/).

## How to convert

[conversion.md](conversion.md) is the step-by-step guide from a Hugging Face checkpoint to a verified `.litertlm` bundle or `.tflite` graph: architecture check, export, quantization, gates against the source model, gates on the target backend and device, publishing. The agent skills under [`skills/`](../skills/) carry the same procedure for coding agents.

## Model list

| Recipe | What it is | Source model | Converted weights |
|---|---|---|---|
| [`minicpm/minicpm5_2b/`](minicpm/minicpm5_2b/) | MiniCPM5-2B (2.5B dense, hybrid thinking) to `.litertlm`; runs on both the CPU and GPU backends | [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | [litert-community/MiniCPM5-2B](https://huggingface.co/litert-community/MiniCPM5-2B) |
| [`bonsai/bonsai_image_4b/`](bonsai/bonsai_image_4b/) | Text-to-image diffusion: three `.tflite` graphs with a Python host loop | [prism-ml/bonsai-image-ternary-4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B) | [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B) |
| [`qwen/qwen3_tts/`](qwen/qwen3_tts/) | Qwen3-TTS: three `.tflite` graphs plus host tables; a Tensor API implementation alongside | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | [litert-community/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base) |
| [`sam3/sam3_image/converted/`](sam3/sam3_image/converted/) | SAM 3 text-prompted detection and segmentation: three GPU-resident `.tflite` graphs | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | [mlboydaisuke/SAM3-LiteRT](https://huggingface.co/mlboydaisuke/SAM3-LiteRT) |
| [`wav2vec2/wav2vec2_kws/`](wav2vec2/wav2vec2_kws/) | wav2vec2 keyword spotting: two `.tflite` graphs | [superb/wav2vec2-base-superb-ks](https://huggingface.co/superb/wav2vec2-base-superb-ks) | [litert-community/wav2vec2-keyword-spotting](https://huggingface.co/litert-community/wav2vec2-keyword-spotting) |
| [`zipformer/zipformer_ctc/`](zipformer/zipformer_ctc/) | Zipformer-medium CR-CTC speech recognition: one `.tflite` graph | [Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018](https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018) | [litert-community/Zipformer-medium-CR-CTC-LiteRT](https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT) |
| [`sam2/sam2_hiera_tiny_video/tensor_api/`](sam2/sam2_hiera_tiny_video/tensor_api/) | SAM 2.1 Hiera-Tiny video tracking, authored with the Tensor API (C++, no converter) | | |
| [`llada/llada_8b/tensor_api/`](llada/llada_8b/tensor_api/) | LLaDA-8B diffusion-LM denoise step, authored with the Tensor API | | |
| [`gemma/gemma3/`](gemma/gemma3/), [`gemma/gemma4/`](gemma/gemma4/) | Reserved for the Gemma recipes | | |
