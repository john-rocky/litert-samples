# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""openbmb/MiniCPM5-2B (dense LlamaForCausalLM, hybrid thinking) -> .litertlm.

The architecture needs nothing special from litert-torch: a 42-layer Llama
with GQA 16:2, untied 130,560-token embeddings and rope_theta 5e6 exports on
the stock HF path. What this recipe adds is the packaging and the two
post-export steps that make the bundle run on BOTH LiteRT-LM backends:

  1. Template packaging. The checkpoint's chat_template.jinja is embedded
     verbatim (use_jinja_template=True) — it carries the `enable_thinking`
     switch and the tool-calling format, and litert-torch declares the
     `thought` channel (<think>/</think>) from it automatically. The
     structured-prefix path (use_jinja_template=False) cannot express the
     thinking switch. start_token stays `<s>`: the template's own
     `{{ bos_token }}` renders empty at runtime and the engine prepends the
     metadata start token, so the model sees exactly one `<s>`, as upstream.

  2. int4 (blockwise-32 + OCTAV): decoder layer 0's MLP has 13 all-zero rows
     (gate_proj and up_proj, the same rows). Blockwise quantization emits a
     zero scale for each of their blocks, and the CPU backend (XNNPACK)
     refuses to load the tensor — `unsupported scale value (0.000000) ... for
     INT4 tensor` — while the GPU backend accepts the same file silently.
     fix_zero_scales_inplace.py replaces those scales by a positive epsilon
     in place; the quantized values in a zero block are 0, so the
     dequantized weights do not change (3,328 bytes of a 1.55 GB file).

  3. int8 (dynamic per-channel): no zero scales, but on the GPU backend the
     runtime's default fp16 activations let one of the gate questions reason
     for 2000+ tokens without closing </think> (empty answer, gate 7/8).
     set_activation_type.py declares `prefer_activation_type = "fp32"` in the
     bundle's model.toml (a repack; every weight byte unchanged): the question
     closes in ~450 tokens and the gate passes 8/8, at ~14 % GPU decode
     speed. The int4 file keeps the fp16 default — it passes the gate there,
     and declaring fp32 on int4 only reproduces the CPU reference's
     non-terminating thinking chains (see the README).

Environment (the released stack the published bundles were built on):

    pip install litert-torch==0.9.3 litert-converter==0.4.0 \
        ai-edge-quantizer==0.9.0 litert-lm-builder==0.16.1 transformers==5.14.1
    pip install litert-lm   # >= 0.16; the int8 step needs its pack/unpack CLI

Usage:

    python build_minicpm5_2b.py --recipe int4 --out out_int4   # ~20 min on an M4 Max
    python build_minicpm5_2b.py --recipe int8 --out out_int8

Outputs `<out>/MiniCPM5-2B_<recipe>.litertlm` (the raw export stays in
`<out>/raw/` for the byte-level comparison the verify script does).
"""

import argparse
import copy
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "openbmb/MiniCPM5-2B"
PREFILL_LADDER = "1024,256,64,16,4,1"  # six signatures: tight chunks per prompt
CACHE_LENGTH = 4096
HERE = Path(__file__).resolve().parent


def register_recipes():
  """Adds the blockwise-32 int4 + OCTAV recipe under the name `BOCTAV4`.

  litert-torch resolves --quantization_recipe by name in
  ai_edge_quantizer.recipe, so a module attribute is all it takes. The recipe
  is the stock dynamic_wi4_afp32 with three changes: BLOCKWISE_32 granularity
  (channelwise int4 collapses decoders), OCTAV optimal clipping instead of
  min-max, and the vocab embedding kept at int8 (EMBEDDING_LOOKUP).
  """
  import ai_edge_quantizer.recipe as aqr

  int4 = aqr.dynamic_wi4_afp32()[0]
  int8 = copy.deepcopy(int4)
  int8["op_config"]["weight_tensor_config"]["num_bits"] = 8
  boctav4 = copy.deepcopy(int4)
  boctav4["algorithm_key"] = aqr.AlgorithmName.OCTAV
  boctav4["op_config"]["weight_tensor_config"]["granularity"] = "BLOCKWISE_32"
  embed_int8 = copy.deepcopy(int8)
  embed_int8["operation"] = "EMBEDDING_LOOKUP"
  aqr.BOCTAV4 = lambda: [boctav4, embed_int8]


def pin_template(template_path):
  """Makes every tokenizer loaded during the export carry the given template.

  transformers already loads chat_template.jinja from the checkpoint; pinning
  it explicitly makes the embedded template a checked input of the build (the
  verify script compares the bundle's template byte-for-byte with the repo's).
  """
  import transformers

  template = Path(template_path).read_text(encoding="utf-8")
  orig = transformers.AutoTokenizer.from_pretrained

  def patched(*args, **kwargs):
    tok = orig(*args, **kwargs)
    tok.chat_template = template
    return tok

  transformers.AutoTokenizer.from_pretrained = patched


def fetch_template(out_dir):
  from huggingface_hub import hf_hub_download

  src = hf_hub_download(MODEL_ID, "chat_template.jinja")
  dst = out_dir / "chat_template.jinja"
  shutil.copyfile(src, dst)
  return dst


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--recipe", choices=["int4", "int8"], required=True)
  ap.add_argument("--out", default=None, help="output dir (default out_<recipe>)")
  ap.add_argument("--model", default=MODEL_ID, help="HF id or local snapshot")
  ap.add_argument("--cache", type=int, default=CACHE_LENGTH)
  ap.add_argument("--prefill", default=PREFILL_LADDER)
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="litert-lm CLI, used by the int8 activation repack")
  args = ap.parse_args()

  out = Path(args.out or f"out_{args.recipe}").resolve()
  out.mkdir(parents=True, exist_ok=True)
  template = fetch_template(out)
  pin_template(template)
  register_recipes()

  from litert_torch.generative.export_hf.export import export

  raw = out / "raw"
  quant = {"int4": "BOCTAV4", "int8": "dynamic_wi8_afp32"}[args.recipe]
  print(f"export {args.model} -> {raw}  recipe={quant}  cache={args.cache}  "
        f"prefill={args.prefill}")
  export(
      model=args.model,
      output_dir=str(raw),
      prefill_lengths=[int(x) for x in args.prefill.split(",") if x.strip()],
      cache_length=args.cache,
      quantization_recipe=quant,
      # Verbatim vendor Jinja: keeps enable_thinking + tool calls, and
      # auto-declares the thought channel.
      use_jinja_template=True,
      # The 130560x2048 embedding table in its own section keeps the main
      # weights section small enough for iOS's single-section mmap ceiling.
      externalize_embedder=True,
      trust_remote_code=True,
  )
  raw_bundle = raw / "model.litertlm"
  if not raw_bundle.exists():
    cands = list(raw.glob("*.litertlm"))
    if not cands:
      sys.exit(f"export produced no .litertlm under {raw}")
    raw_bundle = cands[0]

  final = out / f"MiniCPM5-2B_{args.recipe}.litertlm"
  if args.recipe == "int4":
    # Zero-scale fix (CPU backend refuses scale 0; see the module docstring).
    subprocess.run([sys.executable, str(HERE / "fix_zero_scales_inplace.py"),
                    str(raw_bundle), str(final)], check=True)
  else:
    # fp32 activation declaration (GPU backend; see the module docstring).
    subprocess.run([sys.executable, str(HERE / "set_activation_type.py"),
                    str(raw_bundle), str(final), "--type", "fp32",
                    "--litert-lm", args.litert_lm], check=True)
  print(f"BUILD_DONE {final} ({os.path.getsize(final):,} B)")


if __name__ == "__main__":
  main()
