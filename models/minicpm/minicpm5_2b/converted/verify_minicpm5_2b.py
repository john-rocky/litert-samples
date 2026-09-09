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

"""Quality gate, metadata and thinking-mode checks for a converted MiniCPM5-2B bundle.

Three modes:

  gate (default) — asks 8 fixed, unambiguously checkable questions through
    the `litert-lm` CLI (pip install litert-lm), greedy, one fresh session per
    question, and scores correctness plus degeneracy (looping, token spam,
    empty output, an unclosed thought). MiniCPM5 thinks by default, so only
    the text AFTER the thought channel is scored: the reasoning routinely
    contains the expected string while it weighs alternatives, which would
    fake a pass. Run it on BOTH backends — this model's two failure modes
    each show on one backend only (see the README).

      python verify_minicpm5_2b.py model.litertlm --backend cpu
      python verify_minicpm5_2b.py model.litertlm --backend gpu

  --check-metadata — reads the bundle's LlmMetadata and sections (no
    unpacking) and asserts the shape this recipe produces: start_token `<s>`,
    stop ids 1 and 130073, the vendor Jinja template embedded byte-equal to
    the repo's chat_template.jinja, the `thought` channel declared,
    max_num_tokens 4096, an externalized embedder section, and NO zero
    blockwise scale anywhere (the int4 CPU-load killer). With
    --expect-activation fp32|fp16 it also asserts the activation dtype the
    prefill/decode section declares (the int8 GPU fix). Needs `pip install
    litert-lm-builder`.

      python verify_minicpm5_2b.py model.litertlm --check-metadata [--expect-activation fp32]

  --thinking-ab — the single question that decided the int8 activation
    declaration: the rhyme line, thinking left at the model's default, on the
    given backend. Reports whether the thought closed, how long it ran, and
    the answer. On the raw int8 export the GPU leg runs 2000+ tokens and
    never closes; with fp32 declared it closes in ~450 tokens.

      python verify_minicpm5_2b.py model.litertlm --backend gpu --thinking-ab
"""

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

MODEL_ID = "openbmb/MiniCPM5-2B"
STOP_IDS = {1, 130073}  # </s>, <|im_end|> (generation_config eos_token_id)
TEMPLATE_BYTES = 9060   # chat_template.jinja at the 2026-09 revision
TEMPLATE_MD5 = "87da9b132dd68fd3158b392429290a6d"
TOTAL_PARAMS = 2_516_756_480
BPP_BAND = {"int4": (0.55, 0.70), "int8": (0.95, 1.15)}

SUFFIX = " Answer briefly."
QUESTIONS = [
    ("17+25=42", "What is 17 + 25?", r"\b42\b", None),
    ("capital=Tokyo", "What is the capital of Japan?", r"tokyo", None),
    ("opp(hot)=cold", 'What is the opposite of "hot"?', r"\bcold\b", None),
    ("days/week=7", "How many days are in a week?", r"\bseven\b|\b7\b", None),
    ("thanks(fr)=merci", 'How do you say "thank you" in French?', r"merci", None),
    ("8*7=56", "What is 8 times 7?", r"\b56\b", None),
    ("0.9>0.11", "Which is larger: 0.9 or 0.11?", r"0\.9\b",
     r"0\.11\s*(is|>)\s*(the\s+)?(larg|great|bigg)"),
    ("rhyme=blue", 'Complete the rhyme: "Roses are red, violets are ___"',
     r"\bblue\b", None),
]
RHYME = QUESTIONS[-1]

# The runtime prints the thought channel inline; the CLI and the release-kit
# binaries use different markers for it.
THOUGHT_OPEN = ("<think>", "[thought]")
THOUGHT_CLOSE = ("</think>", "[/thought]")


def split_thought(text):
  """Returns (thought, answer, closed).

  answer is the text after the LAST close marker; if an open marker appears
  without a close, the thought never terminated and there is no answer.
  """
  opened = any(m in text for m in THOUGHT_OPEN)
  closes = [text.rfind(m) + len(m) for m in THOUGHT_CLOSE if m in text]
  if closes:
    cut = max(closes)
    return text[:cut], text[cut:].strip(), True
  if opened:
    return text, "", False
  return "", text.strip(), True


def degenerate(text):
  """True if the output loops or is special-token spam."""
  words = text.split()
  if len(words) >= 10:
    grams = [" ".join(words[i:i + 5]) for i in range(len(words) - 4)]
    top = Counter(grams).most_common(1)[0][1] if grams else 0
    diversity = len(set(words)) / len(words)
    if top >= 3 and (top >= 5 or diversity < 0.50):
      return True
    if diversity < 0.30:
      return True
  if len(text) >= 40 and len(set(text)) < 15:
    return True
  if text.count("<|") >= 5 or text.count("<pad>") >= 5:
    return True
  return False


def run_one(args, prompt, max_tokens=None):
  """Runs one prompt through `litert-lm run`, greedy; returns stdout."""
  cmd = [args.litert_lm, "run", str(args.model), "--backend", args.backend,
         "--temperature", "0", "--top-k", "1", "--cache", "no"]
  if max_tokens:
    cmd += ["--max-num-tokens", str(max_tokens)]
  cmd += ["--prompt", prompt]
  proc = subprocess.run(cmd, capture_output=True, text=True,
                        timeout=args.timeout)
  if proc.returncode != 0:
    raise RuntimeError(
        f"litert-lm run failed (exit={proc.returncode}).\n"
        f"--- stderr tail ---\n{proc.stderr[-1500:]}")
  return proc.stdout


def gate(args):
  results = []
  for label, q, pat, veto in QUESTIONS:
    try:
      raw = run_one(args, q + SUFFIX)
    except Exception as e:  # noqa: BLE001
      print(f"FAIL (harness) on '{label}': {e}", file=sys.stderr)
      return 2
    thought, ans, closed = split_thought(raw)
    low = ans.lower()
    ok = closed and bool(re.search(pat, low)) and not (veto and re.search(veto, low))
    degen = (not closed) or (not ans.strip()) or degenerate(ans)
    results.append({"label": label, "question": q, "ok": ok, "degenerate": degen,
                    "thought_closed": closed, "thought_chars": len(thought),
                    "answer": ans})
    shown = " ".join(ans.split())[:80] if ans.strip() else (
        "(thought never closed)" if not closed else "(empty)")
    print(f"  [{'v' if ok else '.'}]{' DEGEN' if degen else '      '} "
          f"{label:16s} thought {len(thought):5d} ch -> {shown!r}")

  score = sum(r["ok"] for r in results)
  any_degen = any(r["degenerate"] for r in results)
  passed = (score >= args.min_correct) and (not any_degen)
  print(f"\n  correct: {score}/8   degenerate: {'YES' if any_degen else 'no'}")
  print(f"  VERDICT: {'PASS' if passed else 'FAIL'} "
        f"(threshold {args.min_correct}/8, non-degenerate, backend {args.backend})")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "backend": args.backend,
        "score": score, "of": 8, "degenerate": any_degen, "passed": passed,
        "questions": results}, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {args.json}")
  return 0 if passed else 1


def thinking_ab(args):
  label, q, pat, _ = RHYME
  raw = run_one(args, q + SUFFIX, max_tokens=4096)
  thought, ans, closed = split_thought(raw)
  ok = closed and bool(re.search(pat, ans.lower()))
  print(f"  backend {args.backend}: thought {'closed' if closed else 'NEVER CLOSED'} "
        f"after {len(thought)} chars; answer {ans[:60]!r}; {'PASS' if ok else 'FAIL'}")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "model": str(args.model), "backend": args.backend, "question": q,
        "thought_closed": closed, "thought_chars": len(thought),
        "answer": ans, "ok": ok}, indent=2, ensure_ascii=False) + "\n")
  return 0 if ok else 1


def read_metadata(model, jinja_path=None):
  """Returns (LlmMetadata pbtext, [sections]) without unpacking the bundle."""
  from litert_lm_builder import litertlm_peek  # pip install litert-lm-builder

  buf = io.StringIO()
  litertlm_peek.peek_litertlm_file(str(model), None, buf,
                                   jinja_prompt_template_path=jinja_path)
  text = buf.getvalue()
  m = re.search(r"start of LlmMetadata\n(.*?)>{4,} end of LlmMetadata", text, re.S)
  pbtext = m.group(1) if m else ""
  sections = []
  for sm in re.finditer(
      r"Section (\d+):\n(.*?)Begin Offset:\s*(\d+)\n\s*End Offset:\s*(\d+)\n"
      r"\s*Data Type:\s*(\S+)", text, re.S):
    items = sm.group(2)
    mt = re.search(r"model_type, Value \(String\): (\S+)", items)
    sections.append({"index": int(sm.group(1)), "type": sm.group(5),
                     "model_type": mt.group(1) if mt else None,
                     "begin": int(sm.group(3)), "end": int(sm.group(4)),
                     "bytes": int(sm.group(4)) - int(sm.group(3))})
  return pbtext, sections


def zero_blockwise_scales(model, sections):
  """Counts zero fp16 scales in every blockwise-quantized tensor (read-only)."""
  import numpy as np
  from ai_edge_litert import schema_py_generated as schema

  data = Path(model).read_bytes()
  total = 0
  tensors = []
  for s in sections:
    if s["type"] != "TFLiteModel":
      continue
    view = data[s["begin"]:s["end"]]
    fb = schema.Model.GetRootAsModel(view, 0)
    seen = set()
    for si in range(fb.SubgraphsLength()):
      sg = fb.Subgraphs(si)
      for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        q = t.Quantization()
        if q is None or q.DetailsType() != schema.QuantizationDetails.BlockwiseQuantization:
          continue
        bq = schema.BlockwiseQuantization()
        tab = q.Details()
        bq.Init(tab.Bytes, tab.Pos)
        st = sg.Tensors(bq.Scales())
        bidx = st.Buffer()
        if bidx in seen:
          continue
        seen.add(bidx)
        buf = fb.Buffers(bidx)
        if buf.DataLength():
          off, size = buf._tab.Vector(buf._tab.Offset(4)), buf.DataLength()
        else:
          off, size = buf.Offset(), buf.Size()
        if not size:
          continue
        f16 = np.frombuffer(view, dtype=np.float16, count=size // 2, offset=off)
        z = int((f16 == 0).sum())
        tensors.append((t.Name().decode(errors="replace"), z, int(f16.size)))
        total += z
  return total, tensors


def declared_activation(model):
  """The prefer_activation_type the prefill/decode section declares, or None.

  The declaration is a section metadata item (`litert-lm unpack` shows it as
  an `additional_metadata` entry in model.toml); the peek output lists it
  next to the section's model_type, so no unpacking is needed.
  """
  from litert_lm_builder import litertlm_peek

  buf = io.StringIO()
  litertlm_peek.peek_litertlm_file(str(model), None, buf)
  for sm in re.finditer(r"Section \d+:\n(.*?)Data Type:\s*TFLiteModel", buf.getvalue(), re.S):
    items = sm.group(1)
    if "tf_lite_prefill_decode" in items:
      m = re.search(r"prefer_activation_type, Value \(String\): (\S+)", items)
      return m.group(1) if m else None
  return None


def check_metadata(args):
  with tempfile.TemporaryDirectory() as td:
    jinja_path = os.path.join(td, "template.jinja")
    pbtext, sections = read_metadata(args.model, jinja_path)
    embedded = Path(jinja_path).read_bytes() if os.path.exists(jinja_path) else b""
  if not pbtext:
    print("FAIL: no LlmMetadata section found")
    return 1
  checks = []

  def check(name, ok, detail=""):
    checks.append(ok)
    print(f"  [{'v' if ok else 'X'}] {name}{(': ' + detail) if detail else ''}")

  st = re.search(r"start_token\s*\{.*?token_str:\s*\"([^\"]*)\"", pbtext, re.S)
  check("start_token is <s> (template's bos_token renders empty; engine prepends)",
        bool(st) and st.group(1) == "<s>", st.group(1) if st else "missing")

  stop_ids = {int(x) for x in re.findall(r"ids:\s*(\d+)", pbtext)}
  check("stop ids include 1 (</s>) and 130073 (<|im_end|>)",
        STOP_IDS <= stop_ids, f"stop ids {sorted(stop_ids)}")

  md5 = hashlib.md5(embedded).hexdigest()
  detail = f"{len(embedded)} B, md5 {md5[:8]}…"
  try:  # live byte comparison when the Hub is reachable; else the recorded hash
    from huggingface_hub import hf_hub_download
    repo_tpl = Path(hf_hub_download(MODEL_ID, "chat_template.jinja")).read_bytes()
    check("embedded Jinja template is byte-equal to the repo's chat_template.jinja",
          embedded == repo_tpl, detail + f" vs repo {len(repo_tpl)} B")
  except Exception as e:  # noqa: BLE001
    check("embedded Jinja template matches the recorded repo template hash",
          len(embedded) == TEMPLATE_BYTES and md5 == TEMPLATE_MD5,
          detail + f" (Hub unreachable: {type(e).__name__})")

  has_channel = bool(re.search(r"thought", pbtext)) and "<think>" in pbtext
  check("thought channel declared (<think> / </think>)", has_channel)

  ctx = re.search(r"max_num_tokens:\s*(\d+)", pbtext)
  check("max_num_tokens 4096", bool(ctx) and int(ctx.group(1)) == 4096,
        ctx.group(1) if ctx else "missing")

  types = [(s["type"], s["model_type"]) for s in sections]
  check("externalized embedder: tf_lite_prefill_decode + tf_lite_embedder sections",
        ("TFLiteModel", "tf_lite_prefill_decode") in types
        and ("TFLiteModel", "tf_lite_embedder") in types, str(types))

  total = os.path.getsize(args.model)
  bpp = total / TOTAL_PARAMS
  band = BPP_BAND.get(args.recipe)
  if band:
    check(f"bytes per parameter in the {args.recipe} band {band}",
          band[0] <= bpp <= band[1], f"{total:,} B / {TOTAL_PARAMS:,} = {bpp:.2f}")
  else:
    print(f"  (i) bytes per parameter {bpp:.2f} (pass --recipe int4|int8 to assert)")

  zeros, tensors = zero_blockwise_scales(args.model, sections)
  bad = [f"{n}: {z}/{tot}" for n, z, tot in tensors if z]
  check("no zero blockwise scales (the int4 CPU-load killer)", zeros == 0,
        f"{len(tensors)} blockwise tensors scanned"
        + (f"; zeros in {', '.join(bad[:4])}" if bad else ""))

  if args.expect_activation:
    act = declared_activation(args.model)
    # No declaration = the runtime default (fp16 on the GPU backend).
    ok = act == args.expect_activation or (act is None and args.expect_activation == "fp16")
    check(f"activation dtype in effect is {args.expect_activation}", ok,
          f"declared {act!r}" + (" (none = runtime default fp16)" if act is None else ""))

  ok = all(checks)
  print(f"\n  {'PASS' if ok else 'FAIL'}: {sum(checks)}/{len(checks)} metadata checks")
  return 0 if ok else 1


def main():
  ap = argparse.ArgumentParser(
      description="Quality gate / metadata / thinking checks for MiniCPM5-2B")
  ap.add_argument("model", help="path to model.litertlm")
  ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--min-correct", type=int, default=6)
  ap.add_argument("--timeout", type=int, default=1800,
                  help="seconds per prompt (engine init is paid every run; a "
                       "thinking model can run to its budget)")
  ap.add_argument("--litert-lm", default="litert-lm",
                  help="path to the litert-lm CLI (pip install litert-lm)")
  ap.add_argument("--recipe", choices=["int4", "int8"],
                  help="--check-metadata: assert the bytes-per-parameter band")
  ap.add_argument("--expect-activation", choices=["fp32", "fp16"],
                  help="--check-metadata: assert model.toml's prefer_activation_type")
  ap.add_argument("--json", help="write a JSON report here")
  ap.add_argument("--check-metadata", action="store_true")
  ap.add_argument("--thinking-ab", action="store_true")
  args = ap.parse_args()

  if not Path(args.model).exists():
    print(f"ERROR: model not found: {args.model}", file=sys.stderr)
    return 2
  if args.check_metadata:
    return check_metadata(args)
  if args.thinking_ab:
    return thinking_ab(args)
  return gate(args)


if __name__ == "__main__":
  sys.exit(main())
