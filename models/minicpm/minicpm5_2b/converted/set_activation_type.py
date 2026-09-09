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

"""Declare an activation dtype for the TFLiteModel section of a .litertlm.

The GPU executor runs activations in fp16 by default. On MiniCPM5-2B int8 that
is enough to keep one gate question reasoning for 2000+ tokens without ever
closing </think> (empty answer); on hybrid SSM models the same default makes
the sampler report "Invalid decode and sample result". Declaring

    prefer_activation_type = "fp32"

in the bundle's `model.toml` makes the executor keep activations in fp32.
Weights are untouched, so this is a repack, not a re-export: every section's
bytes stay identical, only model.toml gains the line.

    python set_activation_type.py in.litertlm out.litertlm [--type fp32]

Needs the `litert-lm` CLI (>= 0.15, for `unpack`/`pack`) on PATH.
"""
import argparse
import os
import re
import shutil
import subprocess
import tempfile


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("src")
  ap.add_argument("dst")
  ap.add_argument("--type", default="fp32", help="fp32 or fp16 (default fp32)")
  ap.add_argument("--litert-lm", default="litert-lm")
  args = ap.parse_args()

  with tempfile.TemporaryDirectory() as td:
    unpack = os.path.join(td, "unpack")
    subprocess.run(
        [args.litert_lm, "unpack", args.src, "--output-dir", unpack], check=True
    )
    toml_path = os.path.join(unpack, "model.toml")
    toml = open(toml_path).read()

    marker = 'section_type = "TFLiteModel"'
    if marker not in toml:
      raise SystemExit("model.toml has no TFLiteModel section")
    if "prefer_activation_type" in toml:
      toml = re.sub(
          r'prefer_activation_type = "[^"]*"',
          f'prefer_activation_type = "{args.type}"',
          toml,
      )
    else:
      toml = toml.replace(
          marker, marker + f'\nprefer_activation_type = "{args.type}"'
      )
    open(toml_path, "w").write(toml)

    # `litert-lm pack` exits 0 without writing when the output already exists.
    if os.path.exists(args.dst):
      os.remove(args.dst)
    subprocess.run([args.litert_lm, "pack", toml_path, "--output", args.dst],
                   check=True)
  print(f"OK: {args.dst} (prefer_activation_type = {args.type})")


if __name__ == "__main__":
  main()
