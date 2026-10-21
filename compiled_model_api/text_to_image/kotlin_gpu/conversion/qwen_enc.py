"""Qwen3-4B text encoder → LiteRT INTEGER int8 deploy graph.

Encoder use: inputs_embeds [1,L,2560] (host does embed_tokens gather) -> causal
Qwen3 stack -> last_hidden_state [1,L,2560]. Fixed L. Reuses the shipped Qwen3
recipe class. Validate on reduced layers, then full 36.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def tfl_run(path, *inputs):
    """Runs a LiteRT CompiledModel on numpy inputs.

    Args:
      path: Path to the exported .tflite CompiledModel.
      *inputs: One numpy array per signature input, in signature order.

    Returns:
      A list of output numpy arrays.
    """
    from ai_edge_litert.compiled_model import CompiledModel

    model = CompiledModel.from_file(path)
    sigs = model.get_signature_list()
    key = list(sigs)[0]
    in_details = model.get_input_tensor_details(key)
    out_details = model.get_output_tensor_details(key)
    in_buffers = model.create_input_buffers(0)
    out_buffers = model.create_output_buffers(0)
    for name, buffer, x in zip(sigs[key]["inputs"], in_buffers, inputs):
        dtype = np.dtype(in_details[name]["dtype"])
        buffer.write(np.ascontiguousarray(x, dtype))
    model.run_by_index(0, in_buffers, out_buffers)
    outputs = []
    for name, buffer in zip(sigs[key]["outputs"], out_buffers):
        detail = out_details[name]
        count = int(np.prod(detail["shape"]))
        flat = buffer.read(count, np.dtype(detail["dtype"]))
        outputs.append(flat.reshape(detail["shape"]))
    return outputs

L = 64  # fixed prompt length


def build(n_layers, real=False):
    """Builds the Qwen3 text-encoder module for export.

    Args:
      n_layers: Number of transformer layers to keep.
      real: If True, load pretrained weights, else random-init.

    Returns:
      The encoder module in eval mode.
    """
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
    if real:
        m = Qwen3Model.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo", subfolder="text_encoder",
            torch_dtype=torch.float32, attn_implementation="eager").eval()
        if n_layers is not None:
            m.layers = m.layers[:n_layers]
        return m
    cfg = Qwen3Config(
        hidden_size=2560, intermediate_size=9728, num_hidden_layers=n_layers,
        num_attention_heads=32, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, rope_theta=1000000.0, rms_norm_eps=1e-6,
        max_position_embeddings=40960, attn_implementation="eager")
    return Qwen3Model(cfg).eval()


class Enc(nn.Module):
    def __init__(self, qwen):
        super().__init__()
        self.q = qwen
        # constant causal mask [1,1,L,L], 0/-inf-ish (use large negative)
        m = torch.full((L, L), float("-1e9")).triu(1)
        self.register_buffer("mask", m[None, None])
        self.register_buffer("pos", torch.arange(L)[None])

    def forward(self, inputs_embeds):
        # Z-Image pipeline conditions on the PENULTIMATE hidden state
        # (hidden_states[-2]), not last_hidden_state (CLIP-style -2 trick).
        out = self.q(inputs_embeds=inputs_embeds, attention_mask=self.mask,
                     position_ids=self.pos, use_cache=False,
                     output_hidden_states=True)
        return out.hidden_states[-2]


def main():
    """Builds the graphs / reference and reports the result."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--real", action="store_true")
    args = ap.parse_args()
    import litert_torch
    from litert_torch.generative.quantize import quant_recipes as qrs
    from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity

    layers = None if (args.real and args.layers < 0) else args.layers
    m = Enc(build(layers, real=args.real)).eval()
    emb = torch.randn(1, L, 2560)
    with torch.no_grad():
        ref = m(emb).numpy()
    print(f"[forward] {args.layers} layers, out {ref.shape}")

    cfg = qrs.full_dynamic_recipe(
        weight_dtype=Dtype.INT8, granularity=Granularity.CHANNELWISE
    )
    out = f"qwen_enc_L{args.layers}.tflite"
    litert_torch.convert(m, (emb,), quant_config=cfg).export(out)
    import os
    print(f"[int8] {out}  {os.path.getsize(out)/1e6:.0f} MB")
    q = tfl_run(out, emb.numpy())[0]
    c = np.corrcoef(ref.flatten(), q.flatten())[0, 1]
    print(f"[parity] int8 vs fp32: corr {c:.5f}")


if __name__ == "__main__":
    main()
