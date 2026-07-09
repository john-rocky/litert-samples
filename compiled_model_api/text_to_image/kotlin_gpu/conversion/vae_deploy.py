"""Z-Image VAE (AutoencoderKL) decoder → LiteRT INTEGER int8 deploy graph.

Fixed 256px (latent [1,16,32,32] -> image [1,3,256,
    256]). 3D/4D-aware ManualGroupNorm
(mid-attn group_norm on 3D). Runs ONCE per image, so int8 error does not
compound. Verifies decode corr vs the fp32 diffusers VAE on a real latent.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from litert_gpu_toolkit.patches import (
    patch_gelu,
    patch_swish,
    patch_interpolate,
)
from probe_vae import ManualGroupNormND, patch_groupnorm_nd


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

LAT = 32  # 256px


def main():
    """Builds the graphs / reference and reports the result."""
    from diffusers import AutoencoderKL
    import litert_torch
    from litert_torch.generative.quantize import quant_recipes as qrs
    from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity

    vae = AutoencoderKL.from_pretrained("Tongyi-MAI/Z-Image-Turbo",
        subfolder="vae").eval()

    class Dec(nn.Module):
        def __init__(self):
            super().__init__()
            self.vae = vae

        def forward(self, z):
            return self.vae.decode(z, return_dict=False)[0]

    m = Dec().eval()
    z = torch.randn(1, 16, LAT, LAT)
    patch_gelu(m)
    patch_swish(m)
    patch_interpolate()
    patch_groupnorm_nd(m)
    with torch.no_grad():
        ref = m(z).numpy()
    print(f"[forward] latent {tuple(z.shape)} -> {ref.shape}")

    cfg = qrs.full_dynamic_recipe(
        weight_dtype=Dtype.INT8, granularity=Granularity.CHANNELWISE
    )
    out = "zvae_int8_256.tflite"
    litert_torch.convert(m, (z,), quant_config=cfg).export(out)
    import os
    print(f"[int8] {out}  {os.path.getsize(out)/1e6:.0f} MB")

    q = tfl_run(out, z.numpy())[0]
    corr = np.corrcoef(ref.flatten(), q.flatten())[0, 1]
    print(f"[decode] int8 vs fp32: corr {corr:.5f}, "
          f"mean|diff| {np.abs(ref - q).mean():.4e}")


if __name__ == "__main__":
    main()
