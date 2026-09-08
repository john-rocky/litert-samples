// Real-checkpoint weight loading. See sam2_weights.h.

#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_weights.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/examples/gemma3/safetensor_loader.h"
#include "tensor/examples/gemma3/safetensors.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_config.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_graph.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::sam2 {

namespace {

// Reads one stored tensor as host fp32, straight from the mapped file: F32
// is copied as-is, F16 / BF16 are widened. No other dtype appears in the SAM2
// exports.
absl::StatusOr<std::vector<float>> ReadAsFp32(const SafetensorTensorInfo& info) {
  const TensorStorageInfo* storage = info.storage.get();
  if (storage == nullptr || storage->data_base == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat(info.name, ": safetensor storage is invalid"));
  }
  if (info.data_end < info.data_start || info.data_end > storage->data_size) {
    return absl::DataLossError(
        absl::StrCat(info.name, ": tensor data out of range"));
  }
  const std::byte* src = storage->data_base + info.data_start;
  const size_t bytes = info.data_end - info.data_start;
  size_t count = 1;
  for (int64_t d : info.shape) count *= static_cast<size_t>(d);

  std::vector<float> out(count);
  switch (info.dtype) {
    case safetensors::dtype::kFLOAT32:
      if (bytes != count * sizeof(float)) {
        return absl::DataLossError(
            absl::StrCat(info.name, ": F32 byte size mismatch"));
      }
      std::memcpy(out.data(), src, bytes);
      break;
    case safetensors::dtype::kFLOAT16:
    case safetensors::dtype::kBFLOAT16: {
      if (bytes != count * sizeof(uint16_t)) {
        return absl::DataLossError(
            absl::StrCat(info.name, ": 16-bit byte size mismatch"));
      }
      std::vector<uint16_t> raw(count);  // memcpy: offsets need not align
      std::memcpy(raw.data(), src, bytes);
      const bool bf16 = info.dtype == safetensors::dtype::kBFLOAT16;
      for (size_t i = 0; i < count; ++i) {
        out[i] = bf16 ? safetensors::bfloat16_to_float(raw[i])
                      : safetensors::fp16_to_float(raw[i]);
      }
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          info.name, ": dtype ", safetensors::get_dtype_str(info.dtype),
          " is not a float weight"));
  }
  return out;
}

}  // namespace

absl::Status LoadWeightSpecs(const std::string& path,
                             const std::vector<WeightSpec>& specs,
                             WeightMap& weights) {
  auto loader_or = SafetensorLoader::Load(path);
  if (!loader_or.ok()) return loader_or.status();
  auto loader = std::move(*loader_or);

  for (const WeightSpec& spec : specs) {
    auto info_or = loader.GetTensorInfo(spec.name);
    if (!info_or.ok()) {
      return absl::NotFoundError(absl::StrCat("checkpoint missing ", spec.name,
                                              ": ", info_or.status().message()));
    }
    std::vector<int> loaded(info_or->shape.begin(), info_or->shape.end());
    if (loaded != spec.shape) {
      return absl::FailedPreconditionError(absl::StrCat(
          spec.name, ": checkpoint shape [", absl::StrJoin(loaded, ","),
          "] != expected [", absl::StrJoin(spec.shape, ","), "]"));
    }
    auto values_or = ReadAsFp32(*info_or);
    if (!values_or.ok()) return values_or.status();
    weights[spec.name] =
        TfTensor({.name = spec.name,
                  .type = Type::kFP32,
                  .shape = spec.shape,
                  .buffer = OwningCpuBuffer::Copy<Type::kFP32>(*values_or)});
  }
  return absl::OkStatus();
}

absl::StatusOr<WeightMap> LoadCheckpointWeights(const Sam2Config& config,
                                                const std::string& path) {
  WeightMap weights;
  auto status = LoadWeightSpecs(path, GetWeightSpecs(config), weights);
  if (!status.ok()) return status;
  return weights;
}

}  // namespace litert::tensor::examples::sam2
