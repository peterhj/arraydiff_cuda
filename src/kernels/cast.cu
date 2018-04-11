#include "arraydiff/kernels.hh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace arraydiff {

template <typename From, typename To>
__global__ void array_cast_kernel(
    uint32_t size,
    From* x,
    To* y);

template <>
__global__ void array_cast_kernel<uint8_t, float>(
    uint32_t size,
    uint8_t* x,
    float* y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < size) {
    y[idx] = static_cast<float>(x[idx]);
  }
}

template <>
void array_cast<uint8_t, float>(
    uint32_t size,
    uint8_t* x,
    float* y,
    cudaStream_t stream)
{
  array_cast_kernel<uint8_t, float><<<(size+1024-1)/1024, 1024, 0, stream>>>(
      size, x, y);
}

} // namespace arraydiff
