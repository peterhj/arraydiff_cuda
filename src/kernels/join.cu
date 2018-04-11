#include "arraydiff/kernels.hh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
  
template <typename T>
__global__ void arraydiff_sum_batch_join_fwd_kernel(
    uint32_t batch_size,
    const T* x,
    T* y)
{
  __shared__ T cache[1024];
  if (threadIdx.x < batch_size) {
    cache[threadIdx.x] = x[threadIdx.x];
  } else {
    cache[threadIdx.x] = static_cast<T>(0);
  }
  __syncthreads();
  for (uint32_t s = 512; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      cache[threadIdx.x] += cache[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    y[0] = cache[0];
  }
}

template <typename T>
__global__ void arraydiff_sum_batch_join_bwd_kernel(
    uint32_t batch_size,
    const T* dy,
    T* dx)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < batch_size) {
    dx[idx] += dy[0];
  }
}

namespace arraydiff {
//namespace kernels {

template <>
void sum_batch_join_fwd<float>(
    uint32_t batch_size,
    const float* x,
    float* y,
    cudaStream_t stream)
{
  assert(batch_size <= 1024U);
  arraydiff_sum_batch_join_fwd_kernel<<<1, 1024, 0, stream>>>(
      batch_size, x, y);
}

template <>
void sum_batch_join_bwd<float>(
    uint32_t batch_size,
    const float* dy,
    float* dx,
    cudaStream_t stream)
{
  arraydiff_sum_batch_join_bwd_kernel<<<(batch_size+1024-1)/1024, 1024, 0, stream>>>(
      batch_size, dy, dx);
}

//} // namespace kernels
} // namespace arraydiff
