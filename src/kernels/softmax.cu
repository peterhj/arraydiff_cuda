#include "arraydiff/kernels.hh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

extern "C" {
#include <stdio.h>
}
  
namespace arraydiff {
//namespace kernels {

template <typename T>
__global__ void softmax_fwd_kernel(
    uint32_t num_channels,
    uint32_t batch_size,
    const T* x,
    T* y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t j = idx % num_channels;
  uint32_t batch_idx = idx / num_channels;
  if (j < num_channels && batch_idx < batch_size) {
    float logit_i = x[idx];
    if (j < 4) {
      printf("CUDA KERNEL DEBUG: i: %d index: %d chan: %d logit: %.6f\n",
          batch_idx, idx, j, logit_i);
    }
    // TODO
  }
}

template <typename T>
__global__ void softmax_nll_loss_fwd_kernel(
    uint32_t num_channels,
    uint32_t batch_size,
    const T* y,
    const uint32_t* label,
    T* loss)
{
  uint32_t batch_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (batch_idx < batch_size) {
    uint32_t label_i = label[batch_idx];
    uint32_t idx = label_i + num_channels * batch_idx;
    float prob_i = y[idx];
    float loss_i = -logf(prob_i);
    /*printf("CUDA KERNEL DEBUG: i: %d label: %d index: %d prob: %.6f loss: %.6f\n",
        batch_idx, label_i, idx, prob_i, loss_i);*/
    loss[batch_idx] = loss_i;
  }
}

template <typename T>
__global__ void softmax_nll_loss_bwd_kernel(
    uint32_t num_channels,
    uint32_t batch_size,
    const T* y,
    const uint32_t* label,
    const T* dloss,
    T* dx)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t channel = idx % num_channels;
  uint32_t batch_idx = idx / num_channels;
  if (channel < num_channels && batch_idx < batch_size) {
    float y_i = y[idx];
    float label_i = label[batch_idx];
    float dloss_i = dloss[batch_idx];
    if (static_cast<uint32_t>(label_i) == channel) {
      dx[idx] += dloss_i * (y_i - static_cast<T>(1.0));
    } else {
      dx[idx] += dloss_i * y_i;
    }
  }
}

template <>
void softmax_fwd<float>(uint32_t num_channels, uint32_t batch_size, const float* x, float* y, cudaStream_t stream) {
  softmax_fwd_kernel<<<(num_channels*batch_size+1024-1)/1024, 1024, 0, stream>>>(
      num_channels, batch_size, x, y);
}

template <>
void softmax_nll_loss_fwd<float>(uint32_t num_channels, uint32_t batch_size, const float* y, const uint32_t* label, float* loss, cudaStream_t stream) {
  softmax_nll_loss_fwd_kernel<<<(batch_size+1024-1)/1024, 1024, 0, stream>>>(
      num_channels, batch_size, y, label, loss);
}

template <>
void softmax_nll_loss_bwd<float>(uint32_t num_channels, uint32_t batch_size, const float* y, const uint32_t* label, const float* dloss, float* dx, cudaStream_t stream) {
  softmax_nll_loss_bwd_kernel<<<(num_channels*batch_size+1024-1)/1024, 1024, 0, stream>>>(
      num_channels, batch_size, y, label, dloss, dx);
}

//} // namespace kernels
} // namespace arraydiff
