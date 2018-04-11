#include "arraydiff/kernels.hh"

//#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
  
namespace arraydiff {

template <typename T, typename Map>
__global__ void slice2d_map_hw_nchw_packed_kernel(
    uint32_t count_w,
    uint32_t count_h,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t channels,
    uint32_t batch_size,
    const T* src,
    uint32_t src_w_off,
    uint32_t src_h_off,
    T* dst,
    uint32_t dst_w_off,
    uint32_t dst_h_off)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t count = count_w * count_h * channels * batch_size;
  if (idx < count) {
    uint32_t w = idx % count_w;
    uint32_t h = (idx / count_w) % count_h;
    uint32_t c = ((idx / count_w) / count_h) % channels;
    uint32_t n = ((idx / count_w) / count_h) / channels;
    uint32_t src_pidx = (src_w_off + w) + src_width * ((src_h_off + h) + src_height * (c + channels * n));
    uint32_t dst_pidx = (dst_w_off + w) + dst_width * ((dst_h_off + h) + dst_height * (c + channels * n));
    Map::template Map<T>(dst + dst_pidx, src + src_pidx);
  }
  __threadfence();
}

template <>
void slice2d_copy_hw_nchw_packed(
    uint32_t count_w,
    uint32_t count_h,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t channels,
    uint32_t batch_size,
    const float* src,
    uint32_t src_w_off,
    uint32_t src_h_off,
    float* dst,
    uint32_t dst_w_off,
    uint32_t dst_h_off,
    cudaStream_t stream)
{
  assert(src_w_off + count_w <= src_width);
  assert(src_h_off + count_h <= src_height);
  assert(dst_w_off + count_w <= dst_width);
  assert(dst_h_off + count_h <= dst_height);
  uint32_t count = count_w * count_h * channels * batch_size;
  slice2d_map_hw_nchw_packed_kernel<float, CopyMap><<<(count+1024-1)/1024, 1024, 0, stream>>>(
      count_w, count_h,
      src_width, src_height,
      dst_width, dst_height,
      channels, batch_size,
      src, src_w_off, src_h_off,
      dst, dst_w_off, dst_h_off);
}

template <>
void slice2d_add_hw_nchw_packed(
    uint32_t count_w,
    uint32_t count_h,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t dst_width,
    uint32_t dst_height,
    uint32_t channels,
    uint32_t batch_size,
    const float* src,
    uint32_t src_w_off,
    uint32_t src_h_off,
    float* dst,
    uint32_t dst_w_off,
    uint32_t dst_h_off,
    cudaStream_t stream)
{
  assert(src_w_off + count_w <= src_width);
  assert(src_h_off + count_h <= src_height);
  assert(dst_w_off + count_w <= dst_width);
  assert(dst_h_off + count_h <= dst_height);
  uint32_t count = count_w * count_h * channels * batch_size;
  slice2d_map_hw_nchw_packed_kernel<float, ReduceMap><<<(count+1024-1)/1024, 1024, 0, stream>>>(
      count_w, count_h,
      src_width, src_height,
      dst_width, dst_height,
      channels, batch_size,
      src, src_w_off, src_h_off,
      dst, dst_w_off, dst_h_off);
}

} // namespace arraydiff
