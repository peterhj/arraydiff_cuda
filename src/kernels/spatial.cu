#include "arraydiff/kernels.hh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
  
//#define NUM_BLOCKS 64
#define NUM_BLOCKS 80
//#define BLOCK_SIZE 512
#define BLOCK_SIZE 1024
//#define WORK_PER_BLOCK 4096
//#define WORK_PER_BLOCK 8192
#define WORK_PER_BLOCK 16384
//#define WORK_PER_BLOCK 32768
//#define WORK_PER_BLOCK 65536

namespace arraydiff {

class NCHWLayout {
public:
  __device__ __forceinline__ static int32_t Pack(
      const int32_t& w,
      const uint32_t& pitch_w,
      const int32_t& h,
      const uint32_t& pitch_h,
      const int32_t& c,
      const uint32_t& pitch_c,
      const int32_t& n,
      const uint32_t& pitch_n)
  {
    return w + pitch_w * (h + pitch_h * (c + pitch_c * n));
  }
};

class NHWCLayout {
public:
  __device__ __forceinline__ static int32_t Pack(
      const int32_t& w,
      const uint32_t& pitch_w,
      const int32_t& h,
      const uint32_t& pitch_h,
      const int32_t& c,
      const uint32_t& pitch_c,
      const int32_t& n,
      const uint32_t& pitch_n)
  {
    return c + pitch_c * (w + pitch_w * (h + pitch_h * n));
  }
};

class HWSlice {
public:
  template <typename Layout>
  __device__ __forceinline__ static void Unpack(
      const uint32_t& idx,
      uint32_t& c,
      const uint32_t& num_channels,
      uint32_t& n,
      const uint32_t& batch_size);
};

template <>
__device__ __forceinline__ void HWSlice::Unpack<NCHWLayout>(
      const uint32_t& idx,
      uint32_t& c,
      const uint32_t& num_channels,
      uint32_t& n,
      const uint32_t& batch_size)
{
  c = idx % num_channels;
  n = idx / num_channels;
}

template <>
__device__ __forceinline__ void HWSlice::Unpack<NHWCLayout>(
      const uint32_t& idx,
      uint32_t& c,
      const uint32_t& num_channels,
      uint32_t& n,
      const uint32_t& batch_size)
{
  c = idx % num_channels;
  n = idx / num_channels;
}

class HSlice {
public:
  template <typename Layout>
  __device__ __forceinline__ static void Unpack(
      const uint32_t& idx,
      uint32_t& w,
      const uint32_t& width,
      uint32_t& c,
      const uint32_t& num_channels,
      uint32_t& n,
      const uint32_t& batch_size);
};

template <>
__device__ __forceinline__ void HSlice::Unpack<NCHWLayout>(
      const uint32_t& idx,
      uint32_t& w,
      const uint32_t& width,
      uint32_t& c,
      const uint32_t& num_channels,
      uint32_t& n,
      const uint32_t& batch_size)
{
  w = idx % width;
  c = (idx / width) % num_channels;
  n = (idx / width) / num_channels;
}

template <>
__device__ __forceinline__ void HSlice::Unpack<NHWCLayout>(
      const uint32_t& idx,
      uint32_t& w,
      const uint32_t& width,
      uint32_t& c,
      const uint32_t& num_channels,
      uint32_t& n,
      const uint32_t& batch_size)
{
  c = idx % num_channels;
  w = (idx / num_channels) % width;
  n = (idx / num_channels) / width;
}

class WSlice {
public:
  template <typename Layout>
  __device__ __forceinline__ static void Unpack(
      const uint32_t& idx,
      uint32_t& h,
      const uint32_t& height,
      uint32_t& c,
      const uint32_t& num_channels,
      uint32_t& n,
      const uint32_t& batch_size);
};

template <>
__device__ __forceinline__ void WSlice::Unpack<NCHWLayout>(
      const uint32_t& idx,
      uint32_t& h,
      const uint32_t& height,
      uint32_t& c,
      const uint32_t& num_channels,
      uint32_t& n,
      const uint32_t& batch_size)
{
  h = idx % height;
  c = (idx / height) % num_channels;
  n = (idx / height) / num_channels;
}

template <>
__device__ __forceinline__ void WSlice::Unpack<NHWCLayout>(
      const uint32_t& idx,
      uint32_t& h,
      const uint32_t& height,
      uint32_t& c,
      const uint32_t& num_channels,
      uint32_t& n,
      const uint32_t& batch_size)
{
  c = idx % num_channels;
  h = (idx / num_channels) % height;
  n = (idx / num_channels) / height;
}

class ExchOp {
public:
  __device__ __forceinline__ static int32_t IncNextWidth(int32_t base_width)   { return 1; }
  __device__ __forceinline__ static int32_t SrcNextWidth(int32_t base_width)   { return 0; }
  __device__ __forceinline__ static int32_t DstNextWidth(int32_t base_width)   { return base_width; }
  __device__ __forceinline__ static int32_t IncPrevWidth(int32_t base_width)   { return -1; }
  __device__ __forceinline__ static int32_t SrcPrevWidth(int32_t base_width)   { return base_width - 1; }
  __device__ __forceinline__ static int32_t DstPrevWidth(int32_t base_width)   { return -1; }
  __device__ __forceinline__ static int32_t IncNextHeight(int32_t base_height) { return 1; }
  __device__ __forceinline__ static int32_t SrcNextHeight(int32_t base_height) { return 0; }
  __device__ __forceinline__ static int32_t DstNextHeight(int32_t base_height) { return base_height; }
  __device__ __forceinline__ static int32_t IncPrevHeight(int32_t base_height) { return -1; }
  __device__ __forceinline__ static int32_t SrcPrevHeight(int32_t base_height) { return base_height - 1; }
  __device__ __forceinline__ static int32_t DstPrevHeight(int32_t base_height) { return -1; }
};

class ReduceOp {
public:
  __device__ __forceinline__ static int32_t IncNextWidth(int32_t base_width)   { return -1; }
  __device__ __forceinline__ static int32_t SrcNextWidth(int32_t base_width)   { return -1; }
  __device__ __forceinline__ static int32_t DstNextWidth(int32_t base_width)   { return base_width - 1; }
  __device__ __forceinline__ static int32_t IncPrevWidth(int32_t base_width)   { return 1; }
  __device__ __forceinline__ static int32_t SrcPrevWidth(int32_t base_width)   { return base_width; }
  __device__ __forceinline__ static int32_t DstPrevWidth(int32_t base_width)   { return 0; }
  __device__ __forceinline__ static int32_t IncNextHeight(int32_t base_height) { return -1; }
  __device__ __forceinline__ static int32_t SrcNextHeight(int32_t base_height) { return -1; }
  __device__ __forceinline__ static int32_t DstNextHeight(int32_t base_height) { return base_height - 1; }
  __device__ __forceinline__ static int32_t IncPrevHeight(int32_t base_height) { return 1; }
  __device__ __forceinline__ static int32_t SrcPrevHeight(int32_t base_height) { return base_height; }
  __device__ __forceinline__ static int32_t DstPrevHeight(int32_t base_height) { return 0; }
};

template <typename T, typename Map, typename Layout>
__device__ void spatial2d_map_hw_slice(
    uint32_t idx_off,
    uint32_t idx_count,
    uint32_t width,
    int32_t offset_w,
    uint32_t pitch_width,
    uint32_t height,
    int32_t offset_h,
    uint32_t pitch_height,
    uint32_t num_channels,
    uint32_t batch_size,
    const T* src,
    int32_t src_w_off,
    int32_t src_h_off,
    T* dst,
    int32_t dst_w_off,
    int32_t dst_h_off)
{
  const uint32_t count_rounded_down = idx_count / blockDim.x * blockDim.x;
  const uint32_t idx_limit_rounded_down = idx_off + count_rounded_down;
  const uint32_t idx_limit = idx_off + idx_count;
  uint32_t c = 0;
  uint32_t n = 0;
  uint32_t idx_block = idx_off;
  for ( ; idx_block < idx_limit_rounded_down; idx_block += blockDim.x) {
    uint32_t idx = idx_block + threadIdx.x;
    HWSlice::template Unpack<Layout>(
        idx,
        c, num_channels,
        n, batch_size);
    int32_t src_pidx = Layout::Pack(
        offset_w + src_w_off, pitch_width,
        offset_h + src_h_off, pitch_height,
        c, num_channels,
        n, batch_size);
    int32_t dst_pidx = Layout::Pack(
        offset_w + dst_w_off, pitch_width,
        offset_h + dst_h_off, pitch_height,
        c, num_channels,
        n, batch_size);
    Map::template Map<T>(dst + dst_pidx, src + src_pidx);
  }
  uint32_t idx = idx_block + threadIdx.x;
  if (idx < idx_limit) {
    HWSlice::template Unpack<Layout>(
        idx,
        c, num_channels,
        n, batch_size);
    int32_t src_pidx = Layout::Pack(
        offset_w + src_w_off, pitch_width,
        offset_h + src_h_off, pitch_height,
        c, num_channels,
        n, batch_size);
    int32_t dst_pidx = Layout::Pack(
        offset_w + dst_w_off, pitch_width,
        offset_h + dst_h_off, pitch_height,
        c, num_channels,
        n, batch_size);
    Map::template Map<T>(dst + dst_pidx, src + src_pidx);
  }
  __threadfence();
}

template <typename T, typename Map, typename Layout>
__device__ void spatial2d_map_w_slice(
    uint32_t idx_off,
    uint32_t idx_count,
    uint32_t width,
    int32_t offset_w,
    uint32_t pitch_width,
    uint32_t height,
    int32_t offset_h,
    uint32_t pitch_height,
    uint32_t num_channels,
    uint32_t batch_size,
    const T* src,
    int32_t src_w_off,
    T* dst,
    int32_t dst_w_off)
{
  const uint32_t count_rounded_down = idx_count / blockDim.x * blockDim.x;
  const uint32_t idx_limit_rounded_down = idx_off + count_rounded_down;
  const uint32_t idx_limit = idx_off + idx_count;
  uint32_t h = 0;
  uint32_t c = 0;
  uint32_t n = 0;
  uint32_t idx_block = idx_off;
  for ( ; idx_block < idx_limit_rounded_down; idx_block += blockDim.x) {
    uint32_t idx = idx_block + threadIdx.x;
    WSlice::template Unpack<Layout>(
        idx,
        h, height,
        c, num_channels,
        n, batch_size);
    int32_t src_pidx = Layout::Pack(
        offset_w + src_w_off, pitch_width,
        offset_h + h, pitch_height,
        c, num_channels,
        n, batch_size);
    int32_t dst_pidx = Layout::Pack(
        offset_w + dst_w_off, pitch_width,
        offset_h + h, pitch_height,
        c, num_channels,
        n, batch_size);
    Map::template Map<T>(dst + dst_pidx, src + src_pidx);
  }
  uint32_t idx = idx_block + threadIdx.x;
  if (idx < idx_limit) {
    WSlice::template Unpack<Layout>(
        idx,
        h, height,
        c, num_channels,
        n, batch_size);
    int32_t src_pidx = Layout::Pack(
        offset_w + src_w_off, pitch_width,
        offset_h + h, pitch_height,
        c, num_channels,
        n, batch_size);
    int32_t dst_pidx = Layout::Pack(
        offset_w + dst_w_off, pitch_width,
        offset_h + h, pitch_height,
        c, num_channels,
        n, batch_size);
    Map::template Map<T>(dst + dst_pidx, src + src_pidx);
  }
  __threadfence();
}

template <typename T, typename Map, typename Layout>
__device__ void spatial2d_map_h_slice(
    uint32_t idx_off,
    uint32_t idx_count,
    uint32_t width,
    int32_t offset_w,
    uint32_t pitch_width,
    uint32_t height,
    int32_t offset_h,
    uint32_t pitch_height,
    uint32_t num_channels,
    uint32_t batch_size,
    const T* src,
    int32_t src_h_off,
    T* dst,
    int32_t dst_h_off)
{
  const uint32_t count_rounded_down = idx_count / blockDim.x * blockDim.x;
  const uint32_t idx_limit_rounded_down = idx_off + count_rounded_down;
  const uint32_t idx_limit = idx_off + idx_count;
  uint32_t w = 0;
  uint32_t c = 0;
  uint32_t n = 0;
  uint32_t idx_block = idx_off;
  for ( ; idx_block < idx_limit_rounded_down; idx_block += blockDim.x) {
    uint32_t idx = idx_block + threadIdx.x;
    HSlice::template Unpack<Layout>(
        idx,
        w, width,
        c, num_channels,
        n, batch_size);
    int32_t src_pidx = Layout::Pack(
        offset_w + w, pitch_width,
        offset_h + src_h_off, pitch_height,
        c, num_channels,
        n, batch_size);
    int32_t dst_pidx = Layout::Pack(
        offset_w + w, pitch_width,
        offset_h + dst_h_off, pitch_height,
        c, num_channels,
        n, batch_size);
    Map::template Map<T>(dst + dst_pidx, src + src_pidx);
  }
  uint32_t idx = idx_block + threadIdx.x;
  if (idx < idx_limit) {
    HSlice::template Unpack<Layout>(
        idx,
        w, width,
        c, num_channels,
        n, batch_size);
    int32_t src_pidx = Layout::Pack(
        offset_w + w, pitch_width,
        offset_h + src_h_off, pitch_height,
        c, num_channels,
        n, batch_size);
    int32_t dst_pidx = Layout::Pack(
        offset_w + w, pitch_width,
        offset_h + dst_h_off, pitch_height,
        c, num_channels,
        n, batch_size);
    Map::template Map<T>(dst + dst_pidx, src + src_pidx);
  }
  __threadfence();
}

template <typename T, typename Op, typename Map>
__global__ void spatial2d_1x2_halo_op_nchw_kernel(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_with_halo,
    T* x_1_with_halo)
{
  const uint32_t halo_width = base_width + halo_pad * 2;
  const uint32_t halo_height = base_height + halo_pad * 2;

  int32_t inc_h = 0;
  int32_t src_hoff = 0;
  int32_t dst_hoff = 0;
  const T* x_src = NULL;
  T* y_dst = NULL;
  if (rank == 0) {
    /*inc_h = 1;
    src_hoff = 0;
    dst_hoff = base_height;*/
    inc_h     = Op::IncNextHeight(base_height);
    src_hoff  = Op::SrcNextHeight(base_height);
    dst_hoff  = Op::DstNextHeight(base_height);
    x_src = x_1_with_halo;
    y_dst = x_0_with_halo;
  } else if (rank == 1) {
    /*inc_h = -1;
    src_hoff = base_height - 1;
    dst_hoff = -1;*/
    inc_h     = Op::IncPrevHeight(base_height);
    src_hoff  = Op::SrcPrevHeight(base_height);
    dst_hoff  = Op::DstPrevHeight(base_height);
    x_src = x_0_with_halo;
    y_dst = x_1_with_halo;
  }

  const uint32_t total_count = base_width * num_channels * batch_size;
  const uint32_t block_maxcount = (total_count + gridDim.x - 1) / gridDim.x;

  for (uint32_t delta = 0; delta < halo_pad; ++delta) {
    spatial2d_map_h_slice<T, Map, NCHWLayout>(
        block_maxcount * blockIdx.x, min(block_maxcount, total_count - block_maxcount * blockIdx.x),
        base_width, halo_pad, halo_width,
        base_height, halo_pad, halo_height,
        num_channels,
        batch_size,
        x_src, src_hoff + inc_h * delta,
        y_dst, dst_hoff + inc_h * delta);
  }
}

template <>
void spatial2d_1x2_halo_exch_nchw<float>(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    float* x_0_with_halo,
    float* x_1_with_halo,
    cudaStream_t stream)
{
  uint32_t total_work = base_width * num_channels * batch_size;
  uint32_t num_blocks = max(1, min(NUM_BLOCKS, (total_work + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK));
  /*std::clog << "DEBUG: spatial2d 1x2 halo exch nchw:"
      << " work: " << total_work
      << " nblks: " << num_blocks
      << " x0: " << std::hex << x_0_with_halo << std::dec
      << " x1: " << std::hex << x_1_with_halo << std::dec
      << " stream: " << std::hex << stream << std::dec
      << std::endl;*/
  spatial2d_1x2_halo_op_nchw_kernel<float, ExchOp, CopyMap><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      rank,
      halo_pad,
      base_width,
      base_height,
      num_channels,
      batch_size,
      x_0_with_halo,
      x_1_with_halo);
}

template <>
void spatial2d_1x2_halo_reduce_nchw<float>(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    float* x_0_with_halo,
    float* x_1_with_halo,
    cudaStream_t stream)
{
  uint32_t total_work = base_width * num_channels * batch_size;
  uint32_t num_blocks = max(1, min(NUM_BLOCKS, (total_work + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK));
  spatial2d_1x2_halo_op_nchw_kernel<float, ReduceOp, AtomicReduceMap><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      rank,
      halo_pad,
      base_width,
      base_height,
      num_channels,
      batch_size,
      x_0_with_halo,
      x_1_with_halo);
}

template <typename T, typename Op, typename Map>
__global__ void spatial2d_1x4_halo_op_nchw_kernel(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_with_halo,
    T* x_1_with_halo,
    T* x_2_with_halo,
    T* x_3_with_halo)
{
  const uint32_t num_h_ranks = 4;
  const uint32_t halo_width = base_width + halo_pad * 2;
  const uint32_t halo_height = base_height + halo_pad * 2;

  const T* next_x_src = NULL;
  const T* prev_x_src = NULL;
  T* y_dst = NULL;
  if (rank == 0) {
    next_x_src  = x_1_with_halo;
    prev_x_src  = NULL;
    y_dst       = x_0_with_halo;
  } else if (rank == 1) {
    next_x_src  = x_2_with_halo;
    prev_x_src  = x_0_with_halo;
    y_dst       = x_1_with_halo;
  } else if (rank == 2) {
    next_x_src  = x_3_with_halo;
    prev_x_src  = x_1_with_halo;
    y_dst       = x_2_with_halo;
  } else if (rank == 3) {
    next_x_src  = NULL;
    prev_x_src  = x_2_with_halo;
    y_dst       = x_3_with_halo;
  }

  int32_t next_inc_h = 0;
  int32_t next_src_hoff = 0;
  int32_t next_dst_hoff = 0;
  int32_t prev_inc_h = 0;
  int32_t prev_src_hoff = 0;
  int32_t prev_dst_hoff = 0;
  if (rank < num_h_ranks - 1) {
    next_inc_h    = Op::IncNextHeight(base_height);
    next_src_hoff = Op::SrcNextHeight(base_height);
    next_dst_hoff = Op::DstNextHeight(base_height);
  }
  if (rank > 0) {
    prev_inc_h    = Op::IncPrevHeight(base_height);
    prev_src_hoff = Op::SrcPrevHeight(base_height);
    prev_dst_hoff = Op::DstPrevHeight(base_height);
  }

  const uint32_t total_count = base_width * num_channels * batch_size;
  const uint32_t block_maxcount = (total_count + gridDim.x - 1) / gridDim.x;

  if (rank < num_h_ranks - 1) {
    for (uint32_t delta = 0; delta < halo_pad; ++delta) {
      spatial2d_map_h_slice<T, Map, NCHWLayout>(
          block_maxcount * blockIdx.x, min(block_maxcount, total_count - block_maxcount * blockIdx.x),
          base_width, halo_pad, halo_width,
          base_height, halo_pad, halo_height,
          num_channels,
          batch_size,
          next_x_src, next_src_hoff + next_inc_h * delta,
          y_dst,      next_dst_hoff + next_inc_h * delta);
    }
  }
  if (rank > 0) {
    for (uint32_t delta = 0; delta < halo_pad; ++delta) {
      spatial2d_map_h_slice<T, Map, NCHWLayout>(
          block_maxcount * blockIdx.x, min(block_maxcount, total_count - block_maxcount * blockIdx.x),
          base_width, halo_pad, halo_width,
          base_height, halo_pad, halo_height,
          num_channels,
          batch_size,
          prev_x_src, prev_src_hoff + prev_inc_h * delta,
          y_dst,      prev_dst_hoff + prev_inc_h * delta);
    }
  }
}

template <>
void spatial2d_1x4_halo_exch_nchw<float>(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    float* x_0_with_halo,
    float* x_1_with_halo,
    float* x_2_with_halo,
    float* x_3_with_halo,
    cudaStream_t stream)
{
  uint32_t total_work = base_width * num_channels * batch_size;
  uint32_t num_blocks = max(1, min(NUM_BLOCKS, (total_work + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK));
  spatial2d_1x4_halo_op_nchw_kernel<float, ExchOp, CopyMap><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      rank,
      halo_pad,
      base_width,
      base_height,
      num_channels,
      batch_size,
      x_0_with_halo,
      x_1_with_halo,
      x_2_with_halo,
      x_3_with_halo);
}

template <>
void spatial2d_1x4_halo_reduce_nchw<float>(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    float* x_0_with_halo,
    float* x_1_with_halo,
    float* x_2_with_halo,
    float* x_3_with_halo,
    cudaStream_t stream)
{
  uint32_t total_work = base_width * num_channels * batch_size;
  uint32_t num_blocks = max(1, min(NUM_BLOCKS, (total_work + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK));
  spatial2d_1x4_halo_op_nchw_kernel<float, ReduceOp, AtomicReduceMap><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      rank,
      halo_pad,
      base_width,
      base_height,
      num_channels,
      batch_size,
      x_0_with_halo,
      x_1_with_halo,
      x_2_with_halo,
      x_3_with_halo);
}

template <typename T, typename Op, typename Map>
__global__ void spatial2d_2x1_halo_op_nchw_kernel(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_with_halo,
    T* x_1_with_halo)
{
  const uint32_t halo_width = base_width + halo_pad * 2;
  const uint32_t halo_height = base_height + halo_pad * 2;

  int32_t inc_w = 0;
  int32_t src_woff = 0;
  int32_t dst_woff = 0;
  const T* x_src = NULL;
  T* y_dst = NULL;
  if (rank == 0) {
    /*inc_w = 1;
    src_woff = 0;
    dst_woff = base_width;*/
    inc_w     = Op::IncNextWidth(base_width);
    src_woff  = Op::SrcNextWidth(base_width);
    dst_woff  = Op::DstNextWidth(base_width);
    x_src = x_1_with_halo;
    y_dst = x_0_with_halo;
  } else if (rank == 1) {
    /*inc_w = -1;
    src_woff = base_width - 1;
    dst_woff = -1;*/
    inc_w     = Op::IncPrevWidth(base_width);
    src_woff  = Op::SrcPrevWidth(base_width);
    dst_woff  = Op::DstPrevWidth(base_width);
    x_src = x_0_with_halo;
    y_dst = x_1_with_halo;
  }

  const uint32_t total_count = base_height * num_channels * batch_size;
  const uint32_t block_maxcount = (total_count + gridDim.x - 1) / gridDim.x;

  for (uint32_t delta = 0; delta < halo_pad; ++delta) {
    spatial2d_map_w_slice<T, Map, NCHWLayout>(
        block_maxcount * blockIdx.x, min(block_maxcount, total_count - block_maxcount * blockIdx.x),
        base_width, halo_pad, halo_width,
        base_height, halo_pad, halo_height,
        num_channels,
        batch_size,
        x_src, src_woff + inc_w * delta,
        y_dst, dst_woff + inc_w * delta);
  }
}

template <>
void spatial2d_2x1_halo_exch_nchw<float>(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    float* x_0_with_halo,
    float* x_1_with_halo,
    cudaStream_t stream)
{
  uint32_t total_work = halo_pad * base_height * num_channels * batch_size;
  uint32_t num_blocks = max(1, min(NUM_BLOCKS, (total_work + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK));
  spatial2d_2x1_halo_op_nchw_kernel<float, ExchOp, CopyMap><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      rank,
      halo_pad,
      base_width,
      base_height,
      num_channels,
      batch_size,
      x_0_with_halo,
      x_1_with_halo);
}

/*template <>
void spatial2d_2x1_halo_exch_nchw<float>(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    float* x_0_with_halo,
    float* x_1_with_halo,
    cudaStream_t stream);*/

template <typename T>
void spatial2d_2x1_halo_reduce_nchw(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_with_halo,
    T* x_1_with_halo,
    cudaStream_t stream)
{
  uint32_t total_work = halo_pad * base_height * num_channels * batch_size;
  uint32_t num_blocks = max(1, min(NUM_BLOCKS, (total_work + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK));
  spatial2d_2x1_halo_op_nchw_kernel<T, ReduceOp, AtomicReduceMap><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
      rank,
      halo_pad,
      base_width,
      base_height,
      num_channels,
      batch_size,
      x_0_with_halo,
      x_1_with_halo);
}

template <>
void spatial2d_2x1_halo_reduce_nchw<float>(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    float* x_0_with_halo,
    float* x_1_with_halo,
    cudaStream_t stream);

template <typename T, typename Op, typename Map, typename Layout>
__global__ void spatial2d_2x2_halo_op_kernel(
    uint32_t rank_w,
    uint32_t rank_h,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_0_with_halo,
    T* x_0_1_with_halo,
    T* x_1_0_with_halo,
    T* x_1_1_with_halo)
{
  const uint32_t halo_width = base_width + halo_pad * 2;
  const uint32_t halo_height = base_height + halo_pad * 2;

  int32_t inc_w = 0;
  int32_t src_woff = 0;
  int32_t dst_woff = 0;
  int32_t inc_h = 0;
  int32_t src_hoff = 0;
  int32_t dst_hoff = 0;
  const T* adj_w_src = NULL;
  const T* adj_h_src = NULL;
  const T* diag_src = NULL;
  T* dst = NULL;
  if (rank_w == 0 && rank_h == 0) {
    /*inc_w = 1;
    inc_h = 1;
    src_woff = 0;
    dst_woff = base_width;
    src_hoff = 0;
    dst_hoff = base_height;*/
    inc_w     = Op::IncNextWidth(base_width);
    src_woff  = Op::SrcNextWidth(base_width);
    dst_woff  = Op::DstNextWidth(base_width);
    inc_h     = Op::IncNextHeight(base_height);
    src_hoff  = Op::SrcNextHeight(base_height);
    dst_hoff  = Op::DstNextHeight(base_height);
    adj_w_src = x_0_1_with_halo;
    adj_h_src = x_1_0_with_halo;
    diag_src = x_1_1_with_halo;
    dst = x_0_0_with_halo;
  } else if (rank_w == 0 && rank_h == 1) {
    /*inc_w = 1;
    inc_h = -1;
    src_woff = 0;
    dst_woff = base_width;
    src_hoff = base_height - 1;
    dst_hoff = -1;*/
    inc_w     = Op::IncNextWidth(base_width);
    src_woff  = Op::SrcNextWidth(base_width);
    dst_woff  = Op::DstNextWidth(base_width);
    inc_h     = Op::IncPrevHeight(base_height);
    src_hoff  = Op::SrcPrevHeight(base_height);
    dst_hoff  = Op::DstPrevHeight(base_height);
    adj_w_src = x_0_0_with_halo;
    adj_h_src = x_1_1_with_halo;
    diag_src = x_1_0_with_halo;
    dst = x_0_1_with_halo;
  } else if (rank_w == 1 && rank_h == 0) {
    /*inc_w = -1;
    inc_h = 1;
    src_woff = base_width - 1;
    dst_woff = -1;
    src_hoff = 0;
    dst_hoff = base_height;*/
    inc_w     = Op::IncPrevWidth(base_width);
    src_woff  = Op::SrcPrevWidth(base_width);
    dst_woff  = Op::DstPrevWidth(base_width);
    inc_h     = Op::IncNextHeight(base_height);
    src_hoff  = Op::SrcNextHeight(base_height);
    dst_hoff  = Op::DstNextHeight(base_height);
    adj_w_src = x_1_1_with_halo;
    adj_h_src = x_0_0_with_halo;
    diag_src = x_0_1_with_halo;
    dst = x_1_0_with_halo;
  } else if (rank_w == 1 && rank_h == 1) {
    /*inc_w = -1;
    inc_h = -1;
    src_woff = base_width - 1;
    dst_woff = -1;
    src_hoff = base_height - 1;
    dst_hoff = -1;*/
    inc_w     = Op::IncPrevWidth(base_width);
    src_woff  = Op::SrcPrevWidth(base_width);
    dst_woff  = Op::DstPrevWidth(base_width);
    inc_h     = Op::IncPrevHeight(base_height);
    src_hoff  = Op::SrcPrevHeight(base_height);
    dst_hoff  = Op::DstPrevHeight(base_height);
    adj_w_src = x_1_0_with_halo;
    adj_h_src = x_0_1_with_halo;
    diag_src = x_0_0_with_halo;
    dst = x_1_1_with_halo;
  }

  const uint32_t grid_dim_per_edge = gridDim.x / 2;

  if (blockIdx.x < grid_dim_per_edge) {
    const uint32_t total_count = base_width * num_channels * batch_size;
    const uint32_t block_maxcount = (total_count + grid_dim_per_edge - 1) / grid_dim_per_edge;
    const uint32_t blk_idx = blockIdx.x - grid_dim_per_edge;
    for (uint32_t delta = 0; delta < halo_pad; ++delta) {
      spatial2d_map_h_slice<T, Map, Layout>(
          block_maxcount * blk_idx, min(block_maxcount, total_count - block_maxcount * blk_idx),
          base_width, halo_pad, halo_width,
          base_height, halo_pad, halo_height,
          num_channels,
          batch_size,
          adj_h_src, src_hoff + inc_h * delta,
          dst,       dst_hoff + inc_h * delta);
    }
  } else if (blockIdx.x < grid_dim_per_edge * 2) {
    const uint32_t total_count = base_height * num_channels * batch_size;
    const uint32_t block_maxcount = (total_count + grid_dim_per_edge - 1) / grid_dim_per_edge;
    const uint32_t blk_idx = blockIdx.x;
    for (uint32_t delta = 0; delta < halo_pad; ++delta) {
      spatial2d_map_w_slice<T, Map, Layout>(
          block_maxcount * blk_idx, min(block_maxcount, total_count - block_maxcount * blk_idx),
          base_width, halo_pad, halo_width,
          base_height, halo_pad, halo_height,
          num_channels,
          batch_size,
          adj_w_src, src_woff + inc_w * delta,
          dst,       dst_woff + inc_w * delta);
    }
  } else {
    for (uint32_t delta_w = 0; delta_w < halo_pad; ++delta_w) {
      for (uint32_t delta_h = 0; delta_h < halo_pad; ++delta_h) {
        spatial2d_map_hw_slice<T, Map, Layout>(
            0, num_channels * batch_size,
            base_width, halo_pad, halo_width,
            base_height, halo_pad, halo_height,
            num_channels,
            batch_size,
            diag_src, src_woff + inc_w * delta_w, src_hoff + inc_h * delta_h,
            dst,      dst_woff + inc_w * delta_w, dst_hoff + inc_h * delta_h);
      }
    }
  }
}

template <typename T>
void spatial2d_2x2_halo_exch_nchw(
    uint32_t rank_w,
    uint32_t rank_h,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_0_with_halo,
    T* x_0_1_with_halo,
    T* x_1_0_with_halo,
    T* x_1_1_with_halo,
    cudaStream_t stream)
{
  // NOTE: The total work here does not quite correspond to how the kernel is
  // mapped onto blocks.
  uint32_t total_work = halo_pad * (base_width + base_height) * num_channels * batch_size;
  uint32_t num_blocks = max(1, min(NUM_BLOCKS, (total_work + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK));
  uint32_t num_blocks_adjusted = (num_blocks + 1) / 2 * 2 + 1;
  spatial2d_2x2_halo_op_kernel<T, ExchOp, CopyMap, NCHWLayout><<<num_blocks_adjusted, BLOCK_SIZE, 0, stream>>>(
      rank_w,
      rank_h,
      halo_pad,
      base_width,
      base_height,
      num_channels,
      batch_size,
      x_0_0_with_halo,
      x_0_1_with_halo,
      x_1_0_with_halo,
      x_1_1_with_halo);
}

template <>
void spatial2d_2x2_halo_exch_nchw<float>(
    uint32_t rank_w,
    uint32_t rank_h,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    float* x_0_0_with_halo,
    float* x_0_1_with_halo,
    float* x_1_0_with_halo,
    float* x_1_1_with_halo,
    cudaStream_t stream);

template <typename T>
void spatial2d_2x2_halo_reduce_nchw(
    uint32_t rank_w,
    uint32_t rank_h,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_0_with_halo,
    T* x_0_1_with_halo,
    T* x_1_0_with_halo,
    T* x_1_1_with_halo,
    cudaStream_t stream)
{
  // NOTE: The total work here does not quite correspond to how the kernel is
  // mapped onto blocks.
  uint32_t total_work = halo_pad * (base_width + base_height) * num_channels * batch_size;
  uint32_t num_blocks = max(1, min(NUM_BLOCKS, (total_work + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK));
  uint32_t num_blocks_adjusted = (num_blocks + 1) / 2 * 2 + 1;
  spatial2d_2x2_halo_op_kernel<T, ReduceOp, AtomicReduceMap, NCHWLayout><<<num_blocks_adjusted, BLOCK_SIZE, 0, stream>>>(
      rank_w,
      rank_h,
      halo_pad,
      base_width,
      base_height,
      num_channels,
      batch_size,
      x_0_0_with_halo,
      x_0_1_with_halo,
      x_1_0_with_halo,
      x_1_1_with_halo);
}

template <>
void spatial2d_2x2_halo_reduce_nchw<float>(
    uint32_t rank_w,
    uint32_t rank_h,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    float* x_0_0_with_halo,
    float* x_0_1_with_halo,
    float* x_1_0_with_halo,
    float* x_1_1_with_halo,
    cudaStream_t stream);

} // namespace arraydiff
