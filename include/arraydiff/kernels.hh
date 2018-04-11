#ifndef ARRAYDIFF_KERNELS_HH
#define ARRAYDIFF_KERNELS_HH

//#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

//namespace arraydiff {

extern "C" void arraydiff_array_add_packed_kernel(uint32_t len, const float* x, float* y, cudaStream_t stream);

//template <typename T>
//void array_add_packed_kernel(uint32_t len, const T* x, T* y, cudaStream_t stream);

/*template <>
void array_add_packed_kernel<half>(uint32_t len, const half* x, half* y, cudaStream_t stream);

template <>
void array_add_packed_kernel<float>(uint32_t len, const float* x, float* y, cudaStream_t stream);*/

//} // namespace arraydiff

namespace arraydiff {
//namespace kernels {

class ZeroMap {
public:
  template <typename T>
  __device__ __forceinline__ static void Map(T* dst, const T* src);
};

class CopyMap {
public:
  template <typename T>
  __device__ __forceinline__ static void Map(T* dst, const T* src);
};

class ReduceMap {
public:
  template <typename T>
  __device__ __forceinline__ static void Map(T* dst, const T* src);
};

class AtomicReduceMap {
public:
  template <typename T>
  __device__ __forceinline__ static void Map(T* dst, const T* src);
};

template <typename T>
__device__ __forceinline__ void ZeroMap::Map(T* dst, const T* src) {
  *dst = static_cast<T>(0);
}

template <typename T>
__device__ __forceinline__ void CopyMap::Map(T* dst, const T* src) {
  *dst = *src;
}

template <typename T>
__device__ __forceinline__ void ReduceMap::Map(T* dst, const T* src) {
  *dst += *src;
}

template <typename T>
__device__ __forceinline__ void AtomicReduceMap::Map(T* dst, const T* src) {
  (void)atomicAdd(dst, *src);
}

template <typename T>
void array_set_constant_packed(uint32_t dim, T constant, T* x, cudaStream_t stream);

template <typename T>
void array_scale_constant_packed(uint32_t dim, T constant, T* x, cudaStream_t stream);

template <typename T>
void array_add_packed(uint32_t dim, const T* x, T* y, cudaStream_t stream);

template <typename T>
void array_add_scaled_packed(uint32_t dim, T alpha, T beta, const T* x, T* y, cudaStream_t stream);

template <typename T>
void array_add_online_packed(uint32_t dim, T alpha, const T* x, T* y, cudaStream_t stream);

template <typename T>
void array_scale_packed(uint32_t dim, T scale, const T* x, T* y, cudaStream_t stream);

template <typename T>
void array_scale_add_packed(uint32_t dim, T scale, const T* x, T* y, cudaStream_t stream);

template <typename T>
void softmax_fwd(uint32_t num_channels, uint32_t batch_size, const T* x, T* y, cudaStream_t stream);

template <typename T>
void softmax_nll_loss_fwd(uint32_t num_channels, uint32_t batch_size, const T* y, const uint32_t* label, T* loss, cudaStream_t stream);

template <typename T>
void softmax_nll_loss_bwd(uint32_t num_channels, uint32_t batch_size, const T* y, const uint32_t* label, const T* dloss, T* dx, cudaStream_t stream);

template <typename T>
void sum_batch_join_fwd(
    uint32_t batch_size,
    const T* x,
    T* y,
    cudaStream_t stream);
template <typename T>
void sum_batch_join_bwd(
    uint32_t batch_size,
    const T* dy,
    T* dx,
    cudaStream_t stream);

template <typename T>
void spatial2d_1x2_halo_exch_nchw(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_with_halo,
    T* x_1_with_halo,
    cudaStream_t stream);

template <typename T>
void spatial2d_1x2_halo_reduce_nchw(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_with_halo,
    T* x_1_with_halo,
    cudaStream_t stream);

template <typename T>
void spatial2d_1x4_halo_exch_nchw(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_with_halo,
    T* x_1_with_halo,
    T* x_2_with_halo,
    T* x_3_with_halo,
    cudaStream_t stream);

template <typename T>
void spatial2d_1x4_halo_reduce_nchw(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_with_halo,
    T* x_1_with_halo,
    T* x_2_with_halo,
    T* x_3_with_halo,
    cudaStream_t stream);

template <typename T>
void spatial2d_2x1_halo_exch_nchw(
    uint32_t rank,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_with_halo,
    T* x_1_with_halo,
    cudaStream_t stream);

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
    cudaStream_t stream);

template <typename T>
void spatial2d_2x2_halo_exch_nchw(
    uint32_t rank_w,
    uint32_t rank_h,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    T* x_0_with_halo,
    T* x_1_with_halo,
    T* x_2_with_halo,
    T* x_3_with_halo,
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
    T* x_0_with_halo,
    T* x_1_with_halo,
    T* x_2_with_halo,
    T* x_3_with_halo,
    cudaStream_t stream);

template <typename T>
void slice2d_copy_hw_nchw_packed(
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
    uint32_t dst_h_off,
    cudaStream_t stream);

template <typename T>
void slice2d_add_hw_nchw_packed(
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
    uint32_t dst_h_off,
    cudaStream_t stream);

template <typename From, typename To>
void array_cast(
    uint32_t size,
    From* x,
    To* y,
    cudaStream_t stream);

//} // namespace kernels
} // namespace arraydiff

#endif
