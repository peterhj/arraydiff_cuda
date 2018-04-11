#include "arraydiff/kernels.hh"
//#include "kernels/array.hh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
  
template <typename T>
__global__ void arraydiff_array_add_packed_kernel_impl(
    uint32_t dim,
    const T* x,
    T* y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    y[idx] += x[idx];
  }
}

/*template <typename T>
__global__ void arraydiff_array_add_2d_kernel_impl(
    uint32_t dim,
    uint32_t stride,
    const T* x,
    T* y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  // TODO
  if (idx < dim) {
    y[idx] += x[idx];
  }
}*/

/*extern "C" void arraydiff_array_add_packed_kernel(uint32_t dim, const float* x, float* y, cudaStream_t stream) {
  arraydiff_array_add_packed_kernel_impl<float><<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, x, y);
}*/

/*namespace arraydiff {

template <typename T>
void array_add_packed_kernel(uint32_t dim, const T* x, T* y, cudaStream_t stream) {
  arraydiff_array_add_packed_kernel_impl<T><<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, x, y);
}

template <>
void array_add_packed_kernel<half>(uint32_t dim, const half* x, half* y, cudaStream_t stream);

template <>
void array_add_packed_kernel<float>(uint32_t dim, const float* x, float* y, cudaStream_t stream);

} // namespace arraydiff
*/

template <typename T>
__global__ void arraydiff_array_set_constant_packed_kernel(
    uint32_t dim,
    T constant,
    T* x)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    x[idx] = constant;
  }
}

template <typename T>
__global__ void arraydiff_array_scale_constant_packed_kernel(
    uint32_t dim,
    T constant,
    T* x)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    float x_i = x[idx];
    x[idx] = constant * x_i;
  }
}

template <typename T>
__global__ void arraydiff_array_add_packed_kernel(
    uint32_t dim,
    const T* x,
    T* y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    float y_i = y[idx];
    y[idx] = x[idx] + y_i;
  }
}

template <typename T>
__global__ void arraydiff_array_add_scaled_packed_kernel(
    uint32_t dim,
    T alpha,
    T beta,
    const T* x,
    T* y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    float y_i = y[idx];
    y[idx] = alpha * x[idx] + beta * y_i;
  }
}

template <typename T>
__global__ void arraydiff_array_add_online_packed_kernel(
    uint32_t dim,
    T alpha,
    const T* x,
    T* y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    float y_i = y[idx];
    y[idx] = y_i + alpha * (x[idx] - y_i);
  }
}

template <typename T>
__global__ void arraydiff_array_scale_packed_kernel(
    uint32_t dim,
    T scale,
    const T* x,
    T* y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    y[idx] = scale * x[idx];
  }
}

template <typename T>
__global__ void arraydiff_array_scale_add_packed_kernel(
    uint32_t dim,
    T scale,
    const T* x,
    T* y)
{
  uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < dim) {
    float y_i = y[idx];
    y[idx] = scale * x[idx] + y_i;
  }
}

namespace arraydiff {
//namespace kernels {

template <>
void array_set_constant_packed<float>(
    uint32_t dim,
    float constant,
    float* x,
    cudaStream_t stream)
{
  arraydiff_array_set_constant_packed_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, constant, x);
}

template <>
void array_set_constant_packed<uint8_t>(
    uint32_t dim,
    uint8_t constant,
    uint8_t* x,
    cudaStream_t stream)
{
  arraydiff_array_set_constant_packed_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, constant, x);
}

template <>
void array_set_constant_packed<uint32_t>(
    uint32_t dim,
    uint32_t constant,
    uint32_t* x,
    cudaStream_t stream)
{
  arraydiff_array_set_constant_packed_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, constant, x);
}

template <>
void array_scale_constant_packed<float>(
    uint32_t dim,
    float constant,
    float* x,
    cudaStream_t stream)
{
  arraydiff_array_scale_constant_packed_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, constant, x);
}

template <>
void array_add_packed<float>(
    uint32_t dim,
    const float* x,
    float* y,
    cudaStream_t stream)
{
  arraydiff_array_add_packed_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, x, y);
}

template <>
void array_add_scaled_packed<float>(
    uint32_t dim,
    float alpha,
    float beta,
    const float* x,
    float* y,
    cudaStream_t stream)
{
  arraydiff_array_add_scaled_packed_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, alpha, beta, x, y);
}

template <>
void array_add_online_packed<float>(
    uint32_t dim,
    float alpha,
    const float* x,
    float* y,
    cudaStream_t stream)
{
  arraydiff_array_add_online_packed_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, alpha, x, y);
}

template <>
void array_scale_packed<float>(
    uint32_t dim,
    float scale,
    const float* x,
    float* y,
    cudaStream_t stream)
{
  arraydiff_array_scale_packed_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, scale, x, y);
}

template <>
void array_scale_add_packed<float>(
    uint32_t dim,
    float scale,
    const float* x,
    float* y,
    cudaStream_t stream)
{
  arraydiff_array_scale_add_packed_kernel<<<(dim+1024-1)/1024, 1024, 0, stream>>>(
      dim, scale, x, y);
}

//} // namespace kernels
} // namespace arraydiff
