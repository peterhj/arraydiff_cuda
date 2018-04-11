#include "arraydiff/array_gpu.hh"
#include "arraydiff/array_abstract.hh"
#include "arraydiff/cuda_common.hh"

#include <cassert>

namespace arraydiff {

template <>
cudnnDataType_t GPUArray<Shape0D, float>::cudnn_data_type() {
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t GPUArray<Shape1D, float>::cudnn_data_type() {
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t GPUArray<Shape2D, float>::cudnn_data_type() {
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t GPUArray<Shape3D, float>::cudnn_data_type() {
  return CUDNN_DATA_FLOAT;
}

template <>
cudnnDataType_t GPUArray<Shape4D, float>::cudnn_data_type() {
  return CUDNN_DATA_FLOAT;
}

template <>
void GPUArray<Shape0D, float>::set_cudnn_tensor_desc(cudnnTensorDescriptor_t desc, cudnnTensorFormat_t format) {
  if (format == CUDNN_TENSOR_NCHW) {
    if (dim_.is_packed(stride_)) {
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          desc,
          format,
          cudnn_data_type(),
          1,
          1,
          1,
          1));
    } else {
      assert(0 && "unimplemented");
    }
  } else {
    assert(0 && "unimplemented");
  }
}

template <>
void GPUArray<Shape1D, float>::set_cudnn_tensor_desc(cudnnTensorDescriptor_t desc, cudnnTensorFormat_t format) {
  if (format == CUDNN_TENSOR_NCHW) {
    if (dim_.is_packed(stride_)) {
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          desc,
          format,
          cudnn_data_type(),
          1,
          dim_.shape,
          1,
          1));
    } else {
      assert(0 && "unimplemented");
    }
  } else {
    assert(0 && "unimplemented");
  }
}

template <>
void GPUArray<Shape2D, float>::set_cudnn_tensor_desc(cudnnTensorDescriptor_t desc, cudnnTensorFormat_t format) {
  if (format == CUDNN_TENSOR_NCHW) {
    if (dim_.is_packed(stride_)) {
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          desc,
          format,
          cudnn_data_type(),
          dim_.shape.at(1),
          dim_.shape.at(0),
          1,
          1));
    } else {
      assert(0 && "unimplemented");
    }
  } else {
    assert(0 && "unimplemented");
  }
}

template <>
void GPUArray<Shape3D, float>::set_cudnn_tensor_desc(cudnnTensorDescriptor_t desc, cudnnTensorFormat_t format) {
  if (format == CUDNN_TENSOR_NCHW) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        desc,
        cudnn_data_type(),
        dim_.shape.at(2),
        1,
        dim_.shape.at(1),
        dim_.shape.at(0),
        stride_.shape.at(2),
        stride_.shape.at(2),
        stride_.shape.at(1),
        stride_.shape.at(0)));
  } else {
    assert(0 && "unimplemented");
  }
}

template <>
void GPUArray<Shape4D, float>::set_cudnn_tensor_desc(cudnnTensorDescriptor_t desc, cudnnTensorFormat_t format) {
  if (format == CUDNN_TENSOR_NCHW) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        desc,
        cudnn_data_type(),
        dim_.shape.at(3),
        dim_.shape.at(2),
        dim_.shape.at(1),
        dim_.shape.at(0),
        stride_.shape.at(3),
        stride_.shape.at(2),
        stride_.shape.at(1),
        stride_.shape.at(0)));
  } else if (format == CUDNN_TENSOR_NHWC) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        desc,
        cudnn_data_type(),
        dim_.shape.at(3),
        dim_.shape.at(0),
        dim_.shape.at(2),
        dim_.shape.at(1),
        stride_.shape.at(3),
        stride_.shape.at(0),
        stride_.shape.at(2),
        stride_.shape.at(1)));
  } else {
    assert(0 && "unimplemented");
  }
}

template <>
void GPUArray<Shape4D, float>::set_cudnn_filter_desc(cudnnFilterDescriptor_t desc, cudnnTensorFormat_t format) {
  if (format == CUDNN_TENSOR_NCHW) {
    assert(dim_.is_packed(stride_));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        desc,
        cudnn_data_type(),
        format,
        dim_.shape.at(3),
        dim_.shape.at(2),
        dim_.shape.at(1),
        dim_.shape.at(0)));
  } else {
    assert(0 && "unimplemented");
  }
}

} // namespace arraydiff
