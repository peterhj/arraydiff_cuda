#include "arraydiff/context.hh"
#include "arraydiff/array_gpu.hh"
#include "arraydiff/cuda_common.hh"
#include "arraydiff/spatial.hh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <nccl.h>

#include <cassert>
#include <iostream>
#include <memory>

namespace arraydiff {

using std::make_shared;
using std::shared_ptr;
using std::vector;

static thread_local vector<int> device_stack;

GPUConn::GPUConn(int device, cudaStream_t stream, cublasHandle_t cublas_h, cudnnHandle_t cudnn_h)
  : device_(device), stream_(stream), cublas_h_(cublas_h), cudnn_h_(cudnn_h)
{
  device_stack.push_back(device_);
  CUDA_CHECK(cudaSetDevice(device_));
}

GPUConn::~GPUConn() {
  device_stack.pop_back();
  size_t stack_size = device_stack.size();
  if (0 != stack_size) {
    CUDA_CHECK(cudaSetDevice(device_stack[stack_size-1]));
  } else {
    CUDA_CHECK(cudaSetDevice(0));
  }
}

void GPUConn::sync() {
  //std::clog << "DEBUG: GPUConn: sync(): device: " << device_ << " stream: " << stream_ << std::endl;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaDeviceSynchronize());
}

shared_ptr<GPUContext> GPUContext::Make(int device) {
  return make_shared<GPUContext>(device);
}

GPUContext::GPUContext(int device)
  : device_(device), scratch_size_(0)
{
  std::clog << "DEBUG: GPUContext()" << std::endl;
  int prev_device = 0;
  CUDA_CHECK(cudaGetDevice(&prev_device));
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasCreate(&cublas_h_));
  CUDNN_CHECK(cudnnCreate(&cudnn_h_));
  CUDA_CHECK(cudaSetDevice(prev_device));
}

void GPUContext::sync() {
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaDeviceSynchronize());
}

GPUConn GPUContext::conn() {
  return GPUConn(
      device_,
      stream_,
      cublas_h_,
      cudnn_h_);
}

shared_ptr<GPUMemory<uint8_t>> GPUContext::get_scratch() {
  if (!static_cast<bool>(scratch_)) {
    sync();
    size_t scratch_size_rounded_up = (scratch_size_ + 4096UL - 1UL) / 4096UL * 4096UL;
    std::clog << "DEBUG: GPUContext: get scratch: alloc: " << scratch_size_rounded_up << " bytes" << std::endl;
    scratch_size_ = scratch_size_rounded_up;
    scratch_ = make_shared<GPUMemory<uint8_t>>(scratch_size_rounded_up, this);
  }
  return scratch_.value();
}

void GPUContext::reserve_scratch(size_t min_scratch_size) {
  if (min_scratch_size <= scratch_size_) {
    return;
  }
  std::clog << "DEBUG: GPUContext: reserve scratch: " << min_scratch_size << " bytes" << std::endl;
  scratch_size_ = min_scratch_size;
  if (static_cast<bool>(scratch_)) {
    sync();
    size_t scratch_size_rounded_up = (scratch_size_ + 4096UL - 1UL) / 4096UL * 4096UL;
    std::clog << "DEBUG: GPUContext: reserve scratch: alloc: " << scratch_size_rounded_up << " bytes" << std::endl;
    scratch_size_ = scratch_size_rounded_up;
    scratch_.value()->reallocate(scratch_size_rounded_up, this);
  }
}

/*MultiGPUContext::MultiGPUContext(vector<shared_ptr<GPUContext>> dev_ctxs)
  : dev_ctxs_(dev_ctxs), comms_(dev_ctxs.size())
{
  size_t num_ranks = dev_ctxs_.size();
  Spatial2DComm::InitAll(comms_.data(), 1, num_ranks);
}*/

shared_ptr<MultiGPUContext> MultiGPUContext::Make() {
  return make_shared<MultiGPUContext>();
}

MultiGPUContext::MultiGPUContext()
  : num_ranks_(0), dev_ctxs_(), nccl_comms_(NULL), spatial_comms_(NULL)
{
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  assert(device_count > 0);
  std::clog << "DEBUG: MultiGPUContext: init: devices: " << device_count << std::endl;
  num_ranks_ = device_count;
  for (int dev = 0; dev < device_count; ++dev) {
    auto dev_ctx = GPUContext::Make(dev);
    dev_ctxs_.push_back(dev_ctx);
  }
  nccl_devs_ = new int[num_ranks_];
  nccl_comms_ = new ncclComm_t[num_ranks_];
  for (int dev = 0; dev < device_count; ++dev) {
    nccl_devs_[dev] = dev;
  }
  std::clog << "DEBUG: MultiGPUContext: init: nccl..." << std::endl;
  //NCCL_CHECK(ncclCommInitAll(nccl_comms_, device_count, NULL));
  spatial_comms_ = new Spatial2DComm[num_ranks_];
  std::clog << "DEBUG: MultiGPUContext: init: spatial..." << std::endl;
  Spatial2DComm::InitAll(spatial_comms_, 1, num_ranks_);
  std::clog << "DEBUG: MultiGPUContext: init: done" << std::endl;
}

void MultiGPUContext::sync() {
  for (size_t rank = 0; rank < num_ranks_; ++rank) {
    dev_ctxs_.at(rank)->sync();
  }
}

shared_ptr<GPUContext> MultiGPUContext::device_context(size_t rank) {
  return this->dev_ctxs_.at(rank);
}

ncclComm_t MultiGPUContext::nccl_comm(size_t rank) {
  assert(rank < num_ranks_);
  return nccl_comms_[rank];
}

Spatial2DComm* MultiGPUContext::spatial_comm(size_t rank) {
  assert(rank < num_ranks_);
  return spatial_comms_ + rank;
}

} // namespace arraydiff
