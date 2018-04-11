#ifndef ARRAYDIFF_CONTEXT_HH
#define ARRAYDIFF_CONTEXT_HH

#include <cuda_runtime.h>
#include <cublas.h>
#include <cudnn.h>
#include <nccl.h>

#include <cassert>
#include <experimental/optional>
#include <memory>
#include <vector>

namespace arraydiff {

using std::experimental::optional;
using std::shared_ptr;
using std::vector;

template <typename T>
class GPUMemory;

class Context {
public:
  virtual ~Context() {}

  virtual void sync() = 0;
};

class GPUConn {
public:
  GPUConn(int device, cudaStream_t stream, cublasHandle_t cublas_h, cudnnHandle_t cudnn);
  ~GPUConn();

  int device() const {
    return this->device_;
  }
  cudaStream_t stream() {
    return this->stream_;
  }
  cublasHandle_t cublas() {
    return this->cublas_h_;
  }
  cudnnHandle_t cudnn() {
    return this->cudnn_h_;
  }

  void sync();

private:
  int device_;
  cudaStream_t stream_;
  cublasHandle_t cublas_h_;
  cudnnHandle_t cudnn_h_;
};

class GPUContext : public virtual Context {
public:
  static shared_ptr<GPUContext> Make(int device);

  explicit GPUContext(int device);
  virtual ~GPUContext() {}

  virtual void sync();

  GPUConn conn();
  shared_ptr<GPUMemory<uint8_t>> get_scratch();
  void reserve_scratch(size_t min_scratch_size);

private:
  int device_;
  cudaStream_t stream_;
  cublasHandle_t cublas_h_;
  cudnnHandle_t cudnn_h_;
  size_t scratch_size_;
  optional<shared_ptr<GPUMemory<uint8_t>>> scratch_;
};

class Spatial2DComm;

class MultiGPUContext : public virtual Context {
public:
  static shared_ptr<MultiGPUContext> Make();

  MultiGPUContext();

  virtual ~MultiGPUContext() {}

  virtual void sync();
  void sync_rank(size_t rank);

  size_t num_ranks() const { return this->dev_ctxs_.size(); }

  shared_ptr<GPUContext> device_context(size_t rank);
  ncclComm_t nccl_comm(size_t rank);
  Spatial2DComm* spatial_comm(size_t rank);

private:
  size_t num_ranks_;
  vector<shared_ptr<GPUContext>> dev_ctxs_;
  int* nccl_devs_;
  ncclComm_t* nccl_comms_;
  Spatial2DComm* spatial_comms_;
};

} // namespace arraydiff

#endif
