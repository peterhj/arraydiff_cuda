#ifndef ARRAYDIFF_ARRAY_GPU_HH
#define ARRAYDIFF_ARRAY_GPU_HH

#include "arraydiff/array_abstract.hh"
#include "arraydiff/context.hh"
#include "arraydiff/cuda_common.hh"
#include "arraydiff/kernels.hh"
#include "arraydiff/slice.hh"

#include <cudnn.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <experimental/any>
#include <experimental/optional>
#include <iostream>
#include <memory>

namespace arraydiff {

using std::experimental::any;
using std::experimental::any_cast;
using std::experimental::optional;
using std::shared_ptr;

template <typename T>
class HostPinnedMemory {
public:
  static HostPinnedMemory<T> Zeros(size_t size, GPUConn& conn) {
    HostPinnedMemory<T> mem(size, conn);
    for (size_t idx = 0; idx < mem.size_; ++idx) {
      mem.hptr_[idx] = static_cast<T>(0);
    }
    return mem;
  }

  HostPinnedMemory()
    : hptr_(NULL), size_(0), device_(-1), owned_(false) {}

  HostPinnedMemory(size_t size, GPUConn& conn) {
    T* hptr = NULL;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMallocHost((void**)(&hptr), size * sizeof(T)));
    this->hptr_ = hptr;
    this->size_ = size;
    this->device_ = conn.device();
    this->owned_ = true;
  }

  ~HostPinnedMemory() {
    // TODO
  }

  bool invalid() const { return NULL == hptr_; }
  bool valid() const { return NULL != hptr_; }

  T* ptr() { return hptr_; }

private:
  T* hptr_;
  size_t size_;
  int device_;
  bool owned_;
};

template <typename T>
class GPUMemory {
public:
  static GPUMemory<T> Zeros(size_t size, GPUContext* ctx) {
    GPUMemory<T> mem(size, ctx);
    GPUConn conn = ctx->conn();
    mem.set_constant(static_cast<T>(0), conn);
    conn.sync();
    return mem;
  }

  GPUMemory() {}
  //GPUMemory(const GPUMemory<T>& other) = delete;
  //GPUMemory& operator=(const GPUMemory<T>& other) = delete;

  GPUMemory(size_t count, GPUContext* ctx) {
    GPUConn conn = ctx->conn();
    T* dptr = NULL;
    const size_t size_bytes = count * sizeof(T);
    //std::clog << "DEBUG: GPUMemory: alloc:      " << size_bytes << " bytes / " << count << " elems" << std::endl;
    size_t size_bytes_rounded_up = (size_bytes + 4096UL - 1UL) / 4096UL * 4096UL;
    if (0UL == size_bytes_rounded_up) {
      size_bytes_rounded_up = 4096UL;
    }
    //std::clog << "DEBUG: GPUMemory: alloc (up): " << size_bytes_rounded_up << " bytes" << std::endl;
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMalloc((void**)(&dptr), size_bytes_rounded_up));
    assert(dptr != NULL);
    //std::clog << "DEBUG: GPUMemory: dptr: " << reinterpret_cast<uintptr_t>(dptr) << std::endl;
    this->dptr_ = dptr;
    this->size_ = count;
    this->device_ = conn.device();
    this->owned_ = true;
  }

  GPUMemory(T* dptr, size_t size, int device)
    : dptr_(dptr), size_(size), device_(device), owned_(false) {}

  ~GPUMemory() {
    if (this->owned_) {
      // TODO: just leak for now.
    }
  }

  void reallocate(size_t new_size, GPUContext* ctx) {
    if (new_size < size_) {
      return;
    }
    std::clog << "DEBUG: GPUMemory: realloc: " << size_ * sizeof(T) << " bytes => " << new_size * sizeof(T) << " bytes" << std::endl;
    GPUConn conn = ctx->conn();
    // TODO: check for errors.
    CUDA_CHECK(cudaDeviceSynchronize());
    if (this->dptr_ != NULL) {
      CUDA_CHECK(cudaFree((void*)(this->dptr_)));
    }
    T* new_dptr = NULL;
    CUDA_CHECK(cudaMalloc((void**)(&new_dptr), new_size * sizeof(T)));
    this->dptr_ = new_dptr;
    this->size_ = new_size;
    this->device_ = conn.device();
    this->owned_ = true;
  }

  GPUMemory slice(size_t start_offset, size_t end_offset) {
    assert(start_offset <= end_offset);
    assert(end_offset <= this->size_);
    T* slice_dptr = this->dptr_ + start_offset;
    size_t slice_size = end_offset - start_offset;
    return GPUMemory(slice_dptr, slice_size, this->device());
  }

  uintptr_t dptr_addr() {
    return reinterpret_cast<uintptr_t>(dptr_);
  }
  T* dptr() {
    return dptr_;
  }
  size_t size() const {
    return size_;
  }
  int device() const {
    return device_;
  }

  void set_constant(T constant, GPUConn& conn) {
    array_set_constant_packed<T>(
        static_cast<uint32_t>(size()),
        constant,
        dptr(),
        conn.stream());
  }

  void scale(T constant, GPUConn& conn) {
    array_scale_constant_packed<T>(
        static_cast<uint32_t>(size()),
        constant,
        dptr(),
        conn.stream());
  }

  void add(GPUMemory<T> src, size_t buf_sz, size_t dst_offset, size_t src_offset, GPUConn& conn) {
    assert(buf_sz <= size());
    assert(buf_sz <= src.size());
    // TODO: bounds checks.
    array_add_packed<T>(
        buf_sz,
        src.dptr() + src_offset,
        this->dptr() + dst_offset,
        conn.stream());
  }

  void add_scaled(T alpha, GPUMemory<T> src, T beta, size_t buf_sz, size_t dst_offset, size_t src_offset, GPUConn& conn) {
    assert(buf_sz <= size());
    assert(buf_sz <= src.size());
    // TODO: bounds checks.
    array_add_scaled_packed<T>(
        buf_sz,
        alpha, beta,
        src.dptr() + src_offset,
        this->dptr() + dst_offset,
        conn.stream());
  }

  void add_online(T alpha, GPUMemory<T> src, size_t buf_sz, size_t dst_offset, size_t src_offset, GPUConn& conn) {
    assert(buf_sz <= size());
    assert(buf_sz <= src.size());
    // TODO: bounds checks.
    array_add_online_packed<T>(
        buf_sz,
        alpha,
        src.dptr() + src_offset,
        this->dptr() + dst_offset,
        conn.stream());
  }

  void copy(GPUMemory<T> src, size_t buf_sz, size_t dst_offset, size_t src_offset, GPUConn& conn) {
    assert(buf_sz <= size());
    assert(buf_sz <= src.size());
    // TODO: bounds checks.
    //assert(size() >= dst_offset + src.size());
    CUDA_CHECK(cudaMemcpyAsync(
        this->dptr() + dst_offset,
        src.dptr() + src_offset,
        buf_sz * sizeof(T),
        cudaMemcpyDeviceToDevice,
        conn.stream()));
  }

  void load_async(const T* src_ptr, size_t dst_offset, size_t buf_size, GPUConn& conn) {
    assert(size() >= buf_size);
    assert(size() >= buf_size + dst_offset);
    assert(buf_size + dst_offset >= buf_size);
    conn.sync();
    if (host_shadow_.invalid()) {
      host_shadow_ = HostPinnedMemory<T>(size(), conn);
      conn.sync();
    }
    assert(host_shadow_.valid());
    std::memcpy(
        host_shadow_.ptr() + dst_offset,
        src_ptr,
        buf_size);
    CUDA_CHECK(cudaMemcpyAsync(
        this->dptr() + dst_offset,
        host_shadow_.ptr() + dst_offset,
        buf_size * sizeof(T),
        cudaMemcpyHostToDevice,
        conn.stream()));
  }

  void load_sync(const T* src_ptr, size_t dst_offset, size_t buf_size, GPUConn& conn) {
    //std::clog << "DEBUG: GPUMemory: load_sync():"
    //    " dst offset: " << dst_offset <<
    //    " buf size: " << buf_size <<
    //    " bytes size: " << buf_size * sizeof(T) << std::endl;
    assert(size() >= buf_size);
    assert(size() >= buf_size + dst_offset);
    conn.sync();
    CUDA_CHECK(cudaMemcpy(
    //CUDA_CHECK(cudaMemcpyAsync(
        this->dptr() + dst_offset,
        src_ptr,
        buf_size * sizeof(T),
        cudaMemcpyDefault));
        //cudaMemcpyHostToDevice,
        //conn.stream()));
    conn.sync();
  }

  void store_sync(T* dst_ptr, size_t src_offset, size_t buf_size, GPUConn& conn) {
    //std::clog << "DEBUG: GPUMemory: store_sync():"
    //    " src offset: " << src_offset <<
    //    " buf size: " << buf_size <<
    //    " bytes size: " << buf_size * sizeof(T) << std::endl;
    //std::clog << "DEBUG: GPUMemory:   [ 0]: " << dst_ptr[0] << std::endl;
    //std::clog << "DEBUG: GPUMemory:   [ 1]: " << dst_ptr[1] << std::endl;
    //std::clog << "DEBUG: GPUMemory:   [15]: " << dst_ptr[15] << std::endl;
    assert(size() >= buf_size);
    assert(size() >= buf_size + src_offset);
    conn.sync();
    CUDA_CHECK(cudaMemcpy(
    //CUDA_CHECK(cudaMemcpyAsync(
        dst_ptr,
        this->dptr() + src_offset,
        buf_size * sizeof(T),
        cudaMemcpyDefault));
        //cudaMemcpyDeviceToHost,
        //conn.stream()));
    conn.sync();
    //std::clog << "DEBUG: GPUMemory:   [ 0]: " << dst_ptr[0] << std::endl;
    //std::clog << "DEBUG: GPUMemory:   [ 1]: " << dst_ptr[1] << std::endl;
    //std::clog << "DEBUG: GPUMemory:   [15]: " << dst_ptr[15] << std::endl;
  }

private:
  T* dptr_;
  size_t size_;
  int device_;
  bool owned_;
  HostPinnedMemory<T> host_shadow_;
};

template <typename Dim, typename T>
class GPUArray {
public:
  typedef GPUContext Ctx;
  typedef Dim Shape;

  static GPUArray<Dim, T> Zeros(Dim dim, size_t batch_capacity, GPUContext* ctx) {
    assert(batch_capacity == 0 && "unreachable");
    return GPUArray<Dim, T>::Zeros(dim, ctx);
  }

  static GPUArray<Dim, T> Zeros(Dim dim, GPUContext* ctx) {
    GPUArray<Dim, T> arr(dim, ctx);
    GPUConn conn = ctx->conn();
    arr.set_constant(static_cast<T>(0), conn);
    conn.sync();
    return arr;
  }

  static GPUArray<Dim, T> ZerosPadded(Dim dim, Dim offset, Dim stride, size_t batch_capacity, GPUContext* ctx) {
    assert(batch_capacity == 0 && "unreachable");
    return GPUArray<Dim, T>::ZerosPadded(dim, offset, stride, ctx);
  }

  static GPUArray<Dim, T> ZerosPadded(Dim dim, Dim offset, Dim stride, GPUContext* ctx) {
    if (offset.is_zero() && dim.is_packed(stride)) {
      return GPUArray<Dim, T>::Zeros(dim, ctx);
    } else {
      // FIXME
      assert(0 && "unimplemented");
    }
  }

  GPUArray(Dim dim, GPUContext* ctx)
    : dim_(dim), offset_(Dim::Zero()), stride_(dim.packed_stride()), offset_idx_(0)
  {
    //std::clog << "DEBUG: GPUArray: alloc: " << dim.flat_size() << " count" << std::endl;
    mem_ = GPUMemory<T>(dim.flat_size(), ctx);
  }

  GPUArray(GPUMemory<T>& mem, Dim dim, Dim offset, Dim stride)
    : mem_(mem), dim_(dim), offset_(offset), stride_(stride), offset_idx_(offset.to_index(stride))
  {}

  ~GPUArray() {}

  uintptr_t dptr_addr() {
    return mem_.dptr_addr();
  }

  T* dptr() {
    return mem_.dptr() + offset_idx_;
  }

  T* base_dptr() {
    return mem_.dptr();
  }

  Dim dim() const {
    return this->dim_;
  }
  Dim offset() const {
    return this->offset_;
  }
  Dim stride() const {
    return this->stride_;
  }

  cudnnDataType_t cudnn_data_type();
  void set_cudnn_tensor_desc(cudnnTensorDescriptor_t desc, cudnnTensorFormat_t format);
  void set_cudnn_filter_desc(cudnnFilterDescriptor_t desc, cudnnTensorFormat_t format);

  void set_constant(T constant, GPUConn& conn) {
    size_t flat_size = stride().minor() * dim().minor();
    array_set_constant_packed<T>(
        flat_size,
        constant,
        base_dptr(),
        conn.stream());
  }

  void add(GPUArray<Dim, T> src, GPUConn& conn) {
    assert(dim().shape == src.dim().shape);
    if (dim().is_packed(stride()) && src.dim().is_packed(src.stride())) {
      size_t flat_size = stride().minor() * dim().minor();
      array_add_packed<T>(
          flat_size,
          src.base_dptr(),
          base_dptr(),
          conn.stream());
    } else {
      assert(0 && "unimplemented");
    }
  }

  void copy(GPUArray<Dim, T> src, GPUConn& conn) {
    assert(dim().shape == src.dim().shape);
    if (dim().is_packed(stride()) && src.dim().is_packed(src.stride())) {
      mem_.copy(src.mem_, dim().flat_size(), 0, 0, conn);
    } else {
      assert(0 && "unimplemented");
    }
  }

  void load_async(const T* src_ptr, size_t src_size, GPUConn& conn) {
    if (dim().is_packed(stride())) {
      mem_.load_async(src_ptr, 0, src_size, conn);
    } else {
      std::clog << "DEBUG: dim:" << std::endl;
      dim()._debug_print();
      std::clog << "DEBUG: packed stride:" << std::endl;
      dim().packed_stride()._debug_print();
      std::clog << "DEBUG: stride:" << std::endl;
      stride()._debug_print();
      assert(0 && "unimplemented");
    }
  }

  void load_sync(const T* src_ptr, size_t src_size, GPUConn& conn) {
    if (dim().is_packed(stride())) {
      mem_.load_sync(src_ptr, 0, src_size, conn);
    } else {
      assert(0 && "unimplemented");
    }
  }

  void store_sync(T* dst_ptr, size_t dst_size, GPUConn& conn) {
    if (dim().is_packed(stride())) {
      mem_.store_sync(dst_ptr, 0, dst_size, conn);
    } else {
      assert(0 && "unimplemented");
    }
  }

  size_t load_any(size_t offset, any* reader, shared_ptr<Context> ctx) {
    size_t buf_size = dim().flat_size();
    assert(NULL != reader);
    {
      auto buf_reader = any_cast<GPUMemory<T>>(reader);
      if (NULL != buf_reader) {
        GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
        GPUConn conn = gpu_ctx->conn();
        assert(buf_reader->size() >= offset + buf_size);
        assert(buf_size == mem_.size());
        mem_.copy(*buf_reader, buf_size, 0, offset, conn);
        return offset + buf_size;
      }
    }
    /*{
      auto buf_reader = any_cast<vector<T>>(reader);
      if (NULL != buf_reader) {
        GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
        GPUConn conn = gpu_ctx->conn();
        // TODO
        load_sync(buf_reader->data() + offset, buf_size, conn);
        //load_async(buf_reader->data() + offset, buf_size, conn);
        return offset + buf_size;
      }
    }*/
    assert(0 && "unreachable");
  }

  size_t store_any(size_t offset, any* writer, shared_ptr<Context> ctx) {
    size_t buf_size = dim().flat_size();
    //std::clog << "DEBUG: GPUArray: store_any():" << std::endl;
    //std::clog << "DEBUG: GPUArray:   buf size: " << buf_size << std::endl;
    //std::clog << "DEBUG: GPUArray:   dim: ";
    //dim()._debug_print();
    //std::clog << std::endl;
    if (NULL == writer) {
      //std::clog << "DEBUG: GPUArray:   null writer" << std::endl;
      return offset + buf_size;
    }
    {
      auto buf_writer = any_cast<GPUMemory<T>>(writer);
      if (NULL != buf_writer) {
        GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
        GPUConn conn = gpu_ctx->conn();
        assert(buf_writer->size() >= offset + buf_size);
        assert(buf_size == mem_.size());
        buf_writer->copy(mem_, buf_size, offset, 0, conn);
        return offset + buf_size;
      }
    }
    /*{
      auto buf_writer = any_cast<vector<T>>(writer);
      if (NULL != buf_writer) {
        GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
        GPUConn conn = gpu_ctx->conn();
        assert(buf_writer->size() >= offset + buf_size);
        store_sync(buf_writer->data() + offset, buf_size, conn);
        return offset + buf_size;
      }
    }*/
    assert(0 && "unreachable");
  }

private:
  GPUMemory<T> mem_;
  Dim dim_;
  Dim offset_;
  Dim stride_;
  size_t offset_idx_;
};

template <typename Dim, typename T>
class GPUBatchArray {
public:
  typedef GPUContext Ctx;
  typedef Dim Shape;

  static GPUBatchArray<Dim, T> Zeros(Dim dim, size_t batch_capacity, GPUContext* ctx) {
    GPUBatchArray<Dim, T> arr(dim, batch_capacity, ctx);
    GPUConn conn = ctx->conn();
    arr.as_view().set_constant(static_cast<T>(0), conn);
    conn.sync();
    return arr;
  }

  static GPUBatchArray<Dim, T> ZerosPadded(Dim dim, Dim offset, Dim stride, size_t batch_capacity, GPUContext* ctx) {
    if (offset.is_zero() && dim.is_packed(stride)) {
      return GPUBatchArray<Dim, T>::Zeros(dim, batch_capacity, ctx);
    } else {
      GPUBatchArray<Dim, T> arr(dim, offset, stride, batch_capacity, ctx);
      GPUConn conn = ctx->conn();
      arr.base_as_view().set_constant(static_cast<T>(0), conn);
      conn.sync();
      return arr;
    }
  }

  GPUBatchArray(Dim dim, size_t batch_capacity, GPUContext* ctx)
    : dim_(dim), offset_idx_(0), batch_cap_(batch_capacity), batch_size_(batch_capacity)
  {
    offset_ = Dim::Zero();
    stride_ = dim.packed_stride();
    assert(batch_capacity >= 1);
    size_t flat_size = dim.flat_size() * batch_capacity;
    mem_ = GPUMemory<T>(flat_size, ctx);
  }

  GPUBatchArray(Dim dim, Dim offset, Dim stride, size_t batch_capacity, GPUContext* ctx)
    : dim_(dim), offset_(offset), stride_(stride), offset_idx_(0), batch_cap_(batch_capacity), batch_size_(batch_capacity)
  {
    assert(batch_capacity >= 1);
    size_t padded_flat_size = stride.minor() * dim.minor() * batch_capacity;
    mem_ = GPUMemory<T>(padded_flat_size, ctx);
  }

  ~GPUBatchArray() {}

  GPUArray<typename Dim:: Upper, T> as_view() {
    return GPUArray<typename Dim:: Upper, T>(mem_, dim_.append(batch_size_), offset_.append(0), stride_.append(stride_.minor() * dim_.minor()));
  }

  GPUArray<typename Dim:: Upper, T> base_as_view() {
    typedef typename Dim:: Upper Upper_;
    Dim packed_dim = stride_.packed_dim(dim_.minor());
    return GPUArray<typename Dim:: Upper, T>(mem_, packed_dim.append(batch_size_), Upper_::Zero(), stride_.append(stride_.minor() * dim_.minor()));
  }

  uintptr_t dptr_addr() {
    return mem_.dptr_addr();
  }

  Dim dim() const {
    return this->dim_;
  }
  Dim offset() const {
    return this->offset_;
  }
  Dim stride() const {
    return this->stride_;
  }

  size_t batch_capacity() const {
    return this->batch_cap_;
  }
  size_t batch_size() const {
    return this->batch_size_;
  }
  size_t set_batch_size(size_t new_batch_size) {
    assert(new_batch_size <= batch_cap_);
    this->batch_size_ = new_batch_size;
    return this->batch_size_;
  }

  /*void load_async(const T* src_ptr, size_t src_size, GPUConn& conn) {
    // FIXME: check packed shape.
    mem_.load_async(src_ptr, 0, src_size, conn);
  }

  void load_sync(const T* src_ptr, size_t src_size, GPUConn& conn) {
    // FIXME: check packed shape.
    mem_.load_sync(src_ptr, 0, src_size, conn);
  }

  void store_sync(T* dst_ptr, size_t dst_size, GPUConn& conn) {
    // FIXME: check packed shape.
    mem_.store_sync(dst_ptr, 0, dst_size, conn);
  }*/

  size_t load_any(size_t offset, any* reader, shared_ptr<Context> ctx) {
    //std::clog << "DEBUG: GPUBatchArray: load_any()" << std::endl;
    size_t max_buf_size = dim().flat_size() * batch_capacity();
    assert(NULL != reader);
    GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
    {
      auto buf_reader = any_cast<Slice<T>>(reader);
      if (NULL != buf_reader) {
        //std::clog << "DEBUG: GPUBatchArray:   slice reader: " << std::endl;
        assert(0 == offset);
        GPUConn conn = gpu_ctx->conn();
        size_t inferred_batch_size = buf_reader->size / dim().flat_size();
        assert(0 == buf_reader->size % dim().flat_size());
        assert(inferred_batch_size > 0);
        set_batch_size(inferred_batch_size);
        size_t buf_size = dim().flat_size() * inferred_batch_size;
        assert(buf_size == buf_reader->size);
        assert(buf_size <= max_buf_size);
        // TODO
        as_view().load_sync(buf_reader->ptr + offset, buf_size, conn);
        //as_view().load_async(buf_reader->data() + offset, buf_size, conn);
        return offset + buf_size;
      }
    }
    /*{
      auto buf_reader = any_cast<vector<T>>(reader);
      if (NULL != buf_reader) {
        std::clog << "DEBUG: GPUBatchArray:   vector reader: " << std::endl;
        assert(0 == offset);
        GPUConn conn = gpu_ctx->conn();
        size_t inferred_batch_size = buf_reader->size() / dim().flat_size();
        assert(0 == buf_reader->size() % dim().flat_size());
        set_batch_size(inferred_batch_size);
        size_t buf_size = dim().flat_size() * inferred_batch_size;
        assert(buf_size == buf_reader->size());
        assert(buf_size <= max_buf_size);
        // TODO
        as_view().load_sync(buf_reader->data() + offset, buf_size, conn);
        //as_view().load_async(buf_reader->data() + offset, buf_size, conn);
        return offset + buf_size;
      }
    }*/
    assert(0 && "unreachable");
  }

  size_t store_any(size_t offset, any* writer, shared_ptr<Context> ctx) {
    //std::clog << "DEBUG: GPUBatchArray: store_any()" << std::endl;
    size_t buf_size = dim().flat_size() * batch_size();
    //std::clog << "DEBUG: GPUBatchArray:   offset:   " << offset << std::endl;
    //std::clog << "DEBUG: GPUBatchArray:   buf size: " << buf_size << std::endl;
    if (NULL == writer) {
      //std::clog << "DEBUG: GPUBatchArray:   null writer: " << std::endl;
      return offset + buf_size;
    }
    {
      auto buf_writer = any_cast<T*>(writer);
      if (NULL != buf_writer) {
        //std::clog << "DEBUG: GPUBatchArray:   buf writer: " << std::endl;
        GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
        GPUConn conn = gpu_ctx->conn();
        //assert(buf_writer->size() >= buf_size);
        as_view().store_sync(*buf_writer + offset, buf_size, conn);
        return offset + buf_size;
      }
    }
    {
      auto buf_writer = any_cast<Slice<T>>(writer);
      if (NULL != buf_writer) {
        //std::clog << "DEBUG: GPUBatchArray:   slice writer: " << std::endl;
        GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
        GPUConn conn = gpu_ctx->conn();
        assert(buf_writer->size >= buf_size);
        as_view().store_sync(buf_writer->ptr + offset, buf_size, conn);
        return offset + buf_size;
      }
    }
    /*{
      auto buf_writer = any_cast<vector<T>>(writer);
      if (NULL != buf_writer) {
        std::clog << "DEBUG: GPUBatchArray:   vector writer: " << std::endl;
        GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
        GPUConn conn = gpu_ctx->conn();
        assert(buf_writer->size() >= buf_size);
        as_view().store_sync(buf_writer->data() + offset, buf_size, conn);
        return offset + buf_size;
      }
    }*/
    assert(0 && "unreachable");
  }

private:
  GPUMemory<T> mem_;
  Dim dim_;
  Dim offset_;
  Dim stride_;
  size_t offset_idx_;
  size_t batch_cap_;
  size_t batch_size_;
};

template <typename T>
using GPUArray0D = GPUArray<Shape0D, T>;

template <typename T>
using GPUArray1D = GPUArray<Shape1D, T>;

template <typename T>
using GPUArray2D = GPUArray<Shape2D, T>;

template <typename T>
using GPUArray3D = GPUArray<Shape3D, T>;

template <typename T>
using GPUArray4D = GPUArray<Shape4D, T>;

template <typename T>
using GPUBatchArray0D = GPUBatchArray<Shape0D, T>;

template <typename T>
using GPUBatchArray1D = GPUBatchArray<Shape1D, T>;

template <typename T>
using GPUBatchArray2D = GPUBatchArray<Shape2D, T>;

template <typename T>
using GPUBatchArray3D = GPUBatchArray<Shape3D, T>;

} // namespace arraydiff

#endif
