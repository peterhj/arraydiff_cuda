#include "arraydiff/op.hh"
#include "arraydiff/array_gpu.hh"
#include "arraydiff/context.hh"
#include "arraydiff/kernels.hh"
#include "arraydiff/spatial.hh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <nccl.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace arraydiff {

using std::function;
using std::make_shared;
using std::shared_ptr;
using std::vector;

void set_cudnn_tensor_desc_batch1d_ex(
    cudnnTensorDescriptor_t x_desc,
    Shape1D dim,
    Shape1D stride,
    size_t batch_size,
    cudnnDataType_t data_type,
    cudnnTensorFormat_t format)
{
  if (format == CUDNN_TENSOR_NCHW) {
    if (dim.is_packed(stride)) {
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          x_desc,
          format,
          data_type,
          batch_size,
          dim.shape,
          1,
          1));
    } else {
      assert(0 && "unimplemented");
    }
  } else {
    assert(0 && "unimplemented");
  }
}

void set_cudnn_tensor_desc_batch3d_ex(
    cudnnTensorDescriptor_t x_desc,
    Shape3D dim,
    Shape3D stride,
    size_t batch_size,
    cudnnDataType_t data_type,
    cudnnTensorFormat_t format)
{
  if (format == CUDNN_TENSOR_NCHW) {
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        x_desc,
        data_type,
        batch_size,
        dim.shape.at(2),
        dim.shape.at(1),
        dim.shape.at(0),
        stride.minor() * dim.minor(),
        stride.shape.at(2),
        stride.shape.at(1),
        stride.shape.at(0)));
  } else {
    assert(0 && "unimplemented");
  }
}

NodeId::NodeId() {
  static thread_local uint64_t id_ctr = 0;
  ++id_ctr;
  assert(id_ctr != 0);
  this->id_ = id_ctr;
}

EpochId::EpochId() {
  static thread_local uint64_t id_ctr = 0;
  ++id_ctr;
  assert(id_ctr != 0);
  this->id_ = id_ctr;
}

TxnId::TxnId() {
  static thread_local uint64_t id_ctr = 0;
  ++id_ctr;
  assert(id_ctr != 0);
  this->id_ = id_ctr;
}

VarId::VarId() {
  static thread_local uint64_t id_ctr = 0;
  ++id_ctr;
  assert(id_ctr != 0);
  this->id_ = id_ctr;
}

VarSymbol::VarSymbol() {
  static thread_local uint64_t id_ctr = 0;
  ++id_ctr;
  assert(id_ctr != 0);
  this->id_ = id_ctr;
}

OperatorStack::OperatorStack(NodeId node_id)
  : node_id_(node_id), in_degree_(0)
{
}

size_t OperatorStack::push(EpochId epoch) {
  if (entries_.size() != 0 && epoch.raw_id() == entries_[entries_.size() - 1].epoch.raw_id()) {
    entries_[entries_.size() - 1].push_count += 1;
  } else {
    OperatorStackEntry new_entry = {
      .epoch = epoch,
      .push_count = 1,
      .pop_count = 0
    };
    entries_.push_back(new_entry);
  }
  return entries_[entries_.size() - 1].push_count;
}

size_t OperatorStack::degree(EpochId epoch) {
  assert(entries_.size() != 0);
  assert(epoch.raw_id() == entries_[entries_.size() - 1].epoch.raw_id());
  return entries_[entries_.size() - 1].push_count;
}

size_t OperatorStack::pop(EpochId epoch) {
  assert(entries_.size() != 0);
  assert(epoch.raw_id() == entries_[entries_.size() - 1].epoch.raw_id());
  entries_[entries_.size() - 1].pop_count += 1;
  const size_t saved_pop_count = entries_[entries_.size() - 1].pop_count;
  if (entries_[entries_.size() - 1].push_count == entries_[entries_.size() - 1].pop_count) {
    entries_.pop_back();
  }
  return saved_pop_count;
}

size_t Op::load_val(TxnId txn, VarIdSet& vars, size_t offset, any* reader) {
  vars.lock();
  size_t new_offset = offset;
  EpochId epoch;
  this->_push(epoch, [](Op* _op) { (void)_op; });
  this->_pop(epoch, [txn, &vars, &reader, &new_offset](Op* op) {
    new_offset = op->_load_val(txn, vars, new_offset, reader);
  });
  vars.unlock();
  return new_offset;
}

size_t Op::load_grad(TxnId txn, VarIdSet& vars, size_t offset, any* reader) {
  vars.lock();
  size_t new_offset = offset;
  EpochId epoch;
  this->_push(epoch, [](Op* _op) { (void)_op; });
  this->_pop(epoch, [txn, &vars, &reader, &new_offset](Op* op) {
    new_offset = op->_load_grad(txn, vars, new_offset, reader);
  });
  vars.unlock();
  return new_offset;
}

size_t Op::store_val(TxnId txn, VarIdSet& vars, size_t offset, any* writer) {
  vars.lock();
  size_t new_offset = offset;
  EpochId epoch;
  this->_push(epoch, [](Op* _op) { (void)_op; });
  this->_pop(epoch, [txn, &vars, &writer, &new_offset](Op* op) {
    new_offset = op->_store_val(txn, vars, new_offset, writer);
  });
  vars.unlock();
  return new_offset;
}

size_t Op::store_grad(TxnId txn, VarIdSet& vars, size_t offset, any* writer) {
  vars.lock();
  size_t new_offset = offset;
  EpochId epoch;
  this->_push(epoch, [](Op* _op) { (void)_op; });
  this->_pop(epoch, [txn, &vars, &writer, &new_offset](Op* op) {
    new_offset = op->_store_grad(txn, vars, new_offset, writer);
  });
  vars.unlock();
  return new_offset;
}

void Op::init(TxnId txn, std::mt19937_64* rng) {
  EpochId epoch;
  this->_push(epoch, [txn, rng](Op* op) {
    op->_init(txn, rng);
  });
  this->_pop(epoch, [](Op* op) { (void)op; });
  //this->_sync();
}

void Op::persist(TxnId txn, VarIdSet& vars) {
  EpochId epoch;
  this->_push(epoch, [txn, &vars](Op* op) {
    op->_persist(txn, vars);
  });
  this->_pop(epoch, [](Op* op) { (void)op; });
  //this->_sync();
}

void Op::eval(TxnId txn) {
  EpochId epoch;
  this->_push(epoch, [txn](Op* op) {
    op->_forward(txn);
  });
  this->_pop(epoch, [](Op* op) { (void)op; });
  //this->_sync();
}

void Op::gradient(TxnId txn) {
  EpochId epoch;
  this->_push(epoch, [txn](Op* op) {
    op->_forward(txn);
  });
  this->_pop(epoch, [](Op* op) { (void)op; });
  this->_set_grad_sink(txn);
  this->_push(epoch, [](Op* op) { (void)op; });
  this->_pop(epoch, [txn](Op* op) {
    op->_backward(txn);
  });
}

template <>
shared_ptr<IOOp<GPUArray1D<float>>> array_src(Shape1D dim, shared_ptr<Context> ctx) {
  auto src = make_shared<SrcOp<GPUArray1D<float>>>(dim);
  auto io = make_shared<IOOp<GPUArray1D<float>>>(src, ctx);
  return io;
}

template <>
shared_ptr<IOOp<GPUArray2D<float>>> array_src(Shape2D dim, shared_ptr<Context> ctx) {
  auto src = make_shared<SrcOp<GPUArray2D<float>>>(dim);
  auto io = make_shared<IOOp<GPUArray2D<float>>>(src, ctx);
  return io;
}

template <>
shared_ptr<IOOp<GPUArray4D<float>>> array_src(Shape4D dim, shared_ptr<Context> ctx) {
  auto src = make_shared<SrcOp<GPUArray4D<float>>>(dim);
  auto io = make_shared<IOOp<GPUArray4D<float>>>(src, ctx);
  return io;
}

template <>
shared_ptr<IOOp<GPUBatchArray0D<uint32_t>>> batch_array_src(Shape0D dim, size_t batch_size, shared_ptr<Context> ctx) {
  auto src = make_shared<SrcOp<GPUBatchArray0D<uint32_t>>>(dim, batch_size);
  auto io = make_shared<IOOp<GPUBatchArray0D<uint32_t>>>(src, ctx);
  return io;
}

template <>
shared_ptr<IOOp<GPUBatchArray3D<uint8_t>>> batch_array_src(Shape3D dim, size_t batch_size, shared_ptr<Context> ctx) {
  auto src = make_shared<SrcOp<GPUBatchArray3D<uint8_t>>>(dim, batch_size);
  auto io = make_shared<IOOp<GPUBatchArray3D<uint8_t>>>(src, ctx);
  return io;
}

template <>
shared_ptr<IOOp<GPUBatchArray3D<float>>> batch_array_src(Shape3D dim, size_t batch_size, shared_ptr<Context> ctx) {
  auto src = make_shared<SrcOp<GPUBatchArray3D<float>>>(dim, batch_size);
  auto io = make_shared<IOOp<GPUBatchArray3D<float>>>(src, ctx);
  return io;
}

template <>
void zeros_init_fn(GPUArray1D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx) {
  GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
  GPUConn conn = gpu_ctx->conn();
  val->set_constant(0.0f, conn);
}

template <>
function<void(GPUArray1D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx)> zeros_init_gen() {
  return [](auto val, auto rng, auto ctx) {
    zeros_init_fn<GPUArray1D<float>>(val, rng, ctx);
  };
}

template <>
void ones_init_fn(GPUArray1D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx) {
  GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
  GPUConn conn = gpu_ctx->conn();
  val->set_constant(1.0f, conn);
}

template <>
function<void(GPUArray1D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx)> ones_init_gen() {
  return [](auto val, auto rng, auto ctx) {
    ones_init_fn<GPUArray1D<float>>(val, rng, ctx);
  };
}

template <>
void zeros_init_fn(GPUArray2D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx) {
  GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
  GPUConn conn = gpu_ctx->conn();
  val->set_constant(0.0f, conn);
}

template <>
function<void(GPUArray2D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx)> zeros_init_gen() {
  return [](auto val, auto rng, auto ctx) {
    zeros_init_fn<GPUArray2D<float>>(val, rng, ctx);
  };
}

template <>
void ones_init_fn(GPUArray2D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx) {
  GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
  GPUConn conn = gpu_ctx->conn();
  val->set_constant(1.0f, conn);
}

template <>
function<void(GPUArray2D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx)> ones_init_gen() {
  return [](auto val, auto rng, auto ctx) {
    ones_init_fn<GPUArray2D<float>>(val, rng, ctx);
  };
}

template <>
void zeros_init_fn(GPUArray4D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx) {
  GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
  GPUConn conn = gpu_ctx->conn();
  val->set_constant(0.0f, conn);
}

template <>
function<void(GPUArray4D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx)> zeros_init_gen() {
  return [](auto val, auto rng, auto ctx) {
    zeros_init_fn<GPUArray4D<float>>(val, rng, ctx);
  };
}

template <>
void ones_init_fn(GPUArray4D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx) {
  GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
  GPUConn conn = gpu_ctx->conn();
  val->set_constant(1.0f, conn);
}

template <>
function<void(GPUArray4D<float>* val, std::mt19937_64* rng, shared_ptr<Context> ctx)> ones_init_gen() {
  return [](auto val, auto rng, auto ctx) {
    ones_init_fn<GPUArray4D<float>>(val, rng, ctx);
  };
}

template <>
function<void(GPUArray2D<float>*, std::mt19937_64* rng, shared_ptr<Context> ctx)> xavier_init_gen(size_t fanin, size_t fanout) {
  return [fanin, fanout](auto val, auto rng, auto ctx) {
    std::clog << "DEBUG: xavier 2d init" << std::endl;
    std::uniform_real_distribution<float> dist(-std::sqrt(6.0f / (fanin + fanout)), std::sqrt(6.0f / (fanin + fanout)));
    size_t dim_size = val->dim().flat_size();
    vector<float> buf(dim_size);
    for (size_t idx = 0; idx < dim_size; ++idx) {
      buf.at(idx) = dist(*rng);
    }
    GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
    GPUConn conn = gpu_ctx->conn();
    val->load_sync(buf.data(), dim_size, conn);
  };
}

template <>
function<void(GPUArray4D<float>*, std::mt19937_64* rng, shared_ptr<Context> ctx)> xavier_init_gen(size_t fanin, size_t fanout) {
  return [fanin, fanout](auto val, auto rng, auto ctx) {
    std::clog << "DEBUG: xavier 4d init" << std::endl;
    std::uniform_real_distribution<float> dist(-std::sqrt(6.0f / (fanin + fanout)), std::sqrt(6.0f / (fanin + fanout)));
    size_t dim_size = val->dim().flat_size();
    vector<float> buf(dim_size);
    for (size_t idx = 0; idx < dim_size; ++idx) {
      buf.at(idx) = dist(*rng);
    }
    GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
    GPUConn conn = gpu_ctx->conn();
    val->load_sync(buf.data(), dim_size, conn);
  };
}

template <>
function<void(GPUArray4D<float>*, std::mt19937_64* rng, shared_ptr<Context> ctx)> kaiming_init_gen(size_t fanin, size_t fanout) {
  return [fanin](auto val, auto rng, auto ctx) {
    std::clog << "DEBUG: kaiming 4d init" << std::endl;
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fanin));
    size_t dim_size = val->dim().flat_size();
    vector<float> buf(dim_size);
    for (size_t idx = 0; idx < dim_size; ++idx) {
      buf.at(idx) = dist(*rng);
    }
    GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
    GPUConn conn = gpu_ctx->conn();
    val->load_sync(buf.data(), dim_size, conn);
  };
}

template <>
shared_ptr<InitOp<GPUArray1D<float>>> initialize(function<void(GPUArray1D<float>*, std::mt19937_64*, shared_ptr<Context> ctx)> init_fn, shared_ptr<Var<GPUArray1D<float>>> input, shared_ptr<Context> ctx) {
  auto init = make_shared<InitOp<GPUArray1D<float>>>(init_fn, input, ctx);
  return init;
}

template <>
shared_ptr<InitOp<GPUArray2D<float>>> initialize(function<void(GPUArray2D<float>*, std::mt19937_64*, shared_ptr<Context> ctx)> init_fn, shared_ptr<Var<GPUArray2D<float>>> input, shared_ptr<Context> ctx) {
  auto init = make_shared<InitOp<GPUArray2D<float>>>(init_fn, input, ctx);
  return init;
}

template <>
shared_ptr<InitOp<GPUArray4D<float>>> initialize(function<void(GPUArray4D<float>*, std::mt19937_64*, shared_ptr<Context> ctx)> init_fn, shared_ptr<Var<GPUArray4D<float>>> input, shared_ptr<Context> ctx) {
  auto init = make_shared<InitOp<GPUArray4D<float>>>(init_fn, input, ctx);
  return init;
}

template <>
void CastOp<GPUBatchArray3D<uint8_t>, GPUBatchArray3D<float>>::_forward(TxnId txn) {
  auto batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    /*std::clog << "DEBUG: CastOp:"
        << " x: " << "0x" << std::hex << x_.val().get(txn, id()).as_view().dptr_addr() << std::dec
        << " y: " << y_.val().get_excl(txn, id()).as_view().dptr()
        << std::endl;*/
    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    assert(y_dim.shape.at(0) == x_dim.shape.at(0));
    assert(y_dim.shape.at(1) == x_dim.shape.at(1));
    assert(y_dim.shape.at(2) == x_dim.shape.at(2));
    array_cast<uint8_t, float>(
        static_cast<uint32_t>(x_dim.flat_size() * batch_size),
        x_.val().get(txn, id()).as_view().dptr(),
        y_.val().get_excl(txn, id()).as_view().dptr(),
        conn.stream());
  }
}

template <>
void CastOp<GPUBatchArray3D<uint8_t>, GPUBatchArray3D<float>>::_backward(TxnId txn) {
}

template <>
Shape3D TransposeNHWCToNCHWOp<GPUBatchArray3D<float>, Shape3D>::_get_output_shape(Shape3D shape) const {
  return Shape3D(shape.shape.at(1), shape.shape.at(2), shape.shape.at(0));
}

template <>
void TransposeNHWCToNCHWOp<GPUBatchArray3D<float>, Shape3D>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    assert(y_dim.shape.at(0) == x_dim.shape.at(1));
    assert(y_dim.shape.at(1) == x_dim.shape.at(2));
    assert(y_dim.shape.at(2) == x_dim.shape.at(0));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        batch_size,
        x_dim.shape.at(0),
        x_dim.shape.at(2),
        x_dim.shape.at(1)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        y_dim.shape.at(2),
        y_dim.shape.at(1),
        y_dim.shape.at(0)));
    /*std::clog << "DEBUG: TransposeNHWCToNCHWOp:"
        << " x: " << x_.val().get(txn, id()).as_view().dptr()
        << " y: " << y_.val().get_excl(txn, id()).as_view().dptr()
        << std::endl;*/
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnTransformTensor(
        conn.cudnn(),
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
  }
}

template <>
void TransposeNHWCToNCHWOp<GPUBatchArray3D<float>, Shape3D>::_backward(TxnId txn) {
  // FIXME
}

template <>
void ScaleOp<float, GPUBatchArray0D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    array_scale_packed(
        static_cast<uint32_t>(batch_size),
        scale_,
        x_.val().get(txn, id()).as_view().dptr(),
        y_.val().get_excl(txn, id()).as_view().dptr(),
        conn.stream());
  }
}

template <>
void ScaleOp<float, GPUBatchArray0D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    auto conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();
    array_add_scaled_packed(
        static_cast<uint32_t>(batch_size),
        scale_,
        1.0f,
        y_.grad().get(txn, id()).as_view().dptr(),
        x_.grad().get_mut(txn, id()).as_view().dptr(),
        conn.stream());
  }
}

template <>
void ScaleOp<float, GPUBatchArray1D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    assert(x_dim.shape == y_dim.shape);
    array_scale_packed(
        static_cast<uint32_t>(x_dim.flat_size() * batch_size),
        scale_,
        x_.val().get(txn, id()).as_view().dptr(),
        y_.val().get_excl(txn, id()).as_view().dptr(),
        conn.stream());
  }
}

template <>
void ScaleOp<float, GPUBatchArray1D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    auto conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();
    auto y_dim = y_.val().get(txn, id()).dim();
    assert(x_dim.shape == y_dim.shape);
    array_add_scaled_packed(
        static_cast<uint32_t>(x_dim.flat_size() * batch_size),
        scale_,
        1.0f,
        y_.grad().get(txn, id()).as_view().dptr(),
        x_.grad().get_mut(txn, id()).as_view().dptr(),
        conn.stream());
  }
}

template <>
void ScaleOp<float, GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    assert(x_dim.shape == y_dim.shape);
    array_scale_packed(
        static_cast<uint32_t>(x_dim.flat_size() * batch_size),
        scale_,
        x_.val().get(txn, id()).as_view().dptr(),
        y_.val().get_excl(txn, id()).as_view().dptr(),
        conn.stream());
  }
}

template <>
void ScaleOp<float, GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    auto conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();
    auto y_dim = y_.val().get(txn, id()).dim();
    assert(x_dim.shape == y_dim.shape);
    array_add_scaled_packed(
        static_cast<uint32_t>(x_dim.flat_size() * batch_size),
        scale_,
        1.0f,
        y_.grad().get(txn, id()).as_view().dptr(),
        x_.grad().get_mut(txn, id()).as_view().dptr(),
        conn.stream());
  }
}

template <>
void JoinOp<GPUBatchArray1D<float>, SumJoin>::_forward(TxnId txn) {
  size_t batch_size = xs_.at(0).val().get(txn, id()).batch_size();
  auto x_dim = xs_.at(0).val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    assert(x_dim.shape == y_dim.shape);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        x_dim.shape,
        1,
        1));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        y_dim.shape,
        1,
        1));

    GPUConn conn = ctx->conn();

    const float alpha = 1.0f;
    const float first_beta = 0.0f;
    const float more_beta = 1.0f;
    assert(!in_place_);
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnAddTensor(
        conn.cudnn(),
        &alpha,
        x_desc_, xs_.at(0).val().get(txn, id()).as_view().dptr(),
        &first_beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
    for (size_t idx = 1; idx < xs_.size(); ++idx) {
      size_t more_batch_size = xs_.at(idx).val().get(txn, id()).batch_size();
      auto more_x_dim = xs_.at(idx).val().get(txn, id()).dim();
      assert(batch_size == more_batch_size);
      assert(x_dim.shape == more_x_dim.shape);
      CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
      CUDNN_CHECK(cudnnAddTensor(
          conn.cudnn(),
          &alpha,
          x_desc_, xs_.at(idx).val().get(txn, id()).as_view().dptr(),
          &more_beta,
          y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
    }
  }
}

template <>
void JoinOp<GPUBatchArray1D<float>, SumJoin>::_backward(TxnId txn) {
  size_t batch_size = xs_.at(0).val().get(txn, id()).batch_size();
  auto x_dim = xs_.at(0).val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  auto y_dim = y_.val().get(txn, id()).dim();
  assert(x_dim.shape == y_dim.shape);

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      x_desc_,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batch_size,
      x_dim.shape,
      1,
      1));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      y_desc_,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batch_size,
      y_dim.shape,
      1,
      1));

  GPUConn conn = ctx->conn();

  const float alpha = 1.0f;
  const float beta = 1.0f;
  assert(!in_place_);
  for (size_t idx = 0; idx < xs_.size(); ++idx) {
    /*if (in_place_ && idx == 0) {
      continue;
    }*/
    if (xs_.at(idx).grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
      GPUConn conn = ctx->conn();
      grad->set_batch_size(batch_size);
      grad->as_view().set_constant(0.0f, conn);
    })) {
      size_t more_batch_size = xs_.at(idx).val().get(txn, id()).batch_size();
      auto more_x_dim = xs_.at(idx).val().get(txn, id()).dim();
      assert(batch_size == more_batch_size);
      assert(x_dim.shape == more_x_dim.shape);
      CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
      CUDNN_CHECK(cudnnAddTensor(
          conn.cudnn(),
          &alpha,
          y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
          &beta,
          x_desc_, xs_.at(idx).grad().get_mut(txn, id()).as_view().dptr()));
    }
  }
}

template <>
void JoinOp<GPUBatchArray3D<float>, SumJoin>::_forward(TxnId txn) {
  size_t batch_size = xs_.at(0).val().get(txn, id()).batch_size();
  auto x_dim = xs_.at(0).val().get(txn, id()).dim();
  auto x_stride = xs_.at(0).val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride));

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    auto y_stride = y_.val().get_excl(txn, id()).stride();
    assert(x_dim.shape == y_dim.shape);
    assert(y_dim.is_packed(y_stride));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        x_dim.shape.at(2),
        x_dim.shape.at(1),
        x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        y_dim.shape.at(2),
        y_dim.shape.at(1),
        y_dim.shape.at(0)));

    GPUConn conn = ctx->conn();

    const float alpha = 1.0f;
    const float first_beta = 0.0f;
    const float more_beta = 1.0f;
    assert(!in_place_);
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnAddTensor(
        conn.cudnn(),
        &alpha,
        x_desc_, xs_.at(0).val().get(txn, id()).as_view().dptr(),
        &first_beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
    for (size_t idx = 1; idx < xs_.size(); ++idx) {
      size_t more_batch_size = xs_.at(idx).val().get(txn, id()).batch_size();
      auto more_x_dim = xs_.at(idx).val().get(txn, id()).dim();
      assert(batch_size == more_batch_size);
      assert(x_dim.shape == more_x_dim.shape);
      CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
      CUDNN_CHECK(cudnnAddTensor(
          conn.cudnn(),
          &alpha,
          x_desc_, xs_.at(idx).val().get(txn, id()).as_view().dptr(),
          &more_beta,
          y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
    }
  }
}

template <>
void JoinOp<GPUBatchArray3D<float>, SumJoin>::_backward(TxnId txn) {
  size_t batch_size = xs_.at(0).val().get(txn, id()).batch_size();
  auto x_dim = xs_.at(0).val().get(txn, id()).dim();
  auto x_stride = xs_.at(0).val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride));

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  auto y_dim = y_.val().get(txn, id()).dim();
  auto y_stride = y_.val().get(txn, id()).stride();
  assert(x_dim.shape == y_dim.shape);
  assert(y_dim.is_packed(y_stride));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      x_desc_,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batch_size,
      x_dim.shape.at(2),
      x_dim.shape.at(1),
      x_dim.shape.at(0)));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      y_desc_,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batch_size,
      y_dim.shape.at(2),
      y_dim.shape.at(1),
      y_dim.shape.at(0)));

  GPUConn conn = ctx->conn();

  const float alpha = 1.0f;
  const float beta = 1.0f;
  assert(!in_place_);
  for (size_t idx = 0; idx < xs_.size(); ++idx) {
    /*if (in_place_ && idx == 0) {
      continue;
    }*/
    if (xs_.at(idx).grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
      GPUConn conn = ctx->conn();
      grad->set_batch_size(batch_size);
      grad->as_view().set_constant(0.0f, conn);
    })) {
      size_t more_batch_size = xs_.at(idx).val().get(txn, id()).batch_size();
      auto more_x_dim = xs_.at(idx).val().get(txn, id()).dim();
      auto more_x_stride = xs_.at(idx).val().get(txn, id()).stride();
      assert(batch_size == more_batch_size);
      assert(x_dim.shape == more_x_dim.shape);
      assert(more_x_dim.is_packed(more_x_stride));
      CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
      CUDNN_CHECK(cudnnAddTensor(
          conn.cudnn(),
          &alpha,
          y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
          &beta,
          x_desc_, xs_.at(idx).grad().get_mut(txn, id()).as_view().dptr()));
    }
  }
}

template <>
Shape0D BatchJoinOp<GPUBatchArray0D<float>, GPUArray0D<float>, SumJoin>::_get_output_shape(Shape0D in_shape) {
  return Shape0D();
}

template <>
size_t BatchJoinOp<GPUBatchArray0D<float>, GPUArray0D<float>, SumJoin>::_get_output_batch_shape(size_t in_batch_shape) {
  return 0;
}

template <>
void BatchJoinOp<GPUBatchArray0D<float>, GPUArray0D<float>, SumJoin>::_set_grad_sink(TxnId txn) {
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.grad().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    y_.grad().get_excl(txn, id()).set_constant(1.0f, conn);
  }
}

template <>
void BatchJoinOp<GPUBatchArray0D<float>, GPUArray0D<float>, SumJoin>::_forward(TxnId txn) {
  /*size_t batch_size = x_.val().get(txn, id()).batch_size();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    sum_batch_join_fwd(
        batch_size,
        x_.val().get(txn, id()).dptr(),
        y_.val().get_excl(txn, id()).dptr(),
        conn.stream());
  }*/

  //size_t batch_size = x_.val().get(txn, id()).batch_size();
  //auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    //y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    //auto y_dim = y_.val().get_excl(txn, id()).dim();

    GPUConn conn = ctx->conn();

    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reduce_desc_,
        CUDNN_REDUCE_TENSOR_ADD,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES));

    x_.val().get(txn, id()).as_view()
      .set_cudnn_tensor_desc(x_desc_, CUDNN_TENSOR_NCHW);
    y_.val().get_excl(txn, id())
      .set_cudnn_tensor_desc(y_desc_, CUDNN_TENSOR_NCHW);

    size_t reduce_workspace_size = 0UL;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        conn.cudnn(),
        reduce_desc_,
        x_desc_,
        y_desc_,
        &reduce_workspace_size));
    ctx->reserve_scratch(reduce_workspace_size);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnReduceTensor(
        conn.cudnn(),
        reduce_desc_,
        NULL, 0, // indices.
        ctx->get_scratch()->dptr(), reduce_workspace_size, // workspace.
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_excl(txn, id()).dptr()));
  }
}

template <>
void BatchJoinOp<GPUBatchArray0D<float>, GPUArray0D<float>, SumJoin>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();

    x_.val().get(txn, id()).as_view()
      .set_cudnn_tensor_desc(x_desc_, CUDNN_TENSOR_NCHW);
    y_.val().get(txn, id())
      .set_cudnn_tensor_desc(y_desc_, CUDNN_TENSOR_NCHW);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnAddTensor(
        conn.cudnn(),
        &alpha,
        y_desc_, y_.grad().get(txn, id()).dptr(),
        &beta,
        x_desc_, x_.grad().get_mut(txn, id()).as_view().dptr()));
  }
}

template <>
Shape1D SpatialJoinOp<GPUBatchArray3D<float>, GPUBatchArray1D<float>, FlattenNCHWJoin>::_get_output_shape(Shape3D in_shape) {
  assert(1 == in_shape.shape.at(0));
  assert(1 == in_shape.shape.at(1));
  return Shape1D(in_shape.shape.at(2));
}

/*template <>
void SpatialJoinOp<GPUBatchArray2D<float>, GPUBatchArray0D<float>, SumJoin>::_set_grad_sink(TxnId txn) {
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.grad().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    y_.grad().get_excl(txn, id()).as_view().set_constant(1.0f, conn);
  }
}

template <>
void SpatialJoinOp<GPUBatchArray3D<float>, GPUBatchArray1D<float>, FlattenNCHWJoin>::_set_grad_sink(TxnId txn) {
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.grad().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    y_.grad().get_excl(txn, id()).as_view().set_constant(1.0f, conn);
  }
}*/

template <>
void SpatialJoinOp<GPUBatchArray3D<float>, GPUBatchArray1D<float>, FlattenNCHWJoin>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    assert(1 == x_dim.shape.at(0));
    assert(1 == x_dim.shape.at(1));
    assert(y_dim.shape == x_dim.shape.at(2));

    /*x_.val().get(txn, id()).as_view()
      .set_cudnn_tensor_desc(x_desc_, CUDNN_TENSOR_NCHW);
    y_.val().get_excl(txn, id()).as_view()
      .set_cudnn_tensor_desc(y_desc_, CUDNN_TENSOR_NCHW);*/

    set_cudnn_tensor_desc_batch3d_ex(
        x_desc_,
        x_.val().get(txn, id()).dim(),
        x_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);
    set_cudnn_tensor_desc_batch1d_ex(
        y_desc_,
        y_.val().get_excl(txn, id()).dim(),
        y_.val().get_excl(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);

    GPUConn conn = ctx->conn();

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnAddTensor(
        conn.cudnn(),
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
  }
}

template <>
void SpatialJoinOp<GPUBatchArray3D<float>, GPUBatchArray1D<float>, FlattenNCHWJoin>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    auto y_dim = y_.val().get(txn, id()).dim();
    assert(1 == x_dim.shape.at(0));
    assert(1 == x_dim.shape.at(1));
    assert(y_dim.shape == x_dim.shape.at(2));

    /*x_.val().get(txn, id()).as_view()
      .set_cudnn_tensor_desc(x_desc_, CUDNN_TENSOR_NCHW);
    y_.val().get(txn, id()).as_view()
      .set_cudnn_tensor_desc(y_desc_, CUDNN_TENSOR_NCHW);*/

    set_cudnn_tensor_desc_batch3d_ex(
        x_desc_,
        x_.val().get(txn, id()).dim(),
        x_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);
    set_cudnn_tensor_desc_batch1d_ex(
        y_desc_,
        y_.val().get(txn, id()).dim(),
        y_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);

    GPUConn conn = ctx->conn();

    const float alpha = 1.0f;
    const float beta = 1.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnAddTensor(
        conn.cudnn(),
        &alpha,
        y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
        &beta,
        x_desc_, x_.grad().get_mut(txn, id()).as_view().dptr()));
  }
}

template <>
void SpatialJoinOp<GPUBatchArray2D<float>, GPUBatchArray0D<float>, SumJoin>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  //auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    //auto y_dim = y_.val().get_excl(txn, id()).dim();

    GPUConn conn = ctx->conn();

    CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
        reduce_desc_,
        CUDNN_REDUCE_TENSOR_ADD,
        CUDNN_DATA_FLOAT,
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES,
        CUDNN_32BIT_INDICES));

    x_.val().get(txn, id()).as_view()
      .set_cudnn_tensor_desc(x_desc_, CUDNN_TENSOR_NCHW);
    y_.val().get_excl(txn, id()).as_view()
      .set_cudnn_tensor_desc(y_desc_, CUDNN_TENSOR_NCHW);

    size_t reduce_workspace_size = 0UL;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
        conn.cudnn(),
        reduce_desc_,
        x_desc_,
        y_desc_,
        &reduce_workspace_size));
    ctx->reserve_scratch(reduce_workspace_size);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnReduceTensor(
        conn.cudnn(),
        reduce_desc_,
        NULL, 0, // indices.
        ctx->get_scratch()->dptr(), reduce_workspace_size, // workspace.
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
  }
}

template <>
void SpatialJoinOp<GPUBatchArray2D<float>, GPUBatchArray0D<float>, SumJoin>::_backward(TxnId txn) {
  // FIXME
}

/*template <>
void BranchOp<bool, GPUArray1D<float>>::_forward(TxnId txn) {
  // FIXME
  size_t x_dim = 0; // FIXME: get from `x`.
  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    if (!test_.val().get(txn, id())) {
      y_.val().get_excl(txn, id()).copy(x_off_.val().get(txn, id()), conn);
    } else {
      y_.val().get_excl(txn, id()).copy(x_on_.val().get(txn, id()), conn);
    }
  }
}*/

/*template <>
void CublasLinearOp<GPUBatchArray1D<float>, GPUArray2D<float>, GPUArray1D<float>>::_set_grad_sink(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.grad().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();
    y_.grad().get_excl(txn, id()).as_view().set_constant(0.0f, conn);
  }
}*/

template <>
Shape1D CublasLinearOp<GPUBatchArray1D<float>, GPUArray2D<float>, GPUArray1D<float>>::_get_output_shape(Shape1D in_shape) {
  return Shape1D(filters_);
}

template <>
void CublasLinearOp<GPUBatchArray1D<float>, GPUArray2D<float>, GPUArray1D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  const float alpha = 1.0f;
  const float gemm_beta = 0.0f;
  const float bias_beta = 1.0f;

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();

    //std::clog << "DEBUG: LinearOp:"
    //    " batch sz: " << batch_size <<
    //    " x dim: " << x_dim.shape <<
    //    " y dim: " << y_dim.shape << std::endl;

    CUBLAS_CHECK(cublasSetStream(conn.cublas(), conn.stream()));
    CUBLAS_CHECK(cublasSetPointerMode(conn.cublas(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSgemm(
        conn.cublas(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        y_dim.shape, batch_size, x_dim.shape,
        &alpha,
        a_.val().get(txn, id()).dptr(), y_dim.shape,
        x_.val().get(txn, id()).as_view().dptr(), x_dim.shape,
        &gemm_beta,
        y_.val().get_excl(txn, id()).as_view().dptr(), y_dim.shape));

    if (static_cast<bool>(b_)) {
      b_.value().val().get(txn, id())
        .set_cudnn_tensor_desc(bias_desc_, CUDNN_TENSOR_NCHW);
      y_.val().get_excl(txn, id()).as_view()
        .set_cudnn_tensor_desc(y_desc_, CUDNN_TENSOR_NCHW);
      CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
      CUDNN_CHECK(cudnnAddTensor(
          conn.cudnn(),
          &alpha,
          bias_desc_, b_.value().val().get(txn, id()).dptr(),
          &bias_beta,
          y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
    }
  }
}

template <>
void CublasLinearOp<GPUBatchArray1D<float>, GPUArray2D<float>, GPUArray1D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  auto y_dim = y_.val().get(txn, id()).dim();

  const float alpha = 1.0f;
  const float beta = 1.0f;

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (static_cast<bool>(b_)) {
    if (b_.value().grad().accumulate(txn, id(), ctx, [ctx](auto grad) {
      GPUConn conn = ctx->conn();
      grad->set_constant(0.0f, conn);
    }))
    {
      GPUConn conn = ctx->conn();

      b_.value().val().get(txn, id())
        .set_cudnn_tensor_desc(bias_desc_, CUDNN_TENSOR_NCHW);
      y_.val().get(txn, id()).as_view()
        .set_cudnn_tensor_desc(y_desc_, CUDNN_TENSOR_NCHW);

      CUDNN_CHECK(cudnnSetReduceTensorDescriptor(
          bias_grad_reduce_desc_,
          CUDNN_REDUCE_TENSOR_ADD,
          CUDNN_DATA_FLOAT,
          CUDNN_NOT_PROPAGATE_NAN,
          CUDNN_REDUCE_TENSOR_NO_INDICES,
          CUDNN_32BIT_INDICES));

      size_t reduce_workspace_size = 0UL;
      CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
      CUDNN_CHECK(cudnnGetReductionWorkspaceSize(
          conn.cudnn(),
          bias_grad_reduce_desc_,
          y_desc_,
          bias_desc_,
          &reduce_workspace_size));
      ctx->reserve_scratch(reduce_workspace_size);

      CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
      CUDNN_CHECK(cudnnReduceTensor(
          conn.cudnn(),
          bias_grad_reduce_desc_,
          NULL, 0, // indices.
          ctx->get_scratch()->dptr(), reduce_workspace_size, // workspace.
          &alpha,
          y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
          &beta,
          bias_desc_, b_.value().grad().get_mut(txn, id()).dptr()));
    }
  }

  if (a_.grad().accumulate(txn, id(), ctx, [ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_constant(0.0f, conn);
  }))
  {
    GPUConn conn = ctx->conn();
    CUBLAS_CHECK(cublasSetStream(conn.cublas(), conn.stream()));
    CUBLAS_CHECK(cublasSetPointerMode(conn.cublas(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSgemm(
        conn.cublas(),
        CUBLAS_OP_N, CUBLAS_OP_T,
        y_dim.shape, x_dim.shape, batch_size,
        &alpha,
        y_.grad().get(txn, id()).as_view().dptr(), y_dim.shape,
        x_.val().get(txn, id()).as_view().dptr(), x_dim.shape,
        &beta,
        a_.grad().get_mut(txn, id()).dptr(), y_dim.shape));
  }

  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  }))
  {
    GPUConn conn = ctx->conn();
    CUBLAS_CHECK(cublasSetStream(conn.cublas(), conn.stream()));
    CUBLAS_CHECK(cublasSetPointerMode(conn.cublas(), CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSgemm(
        conn.cublas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        x_dim.shape, batch_size, y_dim.shape,
        &alpha,
        a_.val().get(txn, id()).dptr(), y_dim.shape,
        y_.grad().get(txn, id()).as_view().dptr(), y_dim.shape,
        &beta,
        x_.grad().get_mut(txn, id()).as_view().dptr(), x_dim.shape));
  }
}

CudnnConv2DSize::CudnnConv2DSize(Conv2DShape shape, /*size_t halo_pad,*/ Shape3D x_dim, size_t batch_size, GPUContext* ctx) {
  this->batch_size = batch_size;
  this->in_shape = x_dim;

  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  if (shape.enable_hmma) {
    std::clog << "DEBUG: Conv2DSize: enable hmma (tensor core math)" << std::endl;
    //CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
  }

  if (shape.halo_padded_input) {
    std::clog << "DEBUG: Conv2DSize: halo padded convolution (explicit padding)" << std::endl;
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        0,                  0,
        shape.stride.at(1), shape.stride.at(0),
        shape.dilate.at(1), shape.dilate.at(0),
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
  } else {
    std::clog << "DEBUG: Conv2DSize: ordinary convolution (implicit zero padding)" << std::endl;
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        shape.pad.at(1),    shape.pad.at(0),
        shape.stride.at(1), shape.stride.at(0),
        shape.dilate.at(1), shape.dilate.at(0),
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
  }

  if (shape.enable_hmma) {
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
  }

  CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(
      kernel_desc,
      CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW,
      shape.filters,
      x_dim.shape.at(2),
      shape.kernel.at(1),
      shape.kernel.at(0)));

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      bias_desc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      1, shape.filters, 1, 1));

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
  if (shape.halo_padded_input) {
    Shape3D x_padded_dim(
        x_dim.shape.at(0) + shape.pad.at(0) * 2,
        x_dim.shape.at(1) + shape.pad.at(1) * 2,
        x_dim.shape.at(2));
    std::clog << "DEBUG: Conv2DSize:   halo padded input shape: "
        << x_padded_dim.shape.at(0) << " " << x_padded_dim.shape.at(1) << " " << x_padded_dim.shape.at(2) << std::endl;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        x_padded_dim.shape.at(2),
        x_padded_dim.shape.at(1),
        x_padded_dim.shape.at(0)));
  } else {
    std::clog << "DEBUG: Conv2DSize:   ordinary input shape: "
        << x_dim.shape.at(0) << " " << x_dim.shape.at(1) << " " << x_dim.shape.at(2) << std::endl;
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        x_dim.shape.at(2),
        x_dim.shape.at(1),
        x_dim.shape.at(0)));
  }

  int y_n = 0;
  int y_c = 0;
  int y_h = 0;
  int y_w = 0;
  CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
      conv_desc,
      x_desc,
      kernel_desc,
      &y_n,
      &y_c,
      &y_h,
      &y_w));
  assert(static_cast<size_t>(y_n) == batch_size);
  Shape3D y_dim(y_w, y_h, y_c);
  this->out_shape = y_dim;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
  std::clog << "DEBUG: Conv2DSize:   ordinary output shape: "
      << y_dim.shape.at(0) << " " << y_dim.shape.at(1) << " " << y_dim.shape.at(2) << std::endl;
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      y_desc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batch_size,
      y_dim.shape.at(2),
      y_dim.shape.at(1),
      y_dim.shape.at(0)));
  this->out_offset = Shape3D::Zero();
  this->out_stride = y_dim.packed_stride();

  this->max_workspace_size = 0;

  GPUConn conn = ctx->conn();

  int fwd_algo_count = 0;
  CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
  if (shape.enable_hmma) {
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
  }
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
      conn.cudnn(),
      x_desc,
      kernel_desc,
      conv_desc,
      y_desc,
      1,
      &fwd_algo_count,
      &fwd_algo));
  assert(1 == fwd_algo_count);
  this->max_workspace_size = std::max(this->max_workspace_size, fwd_algo.memory);

  int bwd_kernel_algo_count = 0;
  CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
  if (shape.enable_hmma) {
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
  }
  CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
      conn.cudnn(),
      x_desc,
      y_desc,
      conv_desc,
      kernel_desc,
      1,
      &bwd_kernel_algo_count,
      &bwd_kernel_algo));
  assert(1 == bwd_kernel_algo_count);
  this->max_workspace_size = std::max(this->max_workspace_size, bwd_kernel_algo.memory);

  int bwd_input_algo_count = 0;
  CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
  if (shape.enable_hmma) {
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
  }
  CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
      conn.cudnn(),
      kernel_desc,
      y_desc,
      conv_desc,
      x_desc,
      1,
      &bwd_input_algo_count,
      &bwd_input_algo));
  assert(1 == bwd_input_algo_count);
  this->max_workspace_size = std::max(this->max_workspace_size, bwd_input_algo.memory);

  ctx->reserve_scratch(this->max_workspace_size);
}

template <>
Shape3D CudnnConv2DOp<GPUBatchArray3D<float>, GPUArray4D<float>, GPUArray1D<float>>::_get_output_shape(Shape3D in_dim, size_t batch_size) {
  //std::clog << "DEBUG: Conv2D: get output shape: in shape: " << in_shape.shape.at(0) << " " << in_shape.shape.at(1) << " " << in_shape.shape.at(2) << std::endl;
  if (conv_sizes_.find(batch_size) == conv_sizes_.end()) {
    GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
    conv_sizes_.emplace(batch_size, CudnnConv2DSize(shape_, in_dim, batch_size, ctx));
  }
  CudnnConv2DSize& conv_size = conv_sizes_.at(batch_size);
  assert(in_dim.shape == conv_size.in_shape.shape);
  return conv_size.out_shape;
}

template <>
Shape3D CudnnConv2DOp<GPUBatchArray3D<float>, GPUArray4D<float>, GPUArray1D<float>>::_get_output_offset(Shape3D in_dim, size_t batch_size) {
  if (conv_sizes_.find(batch_size) == conv_sizes_.end()) {
    GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
    conv_sizes_.emplace(batch_size, CudnnConv2DSize(shape_, in_dim, batch_size, ctx));
  }
  CudnnConv2DSize& conv_size = conv_sizes_.at(batch_size);
  assert(in_dim.shape == conv_size.in_shape.shape);
  return conv_size.out_offset;
}

template <>
Shape3D CudnnConv2DOp<GPUBatchArray3D<float>, GPUArray4D<float>, GPUArray1D<float>>::_get_output_stride(Shape3D in_dim, size_t batch_size) {
  if (conv_sizes_.find(batch_size) == conv_sizes_.end()) {
    GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
    conv_sizes_.emplace(batch_size, CudnnConv2DSize(shape_, in_dim, batch_size, ctx));
  }
  CudnnConv2DSize& conv_size = conv_sizes_.at(batch_size);
  assert(in_dim.shape == conv_size.in_shape.shape);
  return conv_size.out_stride;
}

template <>
void CudnnConv2DOp<GPUBatchArray3D<float>, GPUArray4D<float>, GPUArray1D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  const float alpha = 1.0f;
  const float conv_beta = 0.0f;
  const float bias_beta = 1.0f;

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  // TODO: fused ConvBiasAct op.

  if (y_.val().overwrite(txn, id(), ctx)) {
    if (conv_sizes_.find(batch_size) == conv_sizes_.end()) {
      conv_sizes_.emplace(batch_size, CudnnConv2DSize(shape_, x_dim, batch_size, ctx));
    }
    CudnnConv2DSize& conv_size = conv_sizes_.at(batch_size);
    if (conv_size.max_workspace_size > workspace_size_) {
      workspace_size_ = conv_size.max_workspace_size;
    }
    ctx->reserve_scratch(workspace_size_);

    GPUConn conn = ctx->conn();

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    if (shape_.enable_hmma) {
      CUDNN_CHECK(cudnnSetConvolutionMathType(conv_size.conv_desc, CUDNN_TENSOR_OP_MATH));
    }
    if (shape_.halo_padded_input) {
      CUDNN_CHECK(cudnnConvolutionForward(
          conn.cudnn(),
          &alpha,
          conv_size.x_desc, x_.val().get(txn, id()).base_as_view().dptr(),
          conv_size.kernel_desc, kernel_.val().get(txn, id()).dptr(),
          conv_size.conv_desc,
          conv_size.fwd_algo.algo,
          ctx->get_scratch()->dptr(),
          conv_size.fwd_algo.memory,
          &conv_beta,
          conv_size.y_desc, y_.val().get_excl(txn, id()).as_view().dptr()));
    } else {
      CUDNN_CHECK(cudnnConvolutionForward(
          conn.cudnn(),
          &alpha,
          conv_size.x_desc, x_.val().get(txn, id()).as_view().dptr(),
          conv_size.kernel_desc, kernel_.val().get(txn, id()).dptr(),
          conv_size.conv_desc,
          conv_size.fwd_algo.algo,
          ctx->get_scratch()->dptr(),
          conv_size.fwd_algo.memory,
          &conv_beta,
          conv_size.y_desc, y_.val().get_excl(txn, id()).as_view().dptr()));
    }

    if (static_cast<bool>(bias_)) {
      CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
      CUDNN_CHECK(cudnnAddTensor(
          conn.cudnn(),
          &alpha,
          conv_size.bias_desc, bias_.value().val().get(txn, id()).dptr(),
          &bias_beta,
          conv_size.y_desc, y_.val().get_excl(txn, id()).as_view().dptr()));
    }
  }
}

template <>
void CudnnConv2DOp<GPUBatchArray3D<float>, GPUArray4D<float>, GPUArray1D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  const float alpha = 1.0f;
  const float beta = 1.0f;

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (conv_sizes_.find(batch_size) == conv_sizes_.end()) {
    conv_sizes_.emplace(batch_size, CudnnConv2DSize(shape_, x_dim, batch_size, ctx));
  }
  CudnnConv2DSize& conv_size = conv_sizes_.at(batch_size);
  if (conv_size.max_workspace_size > workspace_size_) {
    workspace_size_ = conv_size.max_workspace_size;
  }
  ctx->reserve_scratch(workspace_size_);

  if (static_cast<bool>(bias_)) {
    if (bias_.value().grad().accumulate(txn, id(), ctx, [ctx](auto grad) {
      GPUConn conn = ctx->conn();
      grad->set_constant(0.0f, conn);
    })) {
      GPUConn conn = ctx->conn();
      CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
      if (shape_.enable_hmma) {
        CUDNN_CHECK(cudnnSetConvolutionMathType(conv_size.conv_desc, CUDNN_TENSOR_OP_MATH));
      }
      CUDNN_CHECK(cudnnConvolutionBackwardBias(
          conn.cudnn(),
          &alpha,
          conv_size.y_desc, y_.grad().get(txn, id()).as_view().dptr(),
          &beta,
          conv_size.bias_desc, bias_.value().grad().get_mut(txn, id()).dptr()));
    }
  }

  if (kernel_.grad().accumulate(txn, id(), ctx, [ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    if (shape_.enable_hmma) {
      CUDNN_CHECK(cudnnSetConvolutionMathType(conv_size.conv_desc, CUDNN_TENSOR_OP_MATH));
    }
    if (shape_.halo_padded_input) {
      CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          conn.cudnn(),
          &alpha,
          conv_size.x_desc, x_.val().get(txn, id()).base_as_view().dptr(),
          conv_size.y_desc, y_.grad().get(txn, id()).as_view().dptr(),
          conv_size.conv_desc,
          conv_size.bwd_kernel_algo.algo,
          ctx->get_scratch()->dptr(),
          conv_size.bwd_kernel_algo.memory,
          &beta,
          conv_size.kernel_desc, kernel_.grad().get_mut(txn, id()).dptr()));
    } else {
      CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          conn.cudnn(),
          &alpha,
          conv_size.x_desc, x_.val().get(txn, id()).as_view().dptr(),
          conv_size.y_desc, y_.grad().get(txn, id()).as_view().dptr(),
          conv_size.conv_desc,
          conv_size.bwd_kernel_algo.algo,
          ctx->get_scratch()->dptr(),
          conv_size.bwd_kernel_algo.memory,
          &beta,
          conv_size.kernel_desc, kernel_.grad().get_mut(txn, id()).dptr()));
    }
  }

  if (x_.grad().accumulate(txn, id(), ctx, [this, batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    if (this->shape_.halo_padded_input) {
      grad->base_as_view().set_constant(0.0f, conn);
    } else {
      grad->as_view().set_constant(0.0f, conn);
    }
  })) {
    GPUConn conn = ctx->conn();
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    if (shape_.enable_hmma) {
      CUDNN_CHECK(cudnnSetConvolutionMathType(conv_size.conv_desc, CUDNN_TENSOR_OP_MATH));
    }
    if (shape_.halo_padded_input) {
      CUDNN_CHECK(cudnnConvolutionBackwardData(
          conn.cudnn(),
          &alpha,
          conv_size.kernel_desc, kernel_.val().get(txn, id()).dptr(),
          conv_size.y_desc, y_.grad().get(txn, id()).as_view().dptr(),
          conv_size.conv_desc,
          conv_size.bwd_input_algo.algo,
          ctx->get_scratch()->dptr(),
          conv_size.bwd_input_algo.memory,
          &beta,
          conv_size.x_desc, x_.grad().get_mut(txn, id()).base_as_view().dptr()));
    } else {
      CUDNN_CHECK(cudnnConvolutionBackwardData(
          conn.cudnn(),
          &alpha,
          conv_size.kernel_desc, kernel_.val().get(txn, id()).dptr(),
          conv_size.y_desc, y_.grad().get(txn, id()).as_view().dptr(),
          conv_size.conv_desc,
          conv_size.bwd_input_algo.algo,
          ctx->get_scratch()->dptr(),
          conv_size.bwd_input_algo.memory,
          &beta,
          conv_size.x_desc, x_.grad().get_mut(txn, id()).as_view().dptr()));
    }
  }
}

template <>
void BroadcastOp<GPUArray1D<float>>::_forward(TxnId txn) {
  auto x_dim = x_.val().get(txn, id()).dim();
  auto x_stride = x_.val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "BroadcastOp only supports packed inputs");

  size_t num_writes = 0;

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());

  {
    auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);
    if (ys_.at(root_rank_).val().overwrite(txn, id(), root_dev_ctx.get())) {
      auto conn = root_dev_ctx->conn();
      ys_.at(root_rank_).val().get_excl(txn, id())
        .copy(x_.val().get(txn, id()), conn);
      ++num_writes;
    }
  }

  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    if (rank == root_rank_) {
      continue;
    }
    auto dev_ctx = mgpu_ctx->device_context(rank);
    if (ys_.at(rank).val().overwrite(txn, id(), dev_ctx.get())) {
      ++num_writes;
    }
  }

  assert(num_writes == 0 || num_writes == mgpu_ctx->num_ranks());
  if (num_writes > 0) {
    mgpu_ctx->sync();
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupStart());
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      NCCL_CHECK(ncclBcast(
          ys_.at(rank).val().get_excl(txn, id()).dptr(),
          ys_.at(rank).val().get_excl(txn, id()).dim().shape,
          ncclFloat,
          root_rank_,
          mgpu_ctx->nccl_comm(rank),
          conn.stream()));
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupEnd());
    }
    mgpu_ctx->sync();
  }
}

template <>
void BroadcastOp<GPUArray1D<float>>::_backward(TxnId txn) {
  auto x_dim = x_.val().get(txn, id()).dim();
  auto x_stride = x_.val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "BroadcastOp only supports packed inputs");

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
  auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);

  if (x_.grad().accumulate(txn, id(), root_dev_ctx.get(), [root_dev_ctx](auto grad) {
    auto conn = root_dev_ctx->conn();
    grad->set_constant(0.0f, conn);
  })) {
    {
      auto conn = root_dev_ctx->conn();
      x_.grad().get_mut(txn, id())
        .add(ys_.at(root_rank_).grad().get(txn, id()), conn);
    }

    mgpu_ctx->sync();
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupStart());
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      if (rank == root_rank_) {
        NCCL_CHECK(ncclReduce(
            x_.grad().get_mut(txn, id()).dptr(),
            x_.grad().get_mut(txn, id()).dptr(),
            x_.grad().get_mut(txn, id()).dim().shape,
            ncclFloat,
            ncclSum,
            root_rank_,
            mgpu_ctx->nccl_comm(rank),
            conn.stream()));
      } else {
        NCCL_CHECK(ncclReduce(
            ys_.at(rank).grad().get(txn, id()).dptr(),
            NULL,
            ys_.at(rank).grad().get(txn, id()).dim().shape,
            ncclFloat,
            ncclSum,
            root_rank_,
            mgpu_ctx->nccl_comm(rank),
            conn.stream()));
      }
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupEnd());
    }
    mgpu_ctx->sync();
  }
}

template <>
void BroadcastOp<GPUArray4D<float>>::_forward(TxnId txn) {
  auto x_dim = x_.val().get(txn, id()).dim();
  auto x_stride = x_.val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "BroadcastOp only supports packed inputs");

  // FIXME
  assert(0 && "unimplemented");
}

template <>
void BroadcastOp<GPUArray4D<float>>::_backward(TxnId txn) {
  // FIXME
  assert(0 && "unimplemented");
}

template <>
void BroadcastOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  auto x_stride = x_.val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "BroadcastOp only supports packed inputs");

  size_t num_writes = 0;

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());

  {
    auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);
    if (ys_.at(root_rank_).val().overwrite(txn, id(), root_dev_ctx.get())) {
      ys_.at(root_rank_).val().get_excl(txn, id()).set_batch_size(batch_size);
      auto conn = root_dev_ctx->conn();
      ys_.at(root_rank_).val().get_excl(txn, id()).as_view()
        .copy(x_.val().get(txn, id()).as_view(), conn);
      ++num_writes;
    }
  }

  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    if (rank == root_rank_) {
      continue;
    }
    auto dev_ctx = mgpu_ctx->device_context(rank);
    if (ys_.at(rank).val().overwrite(txn, id(), dev_ctx.get())) {
      ys_.at(rank).val().get_excl(txn, id()).set_batch_size(batch_size);
      ++num_writes;
    }
  }

  assert(num_writes == 0 || num_writes == mgpu_ctx->num_ranks());
  if (num_writes > 0) {
    mgpu_ctx->sync();
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupStart());
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      //std::clog << "DEBUG: BroadcastOp: forward: call ncclBcast" << std::endl;
      auto dev_ctx = mgpu_ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      NCCL_CHECK(ncclBcast(
          ys_.at(rank).val().get_excl(txn, id()).as_view().dptr(),
          ys_.at(rank).val().get_excl(txn, id()).as_view().dim().flat_size(),
          ncclFloat,
          root_rank_,
          mgpu_ctx->nccl_comm(rank),
          conn.stream()));
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupEnd());
    }
    mgpu_ctx->sync();
  }
}

template <>
void BroadcastOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  auto x_stride = x_.val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "BroadcastOp only supports packed inputs");

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
  auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);

  if (x_.grad().accumulate(txn, id(), root_dev_ctx.get(), [batch_size, root_dev_ctx](auto grad) {
    auto conn = root_dev_ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    mgpu_ctx->sync();
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupStart());
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      //std::clog << "DEBUG: BroadcastOp: backward: call ncclReduce" << std::endl;
      assert(batch_size == ys_.at(rank).grad().get(txn, id()).batch_size());
      assert(batch_size == x_.grad().get_mut(txn, id()).batch_size());
      assert(ys_.at(rank).grad().get(txn, id()).dim().shape == x_.grad().get_mut(txn, id()).dim().shape);
      auto dev_ctx = mgpu_ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      // FIXME: reduce to a temporary before adding to dx.
      if (rank == root_rank_) {
        NCCL_CHECK(ncclReduce(
            ys_.at(root_rank_).grad().get(txn, id()).as_view().dptr(),
            x_.grad().get_mut(txn, id()).as_view().dptr(),
            x_.grad().get_mut(txn, id()).as_view().dim().flat_size(),
            ncclFloat,
            ncclSum,
            root_rank_,
            mgpu_ctx->nccl_comm(rank),
            conn.stream()));
      } else {
        NCCL_CHECK(ncclReduce(
            ys_.at(rank).grad().get(txn, id()).as_view().dptr(),
            NULL,
            ys_.at(rank).grad().get(txn, id()).as_view().dim().flat_size(),
            ncclFloat,
            ncclSum,
            root_rank_,
            mgpu_ctx->nccl_comm(rank),
            conn.stream()));
      }
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupEnd());
    }
    mgpu_ctx->sync();
  }
}

template <>
void ReduceOp<GPUBatchArray1D<float>>::_forward(TxnId txn) {
  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
  auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);

  if (y_.val().overwrite(txn, id(), root_dev_ctx.get())) {
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    auto y_stride = y_.val().get_excl(txn, id()).stride();
    assert(y_dim.is_packed(y_stride) && "ReduceOp only supports packed outputs");
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
      auto x_dim = xs_.at(rank).val().get(txn, id()).dim();
      auto x_stride = xs_.at(rank).val().get(txn, id()).stride();
      assert(x_dim.is_packed(x_stride) && "ReduceOp only supports packed inputs");
      if (0 == rank) {
        y_.val().get_excl(txn, id()).set_batch_size(batch_size);
      } else {
        assert(batch_size == y_.val().get_excl(txn, id()).batch_size());
      }
      assert(x_dim.shape == y_dim.shape);
    }

    mgpu_ctx->sync();
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupStart());
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      NCCL_CHECK(ncclReduce(
          xs_.at(rank).val().get(txn, id()).as_view().dptr(),
          y_.val().get_excl(txn, id()).as_view().dptr(),
          y_.val().get_excl(txn, id()).as_view().dim().flat_size(),
          ncclFloat,
          ncclSum,
          root_rank_,
          mgpu_ctx->nccl_comm(rank),
          conn.stream()));
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupEnd());
    }
    mgpu_ctx->sync();
  }
}

template <>
void ReduceOp<GPUBatchArray1D<float>>::_backward(TxnId txn) {
  size_t num_writes = 0;

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());

  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
    auto dev_ctx = mgpu_ctx->device_context(rank);
    if (xs_.at(rank).grad().accumulate(txn, id(), dev_ctx.get(), [batch_size, dev_ctx](auto grad) {
      auto conn = dev_ctx->conn();
      grad->set_batch_size(batch_size);
      grad->as_view().set_constant(0.0f, conn);
    })) {
      ++num_writes;
    }
  }

  assert(num_writes == 0 || num_writes == mgpu_ctx->num_ranks());
  if (num_writes > 0) {
    {
      auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);
      auto conn = root_dev_ctx->conn();
      xs_.at(root_rank_).grad().get_mut(txn, id()).as_view()
        .add(y_.grad().get(txn, id()).as_view(), conn);
    }

    mgpu_ctx->sync();
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupStart());
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
      assert(batch_size == y_.grad().get(txn, id()).batch_size());
      assert(batch_size == xs_.at(rank).grad().get_mut(txn, id()).batch_size());
      auto x_dim = xs_.at(rank).val().get(txn, id()).dim();
      auto x_stride = xs_.at(rank).val().get(txn, id()).stride();
      assert(x_dim.is_packed(x_stride) && "ReduceOp only supports packed inputs");
      assert(x_dim.shape == y_.grad().get(txn, id()).dim().shape);
      assert(x_dim.shape == xs_.at(rank).grad().get_mut(txn, id()).dim().shape);
      auto dev_ctx = mgpu_ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      // FIXME: broadcast to a temporary before adding to dx.
      NCCL_CHECK(ncclBcast(
          xs_.at(rank).grad().get_mut(txn, id()).as_view().dptr(),
          xs_.at(rank).grad().get_mut(txn, id()).as_view().dim().flat_size(),
          ncclFloat,
          ncclSum,
          mgpu_ctx->nccl_comm(rank),
          conn.stream()));
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupEnd());
    }
    mgpu_ctx->sync();
  }
}

template <>
void ReduceOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
  auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);

  if (y_.val().overwrite(txn, id(), root_dev_ctx.get())) {
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    auto y_stride = y_.val().get_excl(txn, id()).stride();
    assert(y_dim.is_packed(y_stride) && "ReduceOp only supports packed outputs");
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
      auto x_dim = xs_.at(rank).val().get(txn, id()).dim();
      auto x_stride = xs_.at(rank).val().get(txn, id()).stride();
      assert(x_dim.is_packed(x_stride) && "ReduceOp only supports packed inputs");
      if (0 == rank) {
        y_.val().get_excl(txn, id()).set_batch_size(batch_size);
      } else {
        assert(batch_size == y_.val().get_excl(txn, id()).batch_size());
      }
      assert(x_dim.shape == y_dim.shape);
    }

    mgpu_ctx->sync();
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupStart());
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      NCCL_CHECK(ncclReduce(
          xs_.at(rank).val().get(txn, id()).as_view().dptr(),
          y_.val().get_excl(txn, id()).as_view().dptr(),
          y_.val().get_excl(txn, id()).as_view().dim().flat_size(),
          ncclFloat,
          ncclSum,
          root_rank_,
          mgpu_ctx->nccl_comm(rank),
          conn.stream()));
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupEnd());
    }
    mgpu_ctx->sync();
  }
}

template <>
void ReduceOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t num_writes = 0;

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());

  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
    auto dev_ctx = mgpu_ctx->device_context(rank);
    if (xs_.at(rank).grad().accumulate(txn, id(), dev_ctx.get(), [batch_size, dev_ctx](auto grad) {
      auto conn = dev_ctx->conn();
      grad->set_batch_size(batch_size);
      grad->as_view().set_constant(0.0f, conn);
    })) {
      ++num_writes;
    }
  }

  assert(num_writes == 0 || num_writes == mgpu_ctx->num_ranks());
  if (num_writes > 0) {
    {
      auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);
      auto conn = root_dev_ctx->conn();
      xs_.at(root_rank_).grad().get_mut(txn, id()).as_view()
        .add(y_.grad().get(txn, id()).as_view(), conn);
    }

    mgpu_ctx->sync();
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupStart());
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
      assert(batch_size == y_.grad().get(txn, id()).batch_size());
      assert(batch_size == xs_.at(rank).grad().get_mut(txn, id()).batch_size());
      auto x_dim = xs_.at(rank).val().get(txn, id()).dim();
      auto x_stride = xs_.at(rank).val().get(txn, id()).stride();
      assert(x_dim.is_packed(x_stride) && "ReduceOp only supports packed inputs");
      assert(x_dim.shape == y_.grad().get(txn, id()).dim().shape);
      assert(x_dim.shape == xs_.at(rank).grad().get_mut(txn, id()).dim().shape);
      auto dev_ctx = mgpu_ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      // FIXME: broadcast to a temporary before adding to dx.
      NCCL_CHECK(ncclBcast(
          xs_.at(rank).grad().get_mut(txn, id()).as_view().dptr(),
          xs_.at(rank).grad().get_mut(txn, id()).as_view().dim().flat_size(),
          ncclFloat,
          ncclSum,
          mgpu_ctx->nccl_comm(rank),
          conn.stream()));
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupEnd());
    }
    mgpu_ctx->sync();
  }
}

template <>
void SelectOp<GPUBatchArray1D<float>>::_forward(TxnId txn) {
  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
  auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);

  size_t batch_size = xs_.at(root_rank_).val().get(txn, id()).batch_size();
  auto x_dim = xs_.at(root_rank_).val().get(txn, id()).dim();
  auto x_stride = xs_.at(root_rank_).val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "SelectOp only supports packed inputs");

  if (y_.val().overwrite(txn, id(), root_dev_ctx.get())) {
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    auto y_stride = y_.val().get_excl(txn, id()).stride();
    assert(y_dim.is_packed(y_stride) && "SelectOp only supports packed outputs");
    assert(x_dim.shape == y_dim.shape);

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);

    auto conn = root_dev_ctx->conn();
    y_.val().get_excl(txn, id()).as_view()
      .copy(xs_.at(root_rank_).val().get(txn, id()).as_view(), conn);
  }
}

template <>
void SelectOp<GPUBatchArray1D<float>>::_backward(TxnId txn) {
  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
  auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);

  size_t batch_size = xs_.at(root_rank_).val().get(txn, id()).batch_size();
  auto x_dim = xs_.at(root_rank_).val().get(txn, id()).dim();
  auto x_stride = xs_.at(root_rank_).val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "SelectOp only supports packed inputs");

  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    if (xs_.at(rank).grad().accumulate(txn, id(), root_dev_ctx.get(), [batch_size, root_dev_ctx](auto grad) {
      auto conn = root_dev_ctx->conn();
      grad->set_batch_size(batch_size);
      grad->as_view().set_constant(0.0f, conn);
    })) {
      if (rank == root_rank_) {
        auto y_dim = y_.val().get(txn, id()).dim();
        auto y_stride = y_.val().get(txn, id()).stride();
        assert(y_dim.is_packed(y_stride) && "SelectOp only supports packed outputs");
        assert(x_dim.shape == y_dim.shape);

        auto conn = root_dev_ctx->conn();
        xs_.at(root_rank_).grad().get_mut(txn, id()).as_view()
          .add(y_.grad().get(txn, id()).as_view(), conn);
      }
    }
  }
}

template <>
void SelectOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
  auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);

  size_t batch_size = xs_.at(root_rank_).val().get(txn, id()).batch_size();
  auto x_dim = xs_.at(root_rank_).val().get(txn, id()).dim();
  auto x_stride = xs_.at(root_rank_).val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "SelectOp only supports packed inputs");

  if (y_.val().overwrite(txn, id(), root_dev_ctx.get())) {
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    auto y_stride = y_.val().get_excl(txn, id()).stride();
    assert(y_dim.is_packed(y_stride) && "SelectOp only supports packed outputs");
    assert(x_dim.shape == y_dim.shape);

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);

    auto conn = root_dev_ctx->conn();
    y_.val().get_excl(txn, id()).as_view()
      .copy(xs_.at(root_rank_).val().get(txn, id()).as_view(), conn);
  }
}

template <>
void SelectOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
  auto root_dev_ctx = mgpu_ctx->device_context(root_rank_);

  size_t batch_size = xs_.at(root_rank_).val().get(txn, id()).batch_size();
  auto x_dim = xs_.at(root_rank_).val().get(txn, id()).dim();
  auto x_stride = xs_.at(root_rank_).val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "SelectOp only supports packed inputs");

  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    if (xs_.at(rank).grad().accumulate(txn, id(), root_dev_ctx.get(), [batch_size, root_dev_ctx](auto grad) {
      auto conn = root_dev_ctx->conn();
      grad->set_batch_size(batch_size);
      grad->as_view().set_constant(0.0f, conn);
    })) {
      if (rank == root_rank_) {
        auto y_dim = y_.val().get(txn, id()).dim();
        auto y_stride = y_.val().get(txn, id()).stride();
        assert(y_dim.is_packed(y_stride) && "SelectOp only supports packed outputs");
        assert(x_dim.shape == y_dim.shape);

        auto conn = root_dev_ctx->conn();
        xs_.at(root_rank_).grad().get_mut(txn, id()).as_view()
          .add(y_.grad().get(txn, id()).as_view(), conn);
      }
    }
  }
}

template <>
void AllreduceOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t num_writes = 0;

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());

  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
    auto x_dim = xs_.at(rank).val().get(txn, id()).dim();
    auto x_stride = xs_.at(rank).val().get(txn, id()).stride();
    assert(x_dim.is_packed(x_stride) && "AllreduceOp only supports packed inputs");
    auto dev_ctx = mgpu_ctx->device_context(rank);
    if (ys_.at(rank).val().overwrite(txn, id(), dev_ctx.get())) {
      ys_.at(rank).val().get_excl(txn, id()).set_batch_size(batch_size);
      auto y_dim = ys_.at(rank).val().get_excl(txn, id()).dim();
      auto y_stride = ys_.at(rank).val().get_excl(txn, id()).stride();
      assert(x_dim.shape == y_dim.shape);
      assert(y_dim.is_packed(y_stride) && "AllreduceOp only supports packed outputs");
      ++num_writes;
    }
  }

  assert(num_writes == 0 || num_writes == mgpu_ctx->num_ranks());
  if (num_writes > 0) {
    mgpu_ctx->sync();
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupStart());
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      NCCL_CHECK(ncclAllReduce(
          xs_.at(rank).val().get(txn, id()).as_view().dptr(),
          ys_.at(rank).val().get_excl(txn, id()).as_view().dptr(),
          ys_.at(rank).val().get_excl(txn, id()).as_view().dim().flat_size(),
          ncclFloat,
          ncclSum,
          mgpu_ctx->nccl_comm(rank),
          conn.stream()));
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupEnd());
    }
    mgpu_ctx->sync();
  }
}

template <>
void AllreduceOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t num_writes = 0;

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());

  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
    auto dev_ctx = mgpu_ctx->device_context(rank);
    if (xs_.at(rank).grad().accumulate(txn, id(), dev_ctx.get(), [batch_size, dev_ctx](auto grad) {
      auto conn = dev_ctx->conn();
      grad->set_batch_size(batch_size);
      grad->as_view().set_constant(0.0f, conn);
    })) {
      ++num_writes;
    }
  }

  assert(num_writes == 0 || num_writes == mgpu_ctx->num_ranks());
  if (num_writes > 0) {
    mgpu_ctx->sync();
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupStart());
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
      assert(batch_size == ys_.at(rank).grad().get(txn, id()).batch_size());
      assert(batch_size == xs_.at(rank).grad().get_mut(txn, id()).batch_size());
      auto x_dim = xs_.at(rank).val().get(txn, id()).dim();
      auto x_stride = xs_.at(rank).val().get(txn, id()).stride();
      assert(x_dim.is_packed(x_stride) && "AllreduceOp only supports packed inputs");
      assert(x_dim.shape == ys_.at(rank).grad().get(txn, id()).dim().shape);
      assert(x_dim.shape == xs_.at(rank).grad().get_mut(txn, id()).dim().shape);
      auto dev_ctx = mgpu_ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      // FIXME: all-reduce to a temporary before adding to dx.
      NCCL_CHECK(ncclAllReduce(
          ys_.at(rank).grad().get(txn, id()).as_view().dptr(),
          xs_.at(rank).grad().get_mut(txn, id()).as_view().dptr(),
          xs_.at(rank).grad().get_mut(txn, id()).as_view().dim().flat_size(),
          ncclFloat,
          ncclSum,
          mgpu_ctx->nccl_comm(rank),
          conn.stream()));
    }
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      NCCL_CHECK(ncclGroupEnd());
    }
    mgpu_ctx->sync();
  }
}

template <>
Shape3D SpatialScatter2DOp<GPUBatchArray3D<float>>::_get_output_shape(Shape3D in_shape) {
  assert((0 == in_shape.shape.at(1) % num_ranks_) &&
      "Currently, SpatialScatter2DOp requires the input height to be a multiple "
      "of the number of ranks.");
  return Shape3D(
      in_shape.shape.at(0),
      in_shape.shape.at(1) / num_ranks_,
      //in_shape.shape.at(1),
      in_shape.shape.at(2));
}

template <>
void SpatialScatter2DOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  auto x_stride = x_.val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "SpatialScatter2DOp only supports packed inputs");

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.val().accumulate(txn, id(), ctx, [](auto val) { (void)val; })) {
    auto conn = ctx->conn();
    y_.val().get_mut(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_mut(txn, id()).dim();
    auto y_stride = y_.val().get_mut(txn, id()).stride();
    assert(y_dim.is_packed(y_stride));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));
    //assert(x_dim.shape == y_dim.shape);
    size_t rank_w = 0;
    size_t rank_h = rank_;
    //size_t src_off_w = rank_w * y_dim.shape.at(0);
    //size_t src_off_h = rank_h * y_dim.shape.at(1);
    size_t src_off_w = 0;
    size_t src_off_h = rank_h * x_dim.shape.at(1) / num_ranks_;
    slice2d_copy_hw_nchw_packed(
        //y_dim.shape.at(0), y_dim.shape.at(1),
        x_dim.shape.at(0), x_dim.shape.at(1) / num_ranks_,
        x_dim.shape.at(0), x_dim.shape.at(1),
        y_dim.shape.at(0), y_dim.shape.at(1),
        y_dim.shape.at(2), batch_size,
        x_.val().get(txn, id()).as_view().dptr(),
        src_off_w, src_off_h,
        y_.val().get_mut(txn, id()).as_view().dptr(),
        0, 0,
        conn.stream());
  }
}

template <>
void SpatialScatter2DOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  auto x_stride = x_.val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "SpatialScatter2DOp only supports packed inputs");

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    auto conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    auto conn = ctx->conn();
    auto y_dim = y_.val().get(txn, id()).dim();
    auto y_stride = y_.val().get(txn, id()).stride();
    assert(y_dim.is_packed(y_stride));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));
    //assert(x_dim.shape == y_dim.shape);
    size_t rank_w = 0;
    size_t rank_h = rank_;
    //size_t dst_off_w = rank_w * y_dim.shape.at(0);
    //size_t dst_off_h = rank_h * y_dim.shape.at(1);
    size_t dst_off_w = 0;
    size_t dst_off_h = rank_h * x_dim.shape.at(1) / num_ranks_;
    slice2d_add_hw_nchw_packed(
        //y_dim.shape.at(0), y_dim.shape.at(1),
        x_dim.shape.at(0), x_dim.shape.at(1) / num_ranks_,
        y_dim.shape.at(0), y_dim.shape.at(1),
        x_dim.shape.at(0), x_dim.shape.at(1),
        x_dim.shape.at(2), batch_size,
        y_.grad().get(txn, id()).as_view().dptr(),
        0, 0,
        x_.grad().get_mut(txn, id()).as_view().dptr(),
        dst_off_w, dst_off_h,
        conn.stream());
  }
}

template <>
Shape3D SpatialGather2DOp<GPUBatchArray3D<float>>::_get_output_shape(Shape3D in_shape) {
  return Shape3D(
      in_shape.shape.at(0),
      in_shape.shape.at(1) * num_ranks_,
      //in_shape.shape.at(1),
      in_shape.shape.at(2));
}

template <>
void SpatialGather2DOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  auto x_stride = x_.val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "SpatialGather2DOp only supports packed inputs");

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (y_.val().accumulate(txn, id(), ctx, [](auto val) { (void)val; })) {
    auto conn = ctx->conn();
    y_.val().get_mut(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_mut(txn, id()).dim();
    auto y_stride = y_.val().get_mut(txn, id()).stride();
    assert(y_dim.is_packed(y_stride));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));
    //assert(x_dim.shape == y_dim.shape);
    size_t rank_w = 0;
    size_t rank_h = rank_;
    //size_t dst_off_w = rank_w * y_dim.shape.at(0);
    //size_t dst_off_h = rank_h * y_dim.shape.at(1);
    size_t dst_off_w = 0;
    size_t dst_off_h = rank_h * y_dim.shape.at(1) / num_ranks_;
    y_.val().get_mut(txn, id()).as_view().set_constant(0.0f, conn);
    slice2d_copy_hw_nchw_packed(
        //x_dim.shape.at(0), x_dim.shape.at(1),
        y_dim.shape.at(0), y_dim.shape.at(1) / num_ranks_,
        x_dim.shape.at(0), x_dim.shape.at(1),
        y_dim.shape.at(0), y_dim.shape.at(1),
        y_dim.shape.at(2), batch_size,
        x_.val().get(txn, id()).as_view().dptr(),
        0, 0,
        y_.val().get_mut(txn, id()).as_view().dptr(),
        dst_off_w, dst_off_h,
        conn.stream());
  }
}

template <>
void SpatialGather2DOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();
  auto x_stride = x_.val().get(txn, id()).stride();
  assert(x_dim.is_packed(x_stride) && "SpatialGather2DOp only supports packed inputs");

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());
  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    auto conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    auto conn = ctx->conn();
    auto y_dim = y_.val().get(txn, id()).dim();
    auto y_stride = y_.val().get(txn, id()).stride();
    assert(y_dim.is_packed(y_stride));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));
    //assert(x_dim.shape == y_dim.shape);
    size_t rank_w = 0;
    size_t rank_h = rank_;
    //size_t src_off_w = rank_w * y_dim.shape.at(0);
    //size_t src_off_h = rank_h * y_dim.shape.at(1);
    size_t src_off_w = 0;
    size_t src_off_h = rank_h * y_dim.shape.at(1) / num_ranks_;
    slice2d_add_hw_nchw_packed(
        //x_dim.shape.at(0), x_dim.shape.at(1),
        y_dim.shape.at(0), y_dim.shape.at(1) / num_ranks_,
        y_dim.shape.at(0), y_dim.shape.at(1),
        x_dim.shape.at(0), x_dim.shape.at(1),
        x_dim.shape.at(2), batch_size,
        y_.grad().get(txn, id()).as_view().dptr(),
        src_off_w, src_off_h,
        x_.grad().get_mut(txn, id()).as_view().dptr(),
        0, 0,
        conn.stream());
  }
}

template <>
Shape3D HaloPad2DOp<GPUBatchArray3D<float>>::_get_output_offset(Shape3D in_shape) {
  return Shape3D(halo_pad_, halo_pad_, 0);
}

template <>
Shape3D HaloPad2DOp<GPUBatchArray3D<float>>::_get_output_stride(Shape3D in_shape) {
  /*size_t y_stride_w = 1;
  size_t y_stride_h = (in_shape.shape.at(0) + halo_pad_ * 2) * y_stride_w;
  size_t y_stride_c = (in_shape.shape.at(1) + halo_pad_ * 2) * y_stride_h;
  return Shape3D(y_stride_w, y_stride_h, y_stride_c);*/
  Shape3D y_padded_dim(
      in_shape.shape.at(0) + halo_pad_ * 2,
      in_shape.shape.at(1) + halo_pad_ * 2,
      in_shape.shape.at(2));
  return y_padded_dim.packed_stride();
}

template <>
void HaloPad2DOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().accumulate(txn, id(), ctx, [batch_size, ctx](auto val) {
    auto conn = ctx->conn();
    val->set_batch_size(batch_size);
    val->base_as_view().set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();

    // Check correct dim.
    auto y_dim = y_.val().get_mut(txn, id()).dim();
    assert(x_dim.shape.at(0) == y_dim.shape.at(0));
    assert(x_dim.shape.at(1) == y_dim.shape.at(1));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));
    assert(batch_size == y_.val().get_mut(txn, id()).batch_size());

    // Check correct offset.
    auto y_offset = y_.val().get_mut(txn, id()).offset();
    assert(halo_pad_ == y_offset.shape.at(0));
    assert(halo_pad_ == y_offset.shape.at(1));
    assert(0 == y_offset.shape.at(2));

    // Check correct stride.
    auto y_stride = y_.val().get_mut(txn, id()).stride();
    int y_stride_w = 1;
    int y_stride_h = (x_dim.shape.at(0) + halo_pad_ * 2) * y_stride_w;
    int y_stride_c = (x_dim.shape.at(1) + halo_pad_ * 2) * y_stride_h;
    int y_stride_n = x_dim.shape.at(2) * y_stride_c;
    assert(y_stride_w == y_stride.shape.at(0));
    assert(y_stride_h == y_stride.shape.at(1));
    assert(y_stride_c == y_stride.shape.at(2));
    assert(y_stride_n == y_stride.minor() * y_dim.minor());

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        x_dim.shape.at(2),
        x_dim.shape.at(1),
        x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        y_desc_,
        CUDNN_DATA_FLOAT,
        batch_size,
        y_dim.shape.at(2),
        y_dim.shape.at(1),
        y_dim.shape.at(0),
        y_stride_n,
        y_stride_c,
        y_stride_h,
        y_stride_w));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnAddTensor(
        conn.cudnn(),
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_mut(txn, id()).as_view().dptr()));
  }
}

template <>
void HaloPad2DOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();

    // Check correct dim.
    auto y_dim = y_.val().get(txn, id()).dim();
    assert(x_dim.shape.at(0) == y_dim.shape.at(0));
    assert(x_dim.shape.at(1) == y_dim.shape.at(1));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));
    assert(batch_size == y_.val().get_mut(txn, id()).batch_size());
    assert(batch_size == y_.grad().get(txn, id()).batch_size());

    // Check correct offset.
    auto y_offset = y_.val().get(txn, id()).offset();
    assert(halo_pad_ == y_offset.shape.at(0));
    assert(halo_pad_ == y_offset.shape.at(1));
    assert(0 == y_offset.shape.at(2));

    // Check correct stride.
    auto y_stride = y_.val().get(txn, id()).stride();
    int y_stride_w = 1;
    int y_stride_h = (x_dim.shape.at(0) + halo_pad_ * 2) * y_stride_w;
    int y_stride_c = (x_dim.shape.at(1) + halo_pad_ * 2) * y_stride_h;
    int y_stride_n = x_dim.shape.at(2) * y_stride_c;
    assert(y_stride_w == y_stride.shape.at(0));
    assert(y_stride_h == y_stride.shape.at(1));
    assert(y_stride_c == y_stride.shape.at(2));
    assert(y_stride_n == y_stride.minor() * y_dim.minor());

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        x_dim.shape.at(2),
        x_dim.shape.at(1),
        x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        y_desc_,
        CUDNN_DATA_FLOAT,
        batch_size,
        y_dim.shape.at(2),
        y_dim.shape.at(1),
        y_dim.shape.at(0),
        y_stride_n,
        y_stride_c,
        y_stride_h,
        y_stride_w));

    const float alpha = 1.0f;
    const float beta = 1.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnAddTensor(
        conn.cudnn(),
        &alpha,
        y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
        &beta,
        x_desc_, x_.grad().get_mut(txn, id()).as_view().dptr()));
  }
}

template <>
void HaloUnpad2DOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().accumulate(txn, id(), ctx, [batch_size, ctx](auto val) {
    auto conn = ctx->conn();
    val->set_batch_size(batch_size);
    // NOTE: No need to zero the halo here.
    /*val->as_view().set_constant(0.0f, conn);*/
  })) {
    GPUConn conn = ctx->conn();

    // Check correct dim.
    auto y_dim = y_.val().get_mut(txn, id()).dim();
    assert(x_dim.shape.at(0) == y_dim.shape.at(0));
    assert(x_dim.shape.at(1) == y_dim.shape.at(1));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));
    assert(batch_size == y_.val().get_mut(txn, id()).batch_size());

    // Check correct offset.
    auto x_offset = x_.val().get(txn, id()).offset();
    if (halo_pad_ != x_offset.shape.at(0)) {
      std::clog << "WARNING: HaloUnpad2DOp: offset mismatch (0): "
          << halo_pad_ << " " << x_offset.shape.at(0) << std::endl;
    }
    if (halo_pad_ != x_offset.shape.at(1)) {
      std::clog << "WARNING: HaloUnpad2DOp: offset mismatch (1): "
          << halo_pad_ << " " << x_offset.shape.at(1) << std::endl;
    }
    assert(halo_pad_ == x_offset.shape.at(0));
    assert(halo_pad_ == x_offset.shape.at(1));
    assert(0 == x_offset.shape.at(2));
    auto y_offset = y_.val().get_mut(txn, id()).offset();
    assert(y_offset.is_zero());

    // Check correct stride.
    auto x_stride = x_.val().get(txn, id()).stride();
    int x_stride_w = 1;
    int x_stride_h = (x_dim.shape.at(0) + halo_pad_ * 2) * x_stride_w;
    int x_stride_c = (x_dim.shape.at(1) + halo_pad_ * 2) * x_stride_h;
    int x_stride_n = x_dim.shape.at(2) * x_stride_c;
    assert(x_stride_w == x_stride.shape.at(0));
    assert(x_stride_h == x_stride.shape.at(1));
    assert(x_stride_c == x_stride.shape.at(2));
    assert(x_stride_n == x_stride.minor() * x_dim.minor());
    auto y_stride = y_.val().get_mut(txn, id()).stride();
    assert(y_dim.is_packed(y_stride));

    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        x_desc_,
        CUDNN_DATA_FLOAT,
        batch_size,
        x_dim.shape.at(2),
        x_dim.shape.at(1),
        x_dim.shape.at(0),
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        y_dim.shape.at(2),
        y_dim.shape.at(1),
        y_dim.shape.at(0)));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnAddTensor(
        conn.cudnn(),
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_mut(txn, id()).as_view().dptr()));
  }
}

template <>
void HaloUnpad2DOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();

    // Check correct dim.
    auto y_dim = y_.val().get_mut(txn, id()).dim();
    assert(x_dim.shape.at(0) == y_dim.shape.at(0));
    assert(x_dim.shape.at(1) == y_dim.shape.at(1));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));
    assert(batch_size == y_.val().get_mut(txn, id()).batch_size());
    assert(batch_size == y_.grad().get(txn, id()).batch_size());

    // Check correct offset.
    auto x_offset = x_.val().get(txn, id()).offset();
    assert(halo_pad_ == x_offset.shape.at(0));
    assert(halo_pad_ == x_offset.shape.at(1));
    assert(0 == x_offset.shape.at(2));
    auto y_offset = y_.val().get_mut(txn, id()).offset();
    assert(y_offset.is_zero());

    // Check correct stride.
    auto x_stride = x_.val().get(txn, id()).stride();
    int x_stride_w = 1;
    int x_stride_h = (x_dim.shape.at(0) + halo_pad_ * 2) * x_stride_w;
    int x_stride_c = (x_dim.shape.at(1) + halo_pad_ * 2) * x_stride_h;
    int x_stride_n = x_dim.shape.at(2) * x_stride_c;
    assert(x_stride_w == x_stride.shape.at(0));
    assert(x_stride_h == x_stride.shape.at(1));
    assert(x_stride_c == x_stride.shape.at(2));
    assert(x_stride_n == x_stride.minor() * x_dim.minor());
    auto y_stride = y_.val().get_mut(txn, id()).stride();
    assert(y_dim.is_packed(y_stride));

    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        x_desc_,
        CUDNN_DATA_FLOAT,
        batch_size,
        x_dim.shape.at(2),
        x_dim.shape.at(1),
        x_dim.shape.at(0),
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        y_dim.shape.at(2),
        y_dim.shape.at(1),
        y_dim.shape.at(0)));

    const float alpha = 1.0f;
    const float beta = 1.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnAddTensor(
        conn.cudnn(),
        &alpha,
        y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
        &beta,
        x_desc_, x_.grad().get_mut(txn, id()).as_view().dptr()));
  }
}

template <>
void HaloRepad2DOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().accumulate(txn, id(), ctx, [batch_size, ctx](auto val) {
    auto conn = ctx->conn();
    val->set_batch_size(batch_size);
    val->base_as_view().set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();

    // Check correct dim.
    auto y_dim = y_.val().get_mut(txn, id()).dim();
    assert(x_dim.shape.at(0) == y_dim.shape.at(0));
    assert(x_dim.shape.at(1) == y_dim.shape.at(1));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));
    assert(batch_size == y_.val().get_mut(txn, id()).batch_size());

    // Check correct input offset.
    auto x_offset = x_.val().get(txn, id()).offset();
    assert(src_halo_pad_ == x_offset.shape.at(0));
    assert(src_halo_pad_ == x_offset.shape.at(1));
    assert(0 == x_offset.shape.at(2));

    // Check correct output offset.
    auto y_offset = y_.val().get_mut(txn, id()).offset();
    assert(dst_halo_pad_ == y_offset.shape.at(0));
    assert(dst_halo_pad_ == y_offset.shape.at(1));
    assert(0 == y_offset.shape.at(2));

    // Check correct input stride.
    auto x_stride = x_.val().get(txn, id()).stride();
    int x_stride_w = 1;
    int x_stride_h = (x_dim.shape.at(0) + src_halo_pad_ * 2) * x_stride_w;
    int x_stride_c = (x_dim.shape.at(1) + src_halo_pad_ * 2) * x_stride_h;
    int x_stride_n = x_dim.shape.at(2) * x_stride_c;
    assert(x_stride_w == x_stride.shape.at(0));
    assert(x_stride_h == x_stride.shape.at(1));
    assert(x_stride_c == x_stride.shape.at(2));
    assert(x_stride_n == x_stride.minor() * x_dim.minor());

    // Check correct output stride.
    auto y_stride = y_.val().get_mut(txn, id()).stride();
    int y_stride_w = 1;
    int y_stride_h = (x_dim.shape.at(0) + dst_halo_pad_ * 2) * y_stride_w;
    int y_stride_c = (x_dim.shape.at(1) + dst_halo_pad_ * 2) * y_stride_h;
    int y_stride_n = x_dim.shape.at(2) * y_stride_c;
    assert(y_stride_w == y_stride.shape.at(0));
    assert(y_stride_h == y_stride.shape.at(1));
    assert(y_stride_c == y_stride.shape.at(2));
    assert(y_stride_n == y_stride.minor() * y_dim.minor());

    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        x_desc_,
        CUDNN_DATA_FLOAT,
        batch_size,
        x_dim.shape.at(2),
        x_dim.shape.at(1),
        x_dim.shape.at(0),
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
        y_desc_,
        CUDNN_DATA_FLOAT,
        batch_size,
        y_dim.shape.at(2),
        y_dim.shape.at(1),
        y_dim.shape.at(0),
        y_stride_n,
        y_stride_c,
        y_stride_h,
        y_stride_w));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnAddTensor(
        conn.cudnn(),
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_mut(txn, id()).as_view().dptr()));
  }
}

template <>
void HaloExch2DOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  MultiGPUContext* ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());

  size_t num_writes = 0;
  for (size_t rank = 0; rank < ctx->num_ranks(); ++rank) {
    auto dev_ctx = ctx->device_context(rank);
    if (xs_.at(rank).val().accumulate(txn, id(), dev_ctx.get(), [](auto val) {})) {
      num_writes += 1;
    }
  }
  assert(0 == num_writes || ctx->num_ranks() == num_writes);

  if (num_writes > 0) {
    ctx->sync();
    for (size_t rank = 0; rank < ctx->num_ranks(); ++rank) {
      size_t batch_size = xs_.at(rank).val().get_mut(txn, id()).batch_size();
      auto x_dim = xs_.at(rank).val().get_mut(txn, id()).dim();
      auto x_offset = xs_.at(rank).val().get_mut(txn, id()).offset();
      assert(x_offset.shape.at(0) == halo_pad_);
      assert(x_offset.shape.at(1) == halo_pad_);
      auto dev_ctx = ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      ctx->spatial_comm(rank)->group_start();
      ctx->spatial_comm(rank)->halo_exchange(
          xs_.at(rank).val().get_mut(txn, id()).base_as_view().dptr(),
          halo_pad_,
          x_dim.shape.at(0),
          x_dim.shape.at(1),
          x_dim.shape.at(2),
          batch_size,
          SPATIAL2D_TENSOR_NCHW,
          SPATIAL2D_DATA_FLOAT);
    }
    for (size_t rank = 0; rank < ctx->num_ranks(); ++rank) {
      auto dev_ctx = ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      ctx->spatial_comm(rank)->group_end(conn);
    }
    ctx->sync();
  }
}

template <>
void HaloExch2DOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  MultiGPUContext* ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());

  size_t num_writes = 0;
  for (size_t rank = 0; rank < ctx->num_ranks(); ++rank) {
    auto dev_ctx = ctx->device_context(rank);
    size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
    if (xs_.at(rank).grad().accumulate(txn, id(), dev_ctx.get(), [batch_size, dev_ctx](auto grad) {
      auto conn = dev_ctx->conn();
      grad->set_batch_size(batch_size);
      grad->base_as_view().set_constant(0.0f, conn);
    })) {
      num_writes += 1;
    }
  }
  assert(0 == num_writes || ctx->num_ranks() == num_writes);

  if (num_writes > 0) {
    ctx->sync();
    for (size_t rank = 0; rank < ctx->num_ranks(); ++rank) {
      size_t batch_size = xs_.at(rank).val().get(txn, id()).batch_size();
      assert(batch_size == xs_.at(rank).grad().get_mut(txn, id()).batch_size());
      auto x_dim = xs_.at(rank).val().get(txn, id()).dim();
      auto x_offset = xs_.at(rank).val().get(txn, id()).offset();
      assert(x_offset.shape.at(0) == halo_pad_);
      assert(x_offset.shape.at(1) == halo_pad_);
      auto dev_ctx = ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      ctx->spatial_comm(rank)->group_start();
      ctx->spatial_comm(rank)->halo_reduce(
          xs_.at(rank).grad().get_mut(txn, id()).base_as_view().dptr(),
          halo_pad_,
          x_dim.shape.at(0),
          x_dim.shape.at(1),
          x_dim.shape.at(2),
          batch_size,
          SPATIAL2D_TENSOR_NCHW,
          SPATIAL2D_DATA_FLOAT);
    }
    for (size_t rank = 0; rank < ctx->num_ranks(); ++rank) {
      auto dev_ctx = ctx->device_context(rank);
      auto conn = dev_ctx->conn();
      ctx->spatial_comm(rank)->group_end(conn);
    }
    ctx->sync();
  }
}

template <>
void CudnnActivationOp<GPUBatchArray1D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    assert(x_dim.shape == y_dim.shape);

    cudnnActivationMode_t act_mode;
    if (activation_ == kReluAct) {
      act_mode = CUDNN_ACTIVATION_RELU;
    } else {
      assert(0 && "unreachable");
    }

    CUDNN_CHECK(cudnnSetActivationDescriptor(
        act_desc_,
        act_mode,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0));

    /*x_.val().get(txn, id()).as_view()
      .set_cudnn_tensor_desc(x_desc_, CUDNN_TENSOR_NCHW);
    y_.val().get_excl(txn, id()).as_view()
      .set_cudnn_tensor_desc(y_desc_, CUDNN_TENSOR_NCHW);*/

    set_cudnn_tensor_desc_batch1d_ex(
        x_desc_,
        x_.val().get(txn, id()).dim(),
        x_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);
    set_cudnn_tensor_desc_batch1d_ex(
        y_desc_,
        y_.val().get_excl(txn, id()).dim(),
        y_.val().get_excl(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnActivationForward(
        conn.cudnn(),
        act_desc_,
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
  }
}

template <>
void CudnnActivationOp<GPUBatchArray1D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  bool in_place = this->in_place_;
  if (x_.grad().accumulate(txn, id(), ctx, [in_place, batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    if (in_place) {
      assert(batch_size == grad->batch_size());
    } else {
      grad->set_batch_size(batch_size);
      grad->as_view().set_constant(0.0f, conn);
    }
  })) {
    GPUConn conn = ctx->conn();

    /*CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape, 1, 1));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        dx_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape, 1, 1));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape, 1, 1));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        dy_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape, 1, 1));*/

    set_cudnn_tensor_desc_batch1d_ex(
        x_desc_,
        x_.val().get(txn, id()).dim(),
        x_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);
    set_cudnn_tensor_desc_batch1d_ex(
        y_desc_,
        y_.val().get(txn, id()).dim(),
        y_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);

    const float alpha = 1.0f;
    float beta = 0.0f;
    if (in_place_) {
      beta = 0.0f;
    } else {
      beta = 1.0f;
    }
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnActivationBackward(
        conn.cudnn(),
        act_desc_,
        &alpha,
        y_desc_, y_.val().get(txn, id()).as_view().dptr(),
        y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        x_desc_, x_.grad().get_mut(txn, id()).as_view().dptr()));
  }
}

template <>
void CudnnActivationOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  //if (y_.val().accumulate(txn, id(), ctx, [](auto val) { (void)val; })) {
  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();
    assert(x_dim.shape.at(0) == y_dim.shape.at(0));
    assert(x_dim.shape.at(1) == y_dim.shape.at(1));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));

    cudnnActivationMode_t act_mode;
    if (activation_ == kReluAct) {
      act_mode = CUDNN_ACTIVATION_RELU;
    } else {
      assert(0 && "unreachable");
    }
    CUDNN_CHECK(cudnnSetActivationDescriptor(
        act_desc_,
        act_mode,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0));

    /*x_.val().get(txn, id()).as_view()
      .set_cudnn_tensor_desc(x_desc_, CUDNN_TENSOR_NCHW);
    y_.val().get_mut(txn, id()).as_view()
      .set_cudnn_tensor_desc(y_desc_, CUDNN_TENSOR_NCHW);*/

    set_cudnn_tensor_desc_batch3d_ex(
        x_desc_,
        x_.val().get(txn, id()).dim(),
        x_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);
    set_cudnn_tensor_desc_batch3d_ex(
        y_desc_,
        y_.val().get_excl(txn, id()).dim(),
        y_.val().get_excl(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnActivationForward(
        conn.cudnn(),
        act_desc_,
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
  }
}

template <>
void CudnnActivationOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  bool in_place = this->in_place_;
  if (x_.grad().accumulate(txn, id(), ctx, [in_place, batch_size, ctx](auto grad) {
    if (in_place) {
      assert(batch_size == grad->batch_size());
    } else {
      GPUConn conn = ctx->conn();
      grad->set_batch_size(batch_size);
      grad->as_view().set_constant(0.0f, conn);
    }
  })) {
    GPUConn conn = ctx->conn();

    /*CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        dx_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        dy_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));*/

    set_cudnn_tensor_desc_batch3d_ex(
        x_desc_,
        x_.val().get(txn, id()).dim(),
        x_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);
    set_cudnn_tensor_desc_batch3d_ex(
        y_desc_,
        y_.val().get(txn, id()).dim(),
        y_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);

    const float alpha = 1.0f;
    float beta = 0.0f;
    if (in_place_) {
      beta = 0.0f;
    } else {
      beta = 1.0f;
    }
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnActivationBackward(
        conn.cudnn(),
        act_desc_,
        &alpha,
        y_desc_, y_.val().get(txn, id()).as_view().dptr(),
        y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        x_desc_, x_.grad().get_mut(txn, id()).as_view().dptr()));
  }
}

template <>
shared_ptr<Var<GPUBatchArray1D<float>>> relu(shared_ptr<Var<GPUBatchArray1D<float>>> input, shared_ptr<Context> ctx) {
  return make_shared<CudnnActivationOp<GPUBatchArray1D<float>>>(kReluAct, false, input, ctx);
}

template <>
shared_ptr<Var<GPUBatchArray3D<float>>> relu(shared_ptr<Var<GPUBatchArray3D<float>>> input, shared_ptr<Context> ctx) {
  return make_shared<CudnnActivationOp<GPUBatchArray3D<float>>>(kReluAct, false, input, ctx);
}

template <>
shared_ptr<Var<GPUBatchArray1D<float>>> relu_in_place(shared_ptr<Var<GPUBatchArray1D<float>>> input, shared_ptr<Context> ctx) {
  return make_shared<CudnnActivationOp<GPUBatchArray1D<float>>>(kReluAct, true, input, ctx);
}

template <>
shared_ptr<Var<GPUBatchArray3D<float>>> relu_in_place(shared_ptr<Var<GPUBatchArray3D<float>>> input, shared_ptr<Context> ctx) {
  return make_shared<CudnnActivationOp<GPUBatchArray3D<float>>>(kReluAct, true, input, ctx);
}

template <>
Shape3D CudnnPool2DOp<GPUBatchArray3D<float>>::_get_output_shape(Shape3D in_shape) {
  cudnnPoolingMode_t pool_mode;
  if (shape_.pooling == kAveragePool) {
    pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else if (shape_.pooling == kMaxPool) {
    pool_mode = CUDNN_POOLING_MAX;
  } else {
    assert(0 && "unreachable");
  }
  std::clog << "DEBUG: Pool2D: get output shape: in shape:  " << in_shape.shape.at(0) << " " << in_shape.shape.at(1) << " " << in_shape.shape.at(2) << std::endl;
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(
      pool_desc_,
      pool_mode,
      CUDNN_NOT_PROPAGATE_NAN,
      shape_.window.at(1), shape_.window.at(0),
      shape_.pad.at(1), shape_.pad.at(0),
      shape_.stride.at(1), shape_.stride.at(0)));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      x_desc_,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      1,
      in_shape.shape.at(2),
      in_shape.shape.at(1),
      in_shape.shape.at(0)));
  int out_n = 0;
  int out_c = 0;
  int out_h = 0;
  int out_w = 0;
  CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(
      pool_desc_,
      x_desc_,
      &out_n,
      &out_c,
      &out_h,
      &out_w));
  std::clog << "DEBUG: Pool2D: get output shape: out shape: " << out_w << " " << out_h << " " << out_c << std::endl;
  return Shape3D(out_w, out_h, out_c);
}

/*template <>
void CudnnPool2DOp<GPUBatchArray3D<float>>::_set_grad_sink(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  //std::clog << "DEBUG: Pool2D: set_grad_sink()" << std::endl;
  if (y_.grad().overwrite(txn, id(), ctx)) {
    //std::clog << "DEBUG: Pool2D: set_grad_sink(): overwrite" << std::endl;
    GPUConn conn = ctx->conn();
    y_.grad().get_excl(txn, id()).as_view().set_constant(0.0f, conn);
  }
}*/

template <>
void CudnnPool2DOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();

    cudnnPoolingMode_t pool_mode;
    if (shape_.pooling == kAveragePool) {
      pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    } else if (shape_.pooling == kMaxPool) {
      pool_mode = CUDNN_POOLING_MAX;
    } else {
      assert(0 && "unreachable");
    }

    if (shape_.halo_padded_input) {
      auto x_offset = x_.val().get(txn, id()).offset();
      auto x_stride = x_.val().get(txn, id()).stride();
      assert(x_offset.shape.at(0) == shape_.pad.at(0));
      assert(x_offset.shape.at(1) == shape_.pad.at(1));
      assert(x_offset.shape.at(2) == 0);
      Shape3D x_padded_dim(
          x_dim.shape.at(0) + shape_.pad.at(0) * 2,
          x_dim.shape.at(1) + shape_.pad.at(1) * 2,
          x_dim.shape.at(2));
      assert(x_padded_dim.is_packed(x_stride));
      /*std::clog << "DEBUG: Pool2DOp: halo padded pooling (explicit padding)" << std::endl;
      std::clog << "DEBUG: Pool2DOp:   halo padded input shape: "
          << x_padded_dim.shape.at(0) << " " << x_padded_dim.shape.at(1) << " " << x_padded_dim.shape.at(2) << std::endl;
      std::clog << "DEBUG: Pool2DOp:   ordinary output shape: "
          << y_dim.shape.at(0) << " " << y_dim.shape.at(1) << " " << y_dim.shape.at(2) << std::endl;*/
      CUDNN_CHECK(cudnnSetPooling2dDescriptor(
          pool_desc_,
          pool_mode,
          CUDNN_NOT_PROPAGATE_NAN,
          shape_.window.at(1), shape_.window.at(0),
          0, 0,
          shape_.stride.at(1), shape_.stride.at(0)));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          x_desc_,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          batch_size,
          x_padded_dim.shape.at(2),
          x_padded_dim.shape.at(1),
          x_padded_dim.shape.at(0)));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          y_desc_,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          batch_size,
          y_dim.shape.at(2),
          y_dim.shape.at(1),
          y_dim.shape.at(0)));
    } else {
      /*std::clog << "DEBUG: Pool2DOp: ordinary pooling (implicit zero padding)" << std::endl;
      std::clog << "DEBUG: Pool2DOp:   ordinary input shape: "
          << x_dim.shape.at(0) << " " << x_dim.shape.at(1) << " " << x_dim.shape.at(2) << std::endl;
      std::clog << "DEBUG: Pool2DOp:   ordinary output shape: "
          << y_dim.shape.at(0) << " " << y_dim.shape.at(1) << " " << y_dim.shape.at(2) << std::endl;*/
      CUDNN_CHECK(cudnnSetPooling2dDescriptor(
          pool_desc_,
          pool_mode,
          CUDNN_NOT_PROPAGATE_NAN,
          shape_.window.at(1), shape_.window.at(0),
          shape_.pad.at(1), shape_.pad.at(0),
          shape_.stride.at(1), shape_.stride.at(0)));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          x_desc_,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          batch_size,
          x_dim.shape.at(2),
          x_dim.shape.at(1),
          x_dim.shape.at(0)));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          y_desc_,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          batch_size,
          y_dim.shape.at(2),
          y_dim.shape.at(1),
          y_dim.shape.at(0)));
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    if (shape_.halo_padded_input) {
      CUDNN_CHECK(cudnnPoolingForward(
          conn.cudnn(),
          pool_desc_,
          &alpha,
          x_desc_, x_.val().get(txn, id()).base_as_view().dptr(),
          &beta,
          y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
    } else {
      CUDNN_CHECK(cudnnPoolingForward(
          conn.cudnn(),
          pool_desc_,
          &alpha,
          x_desc_, x_.val().get(txn, id()).as_view().dptr(),
          &beta,
          y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
    }
  }
}

template <>
void CudnnPool2DOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  const float alpha = 1.0f;
  const float beta = 1.0f;

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (x_.grad().accumulate(txn, id(), ctx, [this, batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    if (this->shape_.halo_padded_input) {
      grad->base_as_view().set_constant(0.0f, conn);
    } else {
      grad->as_view().set_constant(0.0f, conn);
    }
  })) {
    //std::clog << "DEBUG: Pool2D: backward(): accumulate" << std::endl;
    GPUConn conn = ctx->conn();

    auto y_dim = y_.val().get_excl(txn, id()).dim();

    cudnnPoolingMode_t pool_mode;
    if (shape_.pooling == kAveragePool) {
      pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    } else if (shape_.pooling == kMaxPool) {
      pool_mode = CUDNN_POOLING_MAX;
    } else {
      assert(0 && "unreachable");
    }

    if (shape_.halo_padded_input) {
      auto x_offset = x_.val().get(txn, id()).offset();
      auto x_stride = x_.val().get(txn, id()).stride();
      assert(x_offset.shape.at(0) == shape_.pad.at(0));
      assert(x_offset.shape.at(1) == shape_.pad.at(1));
      assert(x_offset.shape.at(2) == 0);
      Shape3D x_padded_dim(
          x_dim.shape.at(0) + shape_.pad.at(0) * 2,
          x_dim.shape.at(1) + shape_.pad.at(1) * 2,
          x_dim.shape.at(2));
      assert(x_padded_dim.is_packed(x_stride));
      CUDNN_CHECK(cudnnSetPooling2dDescriptor(
          pool_desc_,
          pool_mode,
          CUDNN_NOT_PROPAGATE_NAN,
          shape_.window.at(1), shape_.window.at(0),
          0, 0,
          shape_.stride.at(1), shape_.stride.at(0)));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          x_desc_,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          batch_size,
          x_padded_dim.shape.at(2),
          x_padded_dim.shape.at(1),
          x_padded_dim.shape.at(0)));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          y_desc_,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          batch_size,
          y_dim.shape.at(2),
          y_dim.shape.at(1),
          y_dim.shape.at(0)));
    } else {
      CUDNN_CHECK(cudnnSetPooling2dDescriptor(
          pool_desc_,
          pool_mode,
          CUDNN_NOT_PROPAGATE_NAN,
          shape_.window.at(1), shape_.window.at(0),
          shape_.pad.at(1), shape_.pad.at(0),
          shape_.stride.at(1), shape_.stride.at(0)));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          x_desc_,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          batch_size,
          x_dim.shape.at(2),
          x_dim.shape.at(1),
          x_dim.shape.at(0)));
      CUDNN_CHECK(cudnnSetTensor4dDescriptor(
          y_desc_,
          CUDNN_TENSOR_NCHW,
          CUDNN_DATA_FLOAT,
          batch_size,
          y_dim.shape.at(2),
          y_dim.shape.at(1),
          y_dim.shape.at(0)));
    }

    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    if (shape_.halo_padded_input) {
      CUDNN_CHECK(cudnnPoolingBackward(
          conn.cudnn(),
          pool_desc_,
          &alpha,
          y_desc_, y_.val().get_excl(txn, id()).as_view().dptr(),
          y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
          x_desc_, x_.val().get(txn, id()).base_as_view().dptr(),
          &beta,
          x_desc_, x_.grad().get_mut(txn, id()).base_as_view().dptr()));
    } else {
      CUDNN_CHECK(cudnnPoolingBackward(
          conn.cudnn(),
          pool_desc_,
          &alpha,
          y_desc_, y_.val().get_excl(txn, id()).as_view().dptr(),
          y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
          x_desc_, x_.val().get(txn, id()).as_view().dptr(),
          &beta,
          x_desc_, x_.grad().get_mut(txn, id()).as_view().dptr()));
    }
  }
}

template <>
void CudnnDownsample2DOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc_,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN,
        downsample_factor_, downsample_factor_,
        0, 0,
        downsample_factor_, downsample_factor_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, y_dim.shape.at(2), y_dim.shape.at(1), y_dim.shape.at(0)));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnPoolingForward(
        conn.cudnn(),
        pool_desc_,
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
  }
}

template <>
void CudnnDownsample2DOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  const float alpha = 1.0f;
  const float beta = 1.0f;

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();

    auto y_dim = y_.val().get_mut(txn, id()).dim();

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc_,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN,
        downsample_factor_, downsample_factor_,
        0, 0,
        downsample_factor_, downsample_factor_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, y_dim.shape.at(2), y_dim.shape.at(1), y_dim.shape.at(0)));

    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnPoolingBackward(
        conn.cudnn(),
        pool_desc_,
        &alpha,
        y_desc_, y_.val().get(txn, id()).as_view().dptr(),
        y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        x_desc_, x_.grad().get_mut(txn, id()).as_view().dptr()));
  }
}

template <>
void CudnnUpsample2DOp<GPUBatchArray3D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);
    auto y_dim = y_.val().get_excl(txn, id()).dim();

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc_,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN,
        upsample_factor_, upsample_factor_,
        0, 0,
        upsample_factor_, upsample_factor_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, y_dim.shape.at(2), y_dim.shape.at(1), y_dim.shape.at(0)));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnPoolingBackward(
        conn.cudnn(),
        pool_desc_,
        &alpha,
        x_desc_, NULL,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        y_desc_, NULL,
        &beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));
    // FIXME: scale the output to undo the window average.
  }
}

template <>
void CudnnUpsample2DOp<GPUBatchArray3D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  const float alpha = 1.0f;
  const float beta = 1.0f;

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    GPUConn conn = ctx->conn();

    auto y_dim = y_.val().get_mut(txn, id()).dim();

    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc_,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN,
        upsample_factor_, upsample_factor_,
        0, 0,
        upsample_factor_, upsample_factor_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, y_dim.shape.at(2), y_dim.shape.at(1), y_dim.shape.at(0)));

    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnPoolingForward(
        conn.cudnn(),
        pool_desc_,
        &alpha,
        y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
        &beta,
        x_desc_, x_.grad().get_mut(txn, id()).as_view().dptr()));
    // FIXME: scale the output to undo the window average.
  }
}

template <>
BatchNormOutput<GPUBatchArray3D<float>, GPUArray1D<float>> CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::Make(
    size_t channels,
    shared_ptr<bool> switch_var,
    shared_ptr<Var<GPUBatchArray3D<float>>> input,
    //function<GPUBatchArray3D<float>(TxnId, NodeId)> output_alloc_fn,
    shared_ptr<Context> ctx)
{
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  auto batch_mean_var = array_src<GPUArray1D<float>, Shape1D>(Shape1D(channels), ctx);
  auto batch_variance_var = array_src<GPUArray1D<float>, Shape1D>(Shape1D(channels), ctx);
  batch_stats.insert(batch_mean_var->vars().at(0));
  batch_stats.insert(batch_variance_var->vars().at(0));

  auto fixed_mean_var =
      initialize<GPUArray1D<float>>(
          //[](auto val, auto ctx) { zeros_init_fn(val, ctx); },
          zeros_init_gen<GPUArray1D<float>>(),
          array_src<GPUArray1D<float>, Shape1D>(Shape1D(channels), ctx),
          ctx);
  auto fixed_variance_var =
      initialize<GPUArray1D<float>>(
          //[](auto val, auto ctx) { ones_init_fn(val, ctx); },
          ones_init_gen<GPUArray1D<float>>(),
          array_src<GPUArray1D<float>, Shape1D>(Shape1D(channels), ctx),
          ctx);
  fixed_stats.insert(fixed_mean_var->vars().at(0));
  fixed_stats.insert(fixed_variance_var->vars().at(0));

  auto scale_var =
      initialize<GPUArray1D<float>>(
          //[](auto val, auto ctx) { ones_init_fn(val, ctx); },
          ones_init_gen<GPUArray1D<float>>(),
          array_src<GPUArray1D<float>, Shape1D>(Shape1D(channels), ctx),
          ctx);
  auto shift_var =
      initialize<GPUArray1D<float>>(
          //[](auto val, auto ctx) { zeros_init_fn(val, ctx); },
          zeros_init_gen<GPUArray1D<float>>(),
          array_src<GPUArray1D<float>, Shape1D>(Shape1D(channels), ctx),
          ctx);
  params.insert(scale_var->vars().at(0));
  params.insert(shift_var->vars().at(0));
  grads.insert(scale_var->vars().at(1));
  grads.insert(shift_var->vars().at(1));

  auto op = make_shared<CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>>(
      switch_var,
      input,
      batch_mean_var,
      batch_variance_var,
      fixed_mean_var,
      fixed_variance_var,
      scale_var,
      shift_var,
      ctx);

  BatchNormOutput<GPUBatchArray3D<float>, GPUArray1D<float>> output = {
    .op = op,
    .batch_mean = batch_mean_var,
    .batch_variance = batch_variance_var,
    .fixed_mean = fixed_mean_var,
    .fixed_variance = fixed_variance_var,
    .scale = scale_var,
    .shift = shift_var,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return output;
}

template <>
Shape1D CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::_get_stat_shape(Shape3D in_shape) {
  return Shape1D(in_shape.shape.at(2));
}

template <>
void CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);

    auto y_dim = y_.val().get_excl(txn, id()).dim();
    assert(x_dim.shape.at(0) == y_dim.shape.at(0));
    assert(x_dim.shape.at(1) == y_dim.shape.at(1));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));

    /*CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, y_dim.shape.at(2), y_dim.shape.at(1), y_dim.shape.at(0)));*/

    /*x_.val().get(txn, id()).as_view()
      .set_cudnn_tensor_desc(x_desc_, CUDNN_TENSOR_NCHW);
    y_.val().get_excl(txn, id()).as_view()
      .set_cudnn_tensor_desc(y_desc_, CUDNN_TENSOR_NCHW);*/

    set_cudnn_tensor_desc_batch3d_ex(
        x_desc_,
        x_.val().get(txn, id()).dim(),
        x_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);
    set_cudnn_tensor_desc_batch3d_ex(
        y_desc_,
        y_.val().get_excl(txn, id()).dim(),
        y_.val().get_excl(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        stat_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, x_dim.shape.at(2), 1, 1));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const double factor = 1.0;
    const double epsilon = 1.0e-5;

    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    if (*switch_var_) {
      //std::clog << "DEBUG: BatchNorm2DOp: fwd: switch on (use fixed stats)" << std::endl;
      CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
          conn.cudnn(),
          CUDNN_BATCHNORM_SPATIAL,
          &alpha,
          &beta,
          x_desc_, x_.val().get(txn, id()).as_view().dptr(),
          y_desc_, y_.val().get_excl(txn, id()).as_view().dptr(),
          stat_desc_,
          scale_.val().get(txn, id()).dptr(),
          shift_.val().get(txn, id()).dptr(),
          fixed_mean_.val().get(txn, id()).dptr(),
          fixed_variance_.val().get(txn, id()).dptr(),
          epsilon));
    } else {
      //std::clog << "DEBUG: BatchNorm2DOp: fwd: switch off (use batch stats)" << std::endl;
      assert(batch_mean_.val().overwrite(txn, id(), ctx));
      assert(batch_variance_.val().overwrite(txn, id(), ctx));
      assert(tmp_mean_.val().overwrite(txn, id(), ctx));
      assert(tmp_variance_.val().overwrite(txn, id(), ctx));
      CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
          conn.cudnn(),
          CUDNN_BATCHNORM_SPATIAL,
          &alpha,
          &beta,
          x_desc_, x_.val().get(txn, id()).as_view().dptr(),
          y_desc_, y_.val().get_excl(txn, id()).as_view().dptr(),
          stat_desc_,
          scale_.val().get(txn, id()).dptr(),
          shift_.val().get(txn, id()).dptr(),
          factor,
          batch_mean_.val().get_excl(txn, id()).dptr(),
          batch_variance_.val().get_excl(txn, id()).dptr(),
          epsilon,
          tmp_mean_.val().get_excl(txn, id()).dptr(),
          tmp_variance_.val().get_excl(txn, id()).dptr()));
    }
  }
}

template <>
void CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  })) {
    assert(scale_.grad().accumulate(txn, id(), ctx, [ctx](auto grad) {
      GPUConn conn = ctx->conn();
      grad->set_constant(0.0f, conn);
    }));
    assert(shift_.grad().accumulate(txn, id(), ctx, [ctx](auto grad) {
      GPUConn conn = ctx->conn();
      grad->set_constant(0.0f, conn);
    }));

    GPUConn conn = ctx->conn();

    auto y_dim = y_.val().get(txn, id()).dim();
    assert(x_dim.shape.at(0) == y_dim.shape.at(0));
    assert(x_dim.shape.at(1) == y_dim.shape.at(1));
    assert(x_dim.shape.at(2) == y_dim.shape.at(2));

    /*CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        dx_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        dy_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, y_dim.shape.at(2), y_dim.shape.at(1), y_dim.shape.at(0)));*/

    set_cudnn_tensor_desc_batch3d_ex(
        x_desc_,
        x_.val().get(txn, id()).dim(),
        x_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);
    set_cudnn_tensor_desc_batch3d_ex(
        y_desc_,
        y_.val().get(txn, id()).dim(),
        y_.val().get(txn, id()).stride(),
        batch_size,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        stat_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, x_dim.shape.at(2), 1, 1));

    const float alpha = 1.0f;
    const float beta = 1.0f;
    const double epsilon = 1.0e-5;

    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    if (*switch_var_) {
      // FIXME
      assert(0 && "unimplemented");
    } else {
      //std::clog << "DEBUG: BatchNorm2DOp: bwd: switch off (use batch stats)" << std::endl;
      CUDNN_CHECK(cudnnBatchNormalizationBackward(
          conn.cudnn(),
          CUDNN_BATCHNORM_SPATIAL,
          &alpha, &beta,
          &alpha, &beta,
          x_desc_, x_.val().get(txn, id()).as_view().dptr(),
          y_desc_, y_.grad().get(txn, id()).as_view().dptr(),
          x_desc_, x_.grad().get_mut(txn, id()).as_view().dptr(),
          stat_desc_,
          scale_.val().get(txn, id()).dptr(),
          scale_.grad().get_mut(txn, id()).dptr(),
          shift_.grad().get_mut(txn, id()).dptr(),
          epsilon,
          tmp_mean_.val().get(txn, id()).dptr(),
          tmp_variance_.val().get(txn, id()).dptr()));
    }
  }
}

template <>
BatchNormOutput<GPUBatchArray3D<float>, GPUArray1D<float>> batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(
    size_t channels,
    shared_ptr<bool> switch_var,
    shared_ptr<Var<GPUBatchArray3D<float>>> input,
    shared_ptr<Context> ctx)
{
  return CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::Make(channels, switch_var, input, ctx);
}

template <>
Shape0D CudnnSoftmaxNLLLossOp<GPUBatchArray1D<float>, GPUBatchArray0D<uint32_t>, GPUBatchArray0D<float>>::_get_output_shape(Shape1D in_shape) {
  return Shape0D();
}

template <>
void CudnnSoftmaxNLLLossOp<GPUBatchArray1D<float>, GPUBatchArray0D<uint32_t>, GPUBatchArray0D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();

    //std::clog << "DEBUG: SoftmaxNLLLossOp: batch sz: " << batch_size << " dim: " << x_dim.shape << std::endl;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape, 1, 1));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape, 1, 1));

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnSoftmaxForward(
        conn.cudnn(),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));

    assert(loss_.val().overwrite(txn, id(), ctx));
    softmax_nll_loss_fwd(
        x_dim.shape,
        batch_size,
        y_.val().get_excl(txn, id()).as_view().dptr(),
        label_.val().get(txn, id()).as_view().dptr(),
        loss_.val().get_excl(txn, id()).as_view().dptr(),
        conn.stream());
  }
}

template <>
void CudnnSoftmaxNLLLossOp<GPUBatchArray1D<float>, GPUBatchArray0D<uint32_t>, GPUBatchArray0D<float>>::_backward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (x_.grad().accumulate(txn, id(), ctx, [batch_size, ctx](auto grad) {
    GPUConn conn = ctx->conn();
    grad->set_batch_size(batch_size);
    grad->as_view().set_constant(0.0f, conn);
  }))
  {
    GPUConn conn = ctx->conn();
    softmax_nll_loss_bwd(
        x_dim.shape,
        batch_size,
        y_.val().get(txn, id()).as_view().dptr(),
        label_.val().get(txn, id()).as_view().dptr(),
        loss_.grad().get(txn, id()).as_view().dptr(),
        x_.grad().get_mut(txn, id()).as_view().dptr(),
        conn.stream());
  }
}

template <>
void CudnnSoftmaxNLLLossOp<GPUBatchArray3D<float>, GPUBatchArray2D<uint32_t>, GPUBatchArray2D<float>>::_forward(TxnId txn) {
  size_t batch_size = x_.val().get(txn, id()).batch_size();
  auto x_dim = x_.val().get(txn, id()).dim();

  GPUContext* ctx = dynamic_cast<GPUContext*>(ctx_.get());

  if (y_.val().overwrite(txn, id(), ctx)) {
    GPUConn conn = ctx->conn();

    y_.val().get_excl(txn, id()).set_batch_size(batch_size);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc_,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, x_dim.shape.at(2), x_dim.shape.at(1), x_dim.shape.at(0)));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnSoftmaxForward(
        conn.cudnn(),
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha,
        x_desc_, x_.val().get(txn, id()).as_view().dptr(),
        &beta,
        y_desc_, y_.val().get_excl(txn, id()).as_view().dptr()));

    assert(0 && "unimplemented");

    // TODO
    /*softmax_nll_loss_2d_fwd(
        x_dim.shape,
        batch_size,
        y_.val().get_excl(txn, id()).dptr(),
        label_.val().get(txn, id()).dptr(),
        loss_.val().get_excl(txn, id()).dptr(),
        conn.stream());*/
  }
}

} // namespace arraydiff
