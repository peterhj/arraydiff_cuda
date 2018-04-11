#ifndef ARRAYDIFF_OP_HH
#define ARRAYDIFF_OP_HH

#include "arraydiff/array_abstract.hh"
#include "arraydiff/array_gpu.hh"
#include "arraydiff/context.hh"
#include "arraydiff/cuda_common.hh"

#include <cublas_v2.h>
#include <cudnn.h>
#include <nccl.h>

#include <array>
#include <cassert>
#include <experimental/any>
#include <experimental/optional>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
//#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace arraydiff {

using std::array;
using std::experimental::any;
//using std::experimental::nullopt_t;
using std::experimental::optional;
using std::function;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;

class NodeId {
public:
  NodeId();

  uint64_t raw_id() const {
    return this->id_;
  }

private:
  uint64_t id_;
};

class EpochId {
public:
  EpochId();

  uint64_t raw_id() const {
    return this->id_;
  }

private:
  uint64_t id_;
};

class TxnId {
public:
  TxnId();

  uint64_t raw_id() const {
    return this->id_;
  }

private:
  uint64_t id_;
};

enum VarKind {
  VAR_KIND_VAL,
  VAR_KIND_GRAD,
};

class VarId {
public:
  VarId();

  uint64_t raw_id() const {
    return this->id_;
  }

  VarKind kind() const {
    return kind_;
  }

private:
  uint64_t id_;
  VarKind kind_;
};

class VarSymbol {
public:
  VarSymbol();

  uint64_t raw_id() const {
    return this->id_;
  }

private:
  uint64_t id_;
};

class VarIdSet {
public:
  static VarIdSet Singleton(VarId v) {
    VarIdSet vs;
    vs.insert(v);
    return vs;
  }

  VarIdSet()
    : locked_(false) {}
  ~VarIdSet() {}

  size_t size() const {
    return vars_.size();
  }

  void lock() {
    assert(!locked_);
    mask_.clear();
    locked_ = true;
  }
  void unlock() {
    assert(locked_);
    locked_ = false;
    mask_.clear();
  }
  bool contains(VarId v) {
    //return vars_.find(v.raw_id()) != vars_.end();
    return vars_.count(v.raw_id()) > 0;
  }
  void insert(VarId v) {
    assert(!locked_);
    //vars_.emplace(v.raw_id());
    vars_.insert(v.raw_id());
  }
  void insert_all(const VarIdSet& vs) {
    assert(!locked_);
    for (auto iter = vs.vars_.begin(); iter != vs.vars_.end(); ++iter) {
      //vars_.emplace(*iter);
      vars_.insert(*iter);
    }
  }
  bool mask(VarId v) {
    assert(locked_);
    if (vars_.count(v.raw_id()) == 0) {
      return false;
    }
    if (mask_.count(v.raw_id()) == 0) {
      //mask_.emplace(v.raw_id());
      mask_.insert(v.raw_id());
      return true;
    }
    return false;
  }

private:
  bool locked_;
  std::unordered_set<uint64_t> vars_;
  std::unordered_set<uint64_t> mask_;
};

template <typename V>
class TxnVar;

template <typename V>
class TxnVarBuf {
friend class TxnVar<V>;
private:
  optional<TxnId> curr_txn_;
  bool rollover_;
  std::unordered_set<uint64_t> reads_;
  std::unordered_set<uint64_t> freezes_;
  std::unordered_map<uint64_t, uint64_t> writes_;
  std::set<std::pair<uint64_t, uint64_t>> read_writes_;
  std::unordered_set<uint64_t> coarse_rws_;
  optional<V> buffer_;
};

template <typename V>
class TxnVar {
public:
  typedef typename V:: Shape VShape_;

  TxnVar() {}

  explicit TxnVar(VShape_ shape)
    : var_(), symbol_(), shape_(shape), batch_shape_(0), init_(true)
  {
    offset_ = VShape_::Zero();
    stride_ = shape_.packed_stride();
    bufs_.push_back(make_shared<TxnVarBuf<V>>());
  }

  TxnVar(VShape_ shape, size_t batch_shape)
    : var_(), symbol_(), shape_(shape), batch_shape_(batch_shape), init_(true)
  {
    offset_ = VShape_::Zero();
    stride_ = shape_.packed_stride();
    bufs_.push_back(make_shared<TxnVarBuf<V>>());
  }

  TxnVar(VShape_ shape, VShape_ offset, VShape_ stride, size_t batch_shape)
    : var_(), symbol_(), shape_(shape), offset_(offset), stride_(stride), batch_shape_(batch_shape), init_(true)
  {
    bufs_.push_back(make_shared<TxnVarBuf<V>>());
  }

  TxnVar clone() const {
    assert(init_);
    TxnVar other;
    other.var_ = var_;
    other.symbol_ = VarSymbol();
    other.shape_ = shape_;
    other.offset_ = offset_;
    other.stride_ = stride_;
    other.batch_shape_ = batch_shape_;
    other.bufs_ = bufs_;
    other.init_ = true;
    return other;
  }

  VarId var() const {
    assert(init_);
    return var_;
  }

  void rollover(TxnId txn) {
    assert(init_);
    TxnVarBuf<V>& buf = *bufs_.at(0);
    if (static_cast<bool>(buf.curr_txn_)) {
      if (buf.curr_txn_.value().raw_id() == txn.raw_id()) {
      } else {
        buf.rollover_ = true;
      }
    } else {
      buf.rollover_ = true;
    }
  }

  bool overwrite(TxnId txn, NodeId node_id, Context* ctx) {
    assert(init_);
    typename V:: Ctx* v_ctx = dynamic_cast<typename V:: Ctx*>(ctx);
    assert(v_ctx != NULL);
    return overwrite(txn, node_id, v_ctx);
  }

  bool overwrite(TxnId txn, NodeId node_id, typename V:: Ctx* ctx) {
    assert(init_);
    assert(ctx != NULL);
    TxnVarBuf<V>& buf = *bufs_.at(0);
    bool incomplete_write = false;
    bool new_txn = false;
    if (!static_cast<bool>(buf.curr_txn_)) {
      incomplete_write = true;
      new_txn = true;
    } else {
      TxnId curr_txn = buf.curr_txn_.value();
      if (curr_txn.raw_id() != txn.raw_id()) {
        incomplete_write = true;
        new_txn = true;
      } else {
        assert(buf.reads_.find(node_id.raw_id()) == buf.reads_.end());
        assert(buf.freezes_.find(node_id.raw_id()) == buf.freezes_.end());
        assert(buf.coarse_rws_.find(node_id.raw_id()) == buf.coarse_rws_.end());
        bool written = buf.writes_.find(node_id.raw_id()) != buf.writes_.end();
        if (written) {
          assert(1 == buf.writes_.size());
          assert(symbol_.raw_id() == (*buf.writes_.find(node_id.raw_id())).second);
        }
        incomplete_write = written;
      }
    }
    if (new_txn || buf.rollover_) {
      assert(incomplete_write);
      buf.curr_txn_.emplace(txn);
      buf.reads_.clear();
      buf.freezes_.clear();
      buf.writes_.clear();
      buf.read_writes_.clear();
      buf.coarse_rws_.clear();
      if (!static_cast<bool>(buf.buffer_)) {
        buf.buffer_.emplace(V::ZerosPadded(shape_, offset_, stride_, batch_shape_, ctx));
      }
      buf.rollover_ = false;
    }
    return incomplete_write;
  }

  bool accumulate_old(TxnId txn, NodeId node_id, typename V:: Ctx* ctx, function<void(V&)> init_fn) {
    assert(init_);
    assert(ctx != NULL);
    TxnVarBuf<V>& buf = *bufs_.at(0);
    bool incomplete_write = false;
    bool new_txn = false;
    if (!static_cast<bool>(buf.curr_txn_)) {
      incomplete_write = true;
      new_txn = true;
    } else {
      TxnId curr_txn = buf.curr_txn_.value();
      if (curr_txn.raw_id() != txn.raw_id()) {
        incomplete_write = true;
        new_txn = true;
      } else {
        assert(buf.reads_.find(node_id.raw_id()) == buf.reads_.end());
        assert(buf.writes_.find(node_id.raw_id()) == buf.writes_.end());
        bool rw = buf.read_writes_.find(std::make_pair(node_id.raw_id(), symbol_.raw_id())) != buf.read_writes_.end();
        bool coarse_rw = buf.coarse_rws_.find(node_id.raw_id()) != buf.coarse_rws_.end();
        if (!coarse_rw) {
          assert(!rw);
        }
        if (rw) {
          assert(coarse_rw);
        }
        incomplete_write = !rw;
      }
    }
    if (new_txn || buf.rollover_) {
      assert(incomplete_write);
      buf.curr_txn_.emplace(txn);
      buf.reads_.clear();
      buf.freezes_.clear();
      buf.writes_.clear();
      buf.read_writes_.clear();
      buf.coarse_rws_.clear();
      if (!static_cast<bool>(buf.buffer_)) {
        buf.buffer_.emplace(V::ZerosPadded(shape_, offset_, stride_, batch_shape_, ctx));
      }
      if (!buf.rollover_) {
        (init_fn)(buf.buffer_.value());
      }
      buf.rollover_ = false;
    }
    return incomplete_write;
  }

  bool accumulate(TxnId txn, NodeId node_id, typename V:: Ctx* ctx, function<void(V*)> init_fn) {
    assert(init_);
    assert(ctx != NULL);
    TxnVarBuf<V>& buf = *bufs_.at(0);
    bool incomplete_write = false;
    bool new_txn = false;
    if (!static_cast<bool>(buf.curr_txn_)) {
      incomplete_write = true;
      new_txn = true;
    } else {
      TxnId curr_txn = buf.curr_txn_.value();
      if (curr_txn.raw_id() != txn.raw_id()) {
        incomplete_write = true;
        new_txn = true;
      } else {
        assert(buf.reads_.find(node_id.raw_id()) == buf.reads_.end());
        assert(buf.writes_.find(node_id.raw_id()) == buf.writes_.end());
        bool rw = buf.read_writes_.find(std::make_pair(node_id.raw_id(), symbol_.raw_id())) != buf.read_writes_.end();
        bool coarse_rw = buf.coarse_rws_.find(node_id.raw_id()) != buf.coarse_rws_.end();
        if (!coarse_rw) {
          assert(!rw);
        }
        if (rw) {
          assert(coarse_rw);
        }
        incomplete_write = !rw;
      }
    }
    if (new_txn || buf.rollover_) {
      assert(incomplete_write);
      buf.curr_txn_.emplace(txn);
      buf.reads_.clear();
      buf.freezes_.clear();
      buf.writes_.clear();
      buf.read_writes_.clear();
      buf.coarse_rws_.clear();
      if (!static_cast<bool>(buf.buffer_)) {
        buf.buffer_.emplace(V::ZerosPadded(shape_, offset_, stride_, batch_shape_, ctx));
      }
      if (!buf.rollover_) {
        (init_fn)(&buf.buffer_.value());
      }
      buf.rollover_ = false;
    }
    return incomplete_write;
  }

  V& get(TxnId txn, NodeId node_id) {
    assert(init_);
    TxnVarBuf<V>& buf = *bufs_.at(0);
    bool new_txn = false;
    if (!static_cast<bool>(buf.curr_txn_)) {
      new_txn = true;
    } else {
      new_txn = buf.curr_txn_.value().raw_id() != txn.raw_id();
    }
    if (new_txn) {
      if (!buf.rollover_) {
        std::clog << "WARNING: TxnVar: get(): bad rollover:"
            " txn: " << txn.raw_id() <<
            " var id: " << var().raw_id() << std::endl;
      }
      assert(buf.rollover_);
      buf.curr_txn_.emplace(txn);
      buf.reads_.clear();
      buf.freezes_.clear();
      buf.writes_.clear();
      buf.read_writes_.clear();
      buf.coarse_rws_.clear();
      buf.rollover_ = false;
    }
    if (buf.writes_.find(node_id.raw_id()) != buf.writes_.end() ||
        buf.coarse_rws_.find(node_id.raw_id()) != buf.coarse_rws_.end())
    {
      if (buf.freezes_.find(node_id.raw_id()) == buf.freezes_.end()) {
        buf.freezes_.emplace(node_id.raw_id());
      }
    }
    assert(static_cast<bool>(buf.buffer_));
    return buf.buffer_.value();
  }

  V& get_excl(TxnId txn, NodeId node_id) {
    assert(init_);
    TxnVarBuf<V>& buf = *bufs_.at(0);
    bool new_txn = false;
    if (!static_cast<bool>(buf.curr_txn_)) {
      new_txn = true;
    } else {
      new_txn = buf.curr_txn_.value().raw_id() != txn.raw_id();
    }
    assert(!new_txn);
    assert(!buf.rollover_);
    assert(buf.reads_.find(node_id.raw_id()) == buf.reads_.end());
    //assert(buf.freezes_.find(node_id.raw_id()) == buf.freezes_.end());
    assert(buf.coarse_rws_.find(node_id.raw_id()) == buf.coarse_rws_.end());
    if (buf.writes_.find(node_id.raw_id()) != buf.writes_.end()) {
      assert(1 == buf.writes_.size());
      assert(symbol_.raw_id() == (*buf.writes_.find(node_id.raw_id())).second);
    } else {
      buf.writes_.emplace(node_id.raw_id(), symbol_.raw_id());
    }
    assert(static_cast<bool>(buf.buffer_));
    return buf.buffer_.value();
  }

  V& get_mut(TxnId txn, NodeId node_id) {
    assert(init_);
    TxnVarBuf<V>& buf = *bufs_.at(0);
    bool new_txn = false;
    if (!static_cast<bool>(buf.curr_txn_)) {
      new_txn = true;
    } else {
      new_txn = buf.curr_txn_.value().raw_id() != txn.raw_id();
    }
    assert(!new_txn);
    assert(!buf.rollover_);
    assert(buf.reads_.find(node_id.raw_id()) == buf.reads_.end());
    //assert(buf.freezes_.find(node_id.raw_id()) == buf.freezes_.end());
    assert(buf.writes_.find(node_id.raw_id()) == buf.writes_.end());
    bool rw = buf.read_writes_.find(std::make_pair(node_id.raw_id(), symbol_.raw_id())) != buf.read_writes_.end();
    bool coarse_rw = buf.coarse_rws_.find(node_id.raw_id()) != buf.coarse_rws_.end();
    if (!coarse_rw) {
      assert(!rw);
    }
    if (rw) {
      assert(coarse_rw);
    } else {
      buf.read_writes_.emplace(std::make_pair(node_id.raw_id(), symbol_.raw_id()));
      buf.coarse_rws_.emplace(node_id.raw_id());
    }
    assert(static_cast<bool>(buf.buffer_));
    return buf.buffer_.value();
  }

private:
  VarId var_;
  VarSymbol symbol_;
  VShape_ shape_;
  VShape_ offset_;
  VShape_ stride_;
  size_t batch_shape_;
  vector<shared_ptr<TxnVarBuf<V>>> bufs_;
  bool init_;
};

template <typename V>
class VarData {
public:
  typedef typename V:: Shape VShape_;

  VarData() {}

  explicit VarData(VShape_ shape)
    : shape_(shape), batch_shape_(0),
      val_(shape), grad_(shape), init_(true)
  {
    offset_ = VShape_::Zero();
    stride_ = shape.packed_stride();
  }

  VarData(VShape_ shape, size_t batch_shape)
    : shape_(shape), batch_shape_(batch_shape),
      val_(shape, batch_shape), grad_(shape, batch_shape), init_(true)
  {
    offset_ = VShape_::Zero();
    stride_ = shape.packed_stride();
  }

  VarData(VShape_ shape, size_t batch_shape, TxnVar<V> val, TxnVar<V> grad)
    : shape_(shape), batch_shape_(batch_shape),
      val_(val), grad_(grad), init_(true)
  {
    offset_ = VShape_::Zero();
    stride_ = shape.packed_stride();
  }

  VarData(VShape_ shape, VShape_ offset, VShape_ stride, size_t batch_shape)
    : shape_(shape), offset_(offset), stride_(stride), batch_shape_(batch_shape),
      val_(shape, offset, stride, batch_shape), grad_(shape, offset, stride, batch_shape), init_(true) {}

  VarData(VShape_ shape, VShape_ offset, VShape_ stride, size_t batch_shape, TxnVar<V> val, TxnVar<V> grad)
    : shape_(shape), offset_(offset), stride_(stride), batch_shape_(batch_shape),
      val_(val), grad_(grad), init_(true) {}

  VShape_ shape() const {
    assert(init_);
    return shape_;
  }
  VShape_ offset() const {
    assert(init_);
    return offset_;
  }
  VShape_ stride() const {
    assert(init_);
    return stride_;
  }
  size_t batch_shape() const {
    assert(init_);
    return batch_shape_;
  }

  VarData<V> clone() const {
    assert(init_);
    return VarData(shape_, offset_, stride_, batch_shape_, val_.clone(), grad_.clone());
  }

  /*const TxnVar<V>& const_val() const {
    assert(init_);
    return this->val_;
  }*/
  TxnVar<V>& val() {
    assert(init_);
    return this->val_;
  }
  /*const TxnVar<V>& const_grad() const {
    assert(init_);
    return this->grad_;
  }*/
  TxnVar<V>& grad() {
    assert(init_);
    return this->grad_;
  }

  void rollover_all(TxnId txn, VarIdSet& vars) {
    /*std::clog << "DEBUG: VarData: rollover_all():"
        " txn: " << txn.raw_id() <<
        " val: " << val().var().raw_id() <<
        " grad: " << grad().var().raw_id() << std::endl;*/
    assert(init_);
    if (vars.contains(val().var())) {
      val().rollover(txn);
    }
    if (vars.contains(grad().var())) {
      grad().rollover(txn);
    }
  }

private:
  VShape_ shape_;
  VShape_ offset_;
  VShape_ stride_;
  size_t batch_shape_;
  TxnVar<V> val_;
  TxnVar<V> grad_;
  bool init_;
};

class OperatorStackEntry {
public:
  EpochId epoch;
  size_t push_count;
  size_t pop_count;
};

class OperatorStack {
public:
  explicit OperatorStack(NodeId node_id);

  /*size_t _set_in_degree(size_t in_degree) {
    this->in_degree_ = in_degree;
  }*/

  size_t push(EpochId epoch);
  size_t degree(EpochId epoch);
  size_t pop(EpochId epoch);

private:
  NodeId node_id_;
  size_t in_degree_;
  vector<OperatorStackEntry> entries_;
};

class Op {
public:
  Op() : node_id_(), stack_(node_id_) {}
  virtual ~Op() {}

  NodeId id() const {
    return this->node_id_;
  }

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) = 0;
  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) = 0;

  virtual size_t _load_val(TxnId txn, VarIdSet& vars, size_t offset, any* reader) {
    (void)txn;
    (void)vars;
    (void)reader;
    return offset;
  }
  virtual size_t _load_grad(TxnId txn, VarIdSet& vars, size_t offset, any* reader) {
    (void)txn;
    (void)vars;
    (void)reader;
    return offset;
  }
  virtual size_t _store_val(TxnId txn, VarIdSet& vars, size_t offset, any* writer) {
    (void)txn;
    (void)vars;
    (void)writer;
    return offset;
  }
  virtual size_t _store_grad(TxnId txn, VarIdSet& vars, size_t offset, any* writer) {
    (void)txn;
    (void)vars;
    (void)writer;
    return offset;
  }
  virtual void _init(TxnId txn, std::mt19937_64* rng) {
    (void)txn;
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) = 0;
  virtual void _set_grad_sink(TxnId txn) {
    (void)txn;
    assert(0 && "unimplemented");
  }
  virtual void _forward(TxnId txn) = 0;
  virtual void _backward(TxnId txn) = 0;
  //virtual void _sync() { assert(0 && "unimplemented"); }
  //virtual void _sync() {}

  virtual string _debug_op_label() { return "Op"; }

  size_t load_val(TxnId txn, VarIdSet& vars, size_t offset, any* reader);
  size_t load_grad(TxnId txn, VarIdSet& vars, size_t offset, any* reader);
  size_t store_val(TxnId txn, VarIdSet& vars, size_t offset, any* writer);
  size_t store_grad(TxnId txn, VarIdSet& vars, size_t offset, any* writer);

  void init(TxnId txn, std::mt19937_64* rng);
  void persist(TxnId txn, VarIdSet& vars);
  void eval(TxnId txn);
  void gradient(TxnId txn);

protected:
  NodeId node_id_;
  OperatorStack stack_;
};

template <typename V>
class Var : public virtual Op {
public:
  Var() {}
  virtual ~Var() {}

  virtual typename V:: Shape shape() = 0;
  virtual typename V:: Shape offset() = 0;
  virtual typename V:: Shape stride() = 0;
  virtual size_t batch_shape() = 0;
  virtual VarData<V> data() = 0;
  virtual vector<VarId> vars() = 0;
};

template <typename V>
class Sink {
public:
  explicit Sink(shared_ptr<Var<V>> sink_var)
    : sink_var_(sink_var) {}

  void gradient(TxnId txn) {
    EpochId epoch_fwd;
    sink_var_->_push(epoch_fwd, [txn](Op* op) {
      op->_forward(txn);
    });
    sink_var_->_pop(epoch_fwd, [](Op* _var) {});
    sink_var_->_set_grad_sink(txn);
    EpochId epoch_bwd;
    sink_var_->_push(epoch_bwd, [](Op* _var) {});
    sink_var_->_pop(epoch_bwd, [txn](Op* op) {
      op->_backward(txn);
    });
    //sink_var_->_sync();
  }

private:
  shared_ptr<Var<V>> sink_var_;
};

template <typename V>
class SrcOp : public virtual Op, public virtual Var<V> {
public:
  explicit SrcOp(typename V:: Shape shape)
    : x_(shape)
  {
    //std::clog << "DEBUG: SrcOp: constructing..." << std::endl;
    //std::clog << "DEBUG: SrcOp: constructed" << std::endl;
  }
  SrcOp(typename V:: Shape shape, size_t batch_shape)
    : x_(shape, batch_shape)
  {
    //std::clog << "DEBUG: SrcOp: constructing..." << std::endl;
    //std::clog << "DEBUG: SrcOp: constructed" << std::endl;
  }

  virtual ~SrcOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    /*std::clog << "DEBUG: SrcOp: _persist():"
        " txn: " << txn.raw_id() <<
        " val: " << x_.val().var().raw_id() << std::endl;
    std::clog << "DEBUG: SrcOp: _persist():   shape: ";
    shape()._debug_print();
    std::clog << std::endl;
    std::clog << "DEBUG: SrcOp: _persist():   batch shape: " << batch_shape() << std::endl;
    std::clog << "DEBUG: SrcOp: _persist():   roll val: " <<
        vars.contains(x_.val().var()) << std::endl;
    std::clog << "DEBUG: SrcOp: _persist():   roll grad: " <<
        vars.contains(x_.grad().var()) << std::endl;*/
    x_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn) {
    (void)txn;
  }

  virtual void _backward(TxnId txn) {
    (void)txn;
  }

  virtual string _debug_op_label() { return "SrcOp"; }

  virtual typename V:: Shape shape() {
    return x_.shape();
  }

  virtual typename V:: Shape offset() {
    return x_.offset();
  }

  virtual typename V:: Shape stride() {
    return x_.stride();
  }

  virtual size_t batch_shape() {
    return x_.batch_shape();
  }

  virtual VarData<V> data() {
    return x_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(x_.val().var());
    vs.push_back(x_.grad().var());
    return vs;
  }

private:
  VarData<V> x_;
};

template <typename V>
class IOOp : public virtual Op, public virtual Var<V> {
public:
  IOOp(shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), input_(input), x_(input->data()) {}
  virtual ~IOOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual size_t _load_val(TxnId txn, VarIdSet& vars, size_t offset, any* reader) {
    //std::clog << "DEBUG: IOOp: _load_val" << std::endl;
    if (vars.mask(x_.val().var())) {
      //std::clog << "DEBUG: IOOp: _load_val: mask passed" << std::endl;
      assert(x_.val().overwrite(txn, id(), ctx_.get()));
      return x_.val().get_excl(txn, id()).load_any(offset, reader, ctx_);
    }
    return offset;
  }

  virtual size_t _load_grad(TxnId txn, VarIdSet& vars, size_t offset, any* reader) {
    if (vars.mask(x_.grad().var())) {
      assert(x_.grad().overwrite(txn, id(), ctx_.get()));
      return x_.grad().get_excl(txn, id()).load_any(offset, reader, ctx_);
    }
    return offset;
  }

  virtual size_t _store_val(TxnId txn, VarIdSet& vars, size_t offset, any* writer) {
    if (vars.mask(x_.val().var())) {
      return x_.val().get(txn, id()).store_any(offset, writer, ctx_);
    }
    return offset;
  }

  virtual size_t _store_grad(TxnId txn, VarIdSet& vars, size_t offset, any* writer) {
    if (vars.mask(x_.grad().var())) {
      return x_.grad().get(txn, id()).store_any(offset, writer, ctx_);
    }
    return offset;
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    x_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn) {
    (void)txn;
  }

  virtual void _backward(TxnId txn) {
    (void)txn;
  }

  virtual string _debug_op_label() { return "IOOp"; }

  virtual typename V:: Shape shape() {
    return x_.shape();
  }

  virtual typename V:: Shape offset() {
    return x_.offset();
  }

  virtual typename V:: Shape stride() {
    return x_.stride();
  }

  virtual size_t batch_shape() {
    return x_.batch_shape();
  }

  virtual VarData<V> data() {
    return x_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(x_.val().var());
    vs.push_back(x_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
};

template <typename V>
shared_ptr<Var<V>> io_var(shared_ptr<Var<V>> input, shared_ptr<Context> ctx) {
  return make_shared<IOOp<V>>(input, ctx);
}

template <typename V, typename Shape>
shared_ptr<IOOp<V>> array_src(Shape dim, shared_ptr<Context> ctx);

template <typename V, typename Shape>
shared_ptr<IOOp<V>> batch_array_src(Shape dim, size_t batch_size, shared_ptr<Context> ctx);

template <typename V, typename Shape>
shared_ptr<IOOp<V>> batch_array_src_with_halo_pad(size_t halo_pad, Shape dim, size_t batch_size, shared_ptr<Context> ctx);

template <typename V>
class InitOp : public virtual Op, public virtual Var<V> {
public:
  InitOp(function<void(V*, std::mt19937_64*, shared_ptr<Context>)> init_fn, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), init_fn_(init_fn), input_(input), x_(input->data()) {}
  virtual ~InitOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _init(TxnId txn, std::mt19937_64* rng) {
    assert(x_.val().overwrite(txn, id(), ctx_.get()));
    (init_fn_)(&x_.val().get_excl(txn, id()), rng, ctx_);
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    x_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn) {
    (void)txn;
  }

  virtual void _backward(TxnId txn) {
    (void)txn;
  }

  virtual string _debug_op_label() { return "InitOp"; }

  virtual typename V:: Shape shape() {
    return x_.shape();
  }

  virtual typename V:: Shape offset() {
    return x_.offset();
  }

  virtual typename V:: Shape stride() {
    return x_.stride();
  }

  virtual size_t batch_shape() {
    return x_.batch_shape();
  }

  virtual VarData<V> data() {
    return x_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(x_.val().var());
    vs.push_back(x_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  function<void(V*, std::mt19937_64*, shared_ptr<Context>)> init_fn_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
};

template <typename V>
void zeros_init_fn(V* val, std::mt19937_64*, shared_ptr<Context> ctx);

template <typename V>
function<void(V*, std::mt19937_64*, shared_ptr<Context> ctx)> zeros_init_gen();

template <typename V>
void ones_init_fn(V* val, std::mt19937_64*, shared_ptr<Context> ctx);

template <typename V>
function<void(V*, std::mt19937_64*, shared_ptr<Context> ctx)> ones_init_gen();

template <typename V>
function<void(V*, std::mt19937_64*, shared_ptr<Context> ctx)> xavier_init_gen(size_t fanin, size_t fanout);

template <typename V>
function<void(V*, std::mt19937_64*, shared_ptr<Context> ctx)> kaiming_init_gen(size_t fanin, size_t fanout);

template <typename V>
shared_ptr<InitOp<V>> initialize(function<void(V*, std::mt19937_64*, shared_ptr<Context> ctx)> init_fn, shared_ptr<Var<V>> input, shared_ptr<Context> ctx);

template <typename V>
class NoOp : public virtual Op, public virtual Var<V> {
public:
  explicit NoOp(shared_ptr<Var<V>> input)
    : input_(input), x_(input->data()) {}
  NoOp(shared_ptr<Op> input, VarData<V> data)
    : input_(input), x_(data) {}
  virtual ~NoOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    x_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn) {
    (void)txn;
  }

  virtual void _backward(TxnId txn) {
    (void)txn;
  }

  virtual string _debug_op_label() { return "NoOp"; }

  virtual typename V:: Shape shape() {
    return x_.shape();
  }

  virtual typename V:: Shape offset() {
    return x_.offset();
  }

  virtual typename V:: Shape stride() {
    return x_.stride();
  }

  virtual size_t batch_shape() {
    return x_.batch_shape();
  }

  virtual VarData<V> data() {
    return x_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(x_.val().var());
    vs.push_back(x_.grad().var());
    return vs;
  }

private:
  //shared_ptr<Var<V>> input_;
  shared_ptr<Op> input_;
  VarData<V> x_;
};

class SumJoin {};
class FlattenNCHWJoin {};

template <typename V, typename Join>
class JoinOp : public virtual Op, public virtual Var<V> {
public:
  static shared_ptr<Var<V>> Make(
      Join join,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    return make_shared<JoinOp<V, Join>>(join, false, inputs, ctx);
  }

  static vector<shared_ptr<Var<V>>> MakeParallel(
      Join join,
      vector<vector<shared_ptr<Var<V>>>> inputs,
      shared_ptr<Context> ctx)
  {
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      vector<shared_ptr<Var<V>>> dev_inputs;
      for (size_t j = 0; j < inputs.size(); ++j) {
        dev_inputs.push_back(inputs.at(j).at(rank));
      }
      outputs.push_back(make_shared<JoinOp<V, Join>>(join, false, dev_inputs, dev_ctx));
    }
    return outputs;
  }

  JoinOp(Join join, bool in_place, vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx)
    : ctx_(ctx), join_(join), in_place_(in_place), inputs_(inputs), xs_()
  {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
    for (size_t idx = 0; idx < inputs_.size(); ++idx) {
      xs_.push_back(inputs_.at(idx)->data());
    }
    if (in_place_) {
      y_ = inputs_.at(0)->data();
    } else {
      y_ = VarData<V>(inputs_.at(0)->shape(), inputs_.at(0)->batch_shape());
    }
  }

  virtual ~JoinOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      for (size_t idx = 0; idx < inputs_.size(); ++idx) {
        inputs_[idx]->_push(epoch, apply);
      }
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      for (size_t idx = inputs_.size() - 1UL; idx < inputs_.size(); --idx) {
        inputs_[idx]->_pop(epoch, apply);
      }
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  Join join_;
  bool in_place_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  vector<shared_ptr<Var<V>>> inputs_;
  vector<VarData<V>> xs_;
  VarData<V> y_;
};

template <typename V>
shared_ptr<Var<V>> sum(vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx) {
  return make_shared<JoinOp<V, SumJoin>>(SumJoin(), false, inputs, ctx);
}

template <typename V>
shared_ptr<Var<V>> sum_in_place_0(vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx) {
  return make_shared<JoinOp<V, SumJoin>>(SumJoin(), true, inputs, ctx);
}

template <typename V>
vector<shared_ptr<Var<V>>> parallel_sum(vector<vector<shared_ptr<Var<V>>>> inputs, shared_ptr<Context> ctx) {
  return JoinOp<V, SumJoin>::MakeParallel(SumJoin(), inputs, ctx);
}

template <typename V, typename W, typename Join>
class BatchJoinOp : public virtual Op, public virtual Var<W> {
public:
  BatchJoinOp(Join join, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), join_(join), input_(input), x_(input->data()), y_(_get_output_shape(input->shape()), _get_output_batch_shape(input->batch_shape()))
  {
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  }

  virtual ~BatchJoinOp() {}

  typename W:: Shape _get_output_shape(typename V:: Shape in_shape);
  size_t _get_output_batch_shape(size_t in_batch_shape);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }
  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _set_grad_sink(TxnId txn);
  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename W:: Shape shape() {
    return y_.shape();
  }

  virtual typename W:: Shape offset() {
    return y_.offset();
  }

  virtual typename W:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<W> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  Join join_;
  cudnnReduceTensorDescriptor_t reduce_desc_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<W> y_;
};

template <typename V, typename W>
shared_ptr<Var<W>> batch_sum(shared_ptr<Var<V>> input, shared_ptr<Context> ctx) {
  return make_shared<BatchJoinOp<V, W, SumJoin>>(SumJoin(), input, ctx);
}

template <typename V, typename W, typename Join>
class SpatialJoinOp : public virtual Op, public virtual Var<W> {
public:
  static shared_ptr<Var<W>> Make(
      Join join,
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    return make_shared<SpatialJoinOp<V, W, Join>>(join, input, ctx);
  }

  static vector<shared_ptr<Var<W>>> MakeParallel(
      Join join,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    vector<shared_ptr<Var<W>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      outputs.push_back(make_shared<SpatialJoinOp<V, W, Join>>(join, inputs.at(rank), dev_ctx));
    }
    return outputs;
  }

  SpatialJoinOp(Join join, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), join_(join), input_(input), x_(input->data())
  {
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
    y_ = VarData<W>(_get_output_shape(input->shape()), input->batch_shape());
  }

  virtual ~SpatialJoinOp() {}

  typename W:: Shape _get_output_shape(typename V:: Shape in_shape);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }
  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  //virtual void _set_grad_sink(TxnId txn);
  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename W:: Shape shape() {
    return y_.shape();
  }

  virtual typename W:: Shape offset() {
    return y_.offset();
  }

  virtual typename W:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<W> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  Join join_;
  cudnnReduceTensorDescriptor_t reduce_desc_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<W> y_;
};

template <typename V, typename W>
shared_ptr<Var<W>> flatten_nchw(shared_ptr<Var<V>> input, shared_ptr<Context> ctx) {
  //return make_shared<SpatialJoinOp<V, W, FlattenNCHWJoin>>(FlattenNCHWJoin(), input, ctx);
  return SpatialJoinOp<V, W, FlattenNCHWJoin>::Make(FlattenNCHWJoin(), input, ctx);
}

template <typename V, typename W>
vector<shared_ptr<Var<W>>> parallel_flatten_nchw(vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx) {
  return SpatialJoinOp<V, W, FlattenNCHWJoin>::MakeParallel(FlattenNCHWJoin(), inputs, ctx);
}

template <typename V, typename W>
shared_ptr<Var<W>> spatial_sum(shared_ptr<Var<V>> input, shared_ptr<Context> ctx) {
  return make_shared<SpatialJoinOp<V, W, SumJoin>>(SumJoin(), input, ctx);
}

/*template <typename Switch, typename V>
class BranchOp : public virtual Op, public virtual Var<V> {
public:
  BranchOp(shared_ptr<Var<Switch>> switch_var, shared_ptr<Var<V>> off_branch, shared_ptr<Var<V>> on_branch, function<V(TxnId, NodeId)> alloc_fn);
  virtual ~BranchOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      switch_var_->_push(epoch, apply);
      off_branch_->_push(epoch, apply);
      on_branch_->_push(epoch, apply);
      apply(this);
    }
  }
  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      on_branch_->_pop(epoch, apply);
      off_branch_->_pop(epoch, apply);
      switch_var_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  shared_ptr<Var<Switch>> switch_var_;
  shared_ptr<Var<V>> off_branch_;
  shared_ptr<Var<V>> on_branch_;
  VarData<Switch> test_;
  VarData<V> x_off_;
  VarData<V> x_on_;
  VarData<V> y_;
};*/

template <typename V, typename W>
class CastOp : public virtual Op, public virtual Var<W> {
public:
  static shared_ptr<Var<W>> Make(
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    return make_shared<CastOp<V, W>>(input, ctx);
  }

  CastOp(shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), input_(input), x_(input->data()), y_(input->shape(), input->batch_shape()) {}
  virtual ~CastOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual string _debug_op_label() { return "CastOp"; }

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<W> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<W> y_;
};

template <typename V, typename Shape>
class TransposeNHWCToNCHWOp : public virtual Op, public virtual Var<V> {
public:
  static shared_ptr<Var<V>> Make(
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    return make_shared<TransposeNHWCToNCHWOp<V, Shape>>(input, ctx);
  }

  TransposeNHWCToNCHWOp(shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), input_(input), x_(input->data()), y_(_get_output_shape(input->shape()), input->batch_shape())
  {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  }
  virtual ~TransposeNHWCToNCHWOp() {}

  Shape _get_output_shape(Shape in_shape) const;

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual string _debug_op_label() { return "TransposeNHWCToNCHWOp"; }

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  /*Shape pre_dim_;
  Shape post_dim_;*/
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<V> y_;
};

template <typename T, typename V>
class ScaleOp : public virtual Op, public virtual Var<V> {
public:
  static shared_ptr<Var<V>> Make(
      T scale,
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    return make_shared<ScaleOp<T, V>>(scale, input, ctx);
  }

  ScaleOp(T scale, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), scale_(scale), input_(input), x_(input->data()), y_(input->shape(), input->batch_shape()) {}
  virtual ~ScaleOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  T scale_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<V> y_;
};

class Pad2DShape {
public:
  size_t pad_h_left;
  size_t pad_h_right;
  size_t pad_w_left;
  size_t pad_w_right;
};

template <typename V, typename A, typename B>
class CublasLinearOp : public virtual Op, public virtual Var<V> {
public:
  CublasLinearOp(size_t filters, shared_ptr<Var<V>> input, shared_ptr<Var<A>> kernel, shared_ptr<Context> ctx)
    : ctx_(ctx), filters_(filters), input_(input), kernel_(kernel), x_(input->data()), a_(kernel->data()), y_(_get_output_shape(input->shape()), input->batch_shape())
  {
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&bias_grad_reduce_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  }

  CublasLinearOp(size_t filters, shared_ptr<Var<V>> input, shared_ptr<Var<A>> kernel, shared_ptr<Var<B>> bias, shared_ptr<Context> ctx)
    : ctx_(ctx), filters_(filters), input_(input), kernel_(kernel), bias_(bias), x_(input->data()), a_(kernel->data()), b_(bias->data()), y_(_get_output_shape(input->shape()), input->batch_shape())
  {
    CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&bias_grad_reduce_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  }

  virtual ~CublasLinearOp() {}

  typename V:: Shape _get_output_shape(typename V:: Shape in_shape);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      kernel_->_push(epoch, apply);
      if (static_cast<bool>(bias_)) {
        bias_.value()->_push(epoch, apply);
      }
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      if (static_cast<bool>(bias_)) {
        bias_.value()->_pop(epoch, apply);
      }
      kernel_->_pop(epoch, apply);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  //virtual void _set_grad_sink(TxnId txn);
  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  size_t filters_;
  cudnnReduceTensorDescriptor_t bias_grad_reduce_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t y_desc_;
  shared_ptr<Var<V>> input_;
  shared_ptr<Var<A>> kernel_;
  optional<shared_ptr<Var<B>>> bias_;
  VarData<V> x_;
  VarData<A> a_;
  optional<VarData<B>> b_;
  VarData<V> y_;
};

enum Activation {
  kReluAct,
};

class Conv2DShape {
public:
  size_t filters;
  array<size_t, 2> axes;
  array<size_t, 2> kernel;
  array<size_t, 2> dilate;
  array<size_t, 2> stride;
  array<size_t, 2> pad;
  bool halo_padded_input;
  //optional<Activation> fused_activation;
  bool enable_hmma;
};

class CudnnConv2DSize {
public:
  CudnnConv2DSize(Conv2DShape shape, /*size_t halo_pad,*/ Shape3D x_dim, size_t batch_size, GPUContext* ctx);

  size_t batch_size;
  Shape3D in_shape;
  Shape3D out_shape;
  Shape3D out_offset;
  Shape3D out_stride;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnFilterDescriptor_t kernel_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnTensorDescriptor_t x_desc;
  cudnnTensorDescriptor_t y_desc;

  size_t max_workspace_size;
  cudnnConvolutionFwdAlgoPerf_t fwd_algo;
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_kernel_algo;
  cudnnConvolutionBwdDataAlgoPerf_t bwd_input_algo;
};

template <typename V, typename A, typename B>
class CudnnConv2DOp : public virtual Op, public virtual Var<V> {
public:
  static shared_ptr<Var<V>> Make(
      Conv2DShape shape,
      shared_ptr<Var<V>> input,
      shared_ptr<Var<A>> kernel,
      shared_ptr<Var<B>> bias,
      shared_ptr<Context> ctx)
  {
    return make_shared<CudnnConv2DOp<V, A, B>>(shape, input, kernel, bias, ctx);
  }

  CudnnConv2DOp(Conv2DShape shape, shared_ptr<Var<V>> input, shared_ptr<Var<A>> kernel, shared_ptr<Context> ctx)
    : ctx_(ctx), shape_(shape), conv_sizes_(), x_var_(input), kernel_var_(kernel), x_(input->data()), kernel_(kernel->data()), workspace_size_(0)
  {
    y_ = VarData<V>(
        _get_output_shape(input->shape(), input->batch_shape()),
        _get_output_offset(input->shape(), input->batch_shape()),
        _get_output_stride(input->shape(), input->batch_shape()),
        input->batch_shape());
  }

  CudnnConv2DOp(Conv2DShape shape, shared_ptr<Var<V>> input, shared_ptr<Var<A>> kernel, shared_ptr<Var<B>> bias, shared_ptr<Context> ctx)
    : ctx_(ctx), shape_(shape), conv_sizes_(), x_var_(input), kernel_var_(kernel), bias_var_(bias), x_(input->data()), kernel_(kernel->data()), bias_(bias->data()), workspace_size_(0)
  {
    y_ = VarData<V>(
        _get_output_shape(input->shape(), input->batch_shape()),
        _get_output_offset(input->shape(), input->batch_shape()),
        _get_output_stride(input->shape(), input->batch_shape()),
        input->batch_shape());
  }

  virtual ~CudnnConv2DOp() {}

  Shape3D _get_output_shape(Shape3D in_shape, size_t batch_size);
  Shape3D _get_output_offset(Shape3D in_shape, size_t batch_size);
  Shape3D _get_output_stride(Shape3D in_shape, size_t batch_size);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      x_var_->_push(epoch, apply);
      kernel_var_->_push(epoch, apply);
      if (static_cast<bool>(bias_var_)) {
        bias_var_.value()->_push(epoch, apply);
      }
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      if (static_cast<bool>(bias_var_)) {
        bias_var_.value()->_pop(epoch, apply);
      }
      kernel_var_->_pop(epoch, apply);
      x_var_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  Conv2DShape shape_;
  std::unordered_map<size_t, CudnnConv2DSize> conv_sizes_;
  shared_ptr<Var<V>> x_var_;
  shared_ptr<Var<A>> kernel_var_;
  optional<shared_ptr<Var<B>>> bias_var_;
  VarData<V> x_;
  VarData<A> kernel_;
  optional<VarData<B>> bias_;
  VarData<V> y_;
  size_t workspace_size_;
};

template <typename V>
class BroadcastOp : public virtual Op {
public:
  static vector<shared_ptr<Var<V>>> MakeParallel(
      size_t root_rank,
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    auto op = make_shared<BroadcastOp<V>>(root_rank, input, ctx);
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      outputs.push_back(make_shared<NoOp<V>>(op, op->ys_.at(rank).clone()));
    }
    return outputs;
  }

  BroadcastOp(size_t root_rank, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), root_rank_(root_rank), input_(input), x_(input->data()), ys_()
  {
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      ys_.push_back(VarData<V>(input_->shape(), input_->batch_shape()));
    }
  }

  virtual ~BroadcastOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

private:
  shared_ptr<Context> ctx_;
  size_t root_rank_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  vector<VarData<V>> ys_;
};

template <typename V>
class ReduceOp : public virtual Op {
public:
  static shared_ptr<Var<V>> MakeParallel(
      size_t root_rank,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    auto op = make_shared<ReduceOp<V>>(root_rank, inputs, ctx);
    return make_shared<NoOp<V>>(op, op->y_.clone());
  }

  ReduceOp(size_t root_rank, vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx)
    : ctx_(ctx), root_rank_(root_rank), inputs_(inputs), xs_()
  {
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      xs_.push_back(inputs_.at(rank)->data());
    }
    y_ = VarData<V>(inputs_.at(root_rank_)->shape(), inputs_.at(root_rank_)->batch_shape());
  }

  virtual ~ReduceOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        inputs_.at(rank)->_push(epoch, apply);
      }
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
      for (size_t rank = mgpu_ctx->num_ranks() - 1UL; rank < mgpu_ctx->num_ranks(); --rank) {
        inputs_.at(rank)->_pop(epoch, apply);
      }
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

private:
  shared_ptr<Context> ctx_;
  size_t root_rank_;
  vector<shared_ptr<Var<V>>> inputs_;
  vector<VarData<V>> xs_;
  VarData<V> y_;
};

template <typename V>
class SelectOp : public virtual Op {
public:
  static shared_ptr<Var<V>> MakeParallel(
      size_t root_rank,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    auto op = make_shared<SelectOp<V>>(root_rank, inputs, ctx);
    return make_shared<NoOp<V>>(op, op->y_.clone());
  }

  SelectOp(size_t root_rank, vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx)
    : ctx_(ctx), root_rank_(root_rank), inputs_(inputs), xs_()
  {
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      xs_.push_back(inputs_.at(rank)->data());
    }
    y_ = VarData<V>(inputs_.at(root_rank_)->shape(), inputs_.at(root_rank_)->batch_shape());
  }

  virtual ~SelectOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        inputs_.at(rank)->_push(epoch, apply);
      }
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
      for (size_t rank = mgpu_ctx->num_ranks() - 1UL; rank < mgpu_ctx->num_ranks(); --rank) {
        inputs_.at(rank)->_pop(epoch, apply);
      }
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

private:
  shared_ptr<Context> ctx_;
  size_t root_rank_;
  vector<shared_ptr<Var<V>>> inputs_;
  vector<VarData<V>> xs_;
  VarData<V> y_;
};

template <typename V>
class AllreduceOp : public virtual Op {
public:
  static vector<shared_ptr<Var<V>>> MakeParallel(
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    auto op = make_shared<AllreduceOp<V>>(inputs, ctx);
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      outputs.push_back(make_shared<NoOp<V>>(op, op->ys_.at(rank).clone()));
    }
    return outputs;
  }

  AllreduceOp(vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx)
    : ctx_(ctx), inputs_(inputs), xs_(), ys_()
  {
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      xs_.push_back(inputs_.at(rank)->data());
      ys_.push_back(VarData<V>(inputs_.at(rank)->shape(), inputs_.at(rank)->batch_shape()));
    }
  }

  virtual ~AllreduceOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        inputs_.at(rank)->_push(epoch, apply);
      }
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
      for (size_t rank = mgpu_ctx->num_ranks() - 1UL; rank < mgpu_ctx->num_ranks(); --rank) {
        inputs_.at(rank)->_pop(epoch, apply);
      }
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

private:
  shared_ptr<Context> ctx_;
  vector<shared_ptr<Var<V>>> inputs_;
  vector<VarData<V>> xs_;
  vector<VarData<V>> ys_;
};

template <typename V>
class SpatialScatter2DOp : public virtual Op, public virtual Var<V> {
public:
  static shared_ptr<Var<V>> Make(
      size_t rank,
      size_t num_ranks,
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    return make_shared<SpatialScatter2DOp<V>>(rank, num_ranks, input, ctx);
  }

  static vector<shared_ptr<Var<V>>> MakeParallel(
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      outputs.push_back(make_shared<SpatialScatter2DOp<V>>(rank, mgpu_ctx->num_ranks(), inputs.at(rank), dev_ctx));
    }
    return outputs;
  }

  SpatialScatter2DOp(size_t rank, size_t num_ranks, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), rank_(rank), num_ranks_(num_ranks), input_(input), x_(input->data()), y_(_get_output_shape(input_->shape()), input_->batch_shape())
  {}

  virtual ~SpatialScatter2DOp() {}

  typename V:: Shape _get_output_shape(typename V:: Shape in_shape);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  size_t rank_;
  size_t num_ranks_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<V> y_;
};

template <typename V>
class SpatialGather2DOp : public virtual Op, public virtual Var<V> {
public:
  static shared_ptr<Var<V>> Make(
      size_t rank,
      size_t num_ranks,
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    return make_shared<SpatialGather2DOp<V>>(rank, num_ranks, input, ctx);
  }

  static vector<shared_ptr<Var<V>>> MakeParallel(
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      outputs.push_back(make_shared<SpatialGather2DOp<V>>(rank, mgpu_ctx->num_ranks(), inputs.at(rank), dev_ctx));
    }
    return outputs;
  }

  SpatialGather2DOp(size_t rank, size_t num_ranks, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), rank_(rank), num_ranks_(num_ranks), input_(input), x_(input->data()), y_(_get_output_shape(input->shape()), input->batch_shape())
  {}

  virtual ~SpatialGather2DOp() {}

  typename V:: Shape _get_output_shape(typename V:: Shape in_shape);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  size_t rank_;
  size_t num_ranks_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<V> y_;
};

template <typename V>
class HaloPad2DOp : public virtual Op, public virtual Var<V> {
public:
  static shared_ptr<Var<V>> Make(
      size_t halo_pad,
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    return make_shared<HaloPad2DOp<V>>(halo_pad, input, ctx);
  }

  static vector<shared_ptr<Var<V>>> MakeParallel(
      size_t halo_pad,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      outputs.push_back(make_shared<HaloPad2DOp<V>>(halo_pad, inputs.at(rank), dev_ctx));
    }
    return outputs;
  }

  HaloPad2DOp(size_t halo_pad, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), halo_pad_(halo_pad), input_(input), x_(input->data())
  {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
    y_ = VarData<V>(
        input_->shape(),
        _get_output_offset(input_->shape()),
        _get_output_stride(input_->shape()),
        input_->batch_shape());
  }

  virtual ~HaloPad2DOp() {}

  typename V:: Shape _get_output_offset(typename V:: Shape in_shape);
  typename V:: Shape _get_output_stride(typename V:: Shape in_shape);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  size_t halo_pad_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<V> y_;
};

template <typename V>
class HaloUnpad2DOp : public virtual Op, public virtual Var<V> {
public:
  static shared_ptr<Var<V>> Make(
      size_t halo_pad,
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    return make_shared<HaloUnpad2DOp<V>>(halo_pad, input, ctx);
  }

  static vector<shared_ptr<Var<V>>> MakeParallel(
      size_t halo_pad,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      outputs.push_back(make_shared<HaloUnpad2DOp<V>>(halo_pad, inputs.at(rank), dev_ctx));
    }
    return outputs;
  }

  HaloUnpad2DOp(size_t halo_pad, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), halo_pad_(halo_pad), input_(input), x_(input->data()), y_(input->shape(), input->batch_shape())
  {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  }

  virtual ~HaloUnpad2DOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  size_t halo_pad_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<V> y_;
};

template <typename V>
class HaloRepad2DOp : public virtual Op, public virtual Var<V> {
public:
  static shared_ptr<Var<V>> Make(
      size_t src_halo_pad,
      size_t dst_halo_pad,
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    return make_shared<HaloRepad2DOp<V>>(src_halo_pad, dst_halo_pad, input, ctx);
  }

  static vector<shared_ptr<Var<V>>> MakeParallel(
      size_t src_halo_pad,
      size_t dst_halo_pad,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      outputs.push_back(make_shared<HaloRepad2DOp<V>>(src_halo_pad, dst_halo_pad, inputs.at(rank), dev_ctx));
    }
    return outputs;
  }

  HaloRepad2DOp(size_t src_halo_pad, size_t dst_halo_pad, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), src_halo_pad_(src_halo_pad), dst_halo_pad_(dst_halo_pad), input_(input), x_(input->data()), y_(input->shape(), input->batch_shape())
  {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  }

  virtual ~HaloRepad2DOp() {}

  typename V:: Shape _get_output_offset(typename V:: Shape in_shape);
  typename V:: Shape _get_output_stride(typename V:: Shape in_shape);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  size_t src_halo_pad_;
  size_t dst_halo_pad_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<V> y_;
};

template <typename V>
class HaloExch2DOp : public virtual Op {
public:
  static vector<shared_ptr<Var<V>>> MakeParallel(
      size_t halo_pad,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    auto op = make_shared<HaloExch2DOp<V>>(halo_pad, inputs, ctx);
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      outputs.push_back(make_shared<NoOp<V>>(op, op->xs_.at(rank).clone()));
    }
    return outputs;
  }

  HaloExch2DOp(size_t halo_pad, vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx)
    : ctx_(ctx), halo_pad_(halo_pad), inputs_(inputs)
  {
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
    assert(inputs.size() == mgpu_ctx->num_ranks());
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
      xs_.push_back(inputs.at(idx)->data());
    }
  }

  virtual ~HaloExch2DOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        inputs_.at(rank)->_push(epoch, apply);
      }
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx_.get());
      for (size_t rank = mgpu_ctx->num_ranks() - 1UL; rank < mgpu_ctx->num_ranks(); --rank) {
        inputs_.at(rank)->_pop(epoch, apply);
      }
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

private:
  shared_ptr<Context> ctx_;
  shared_ptr<Spatial2DComm> comm_;
  size_t halo_pad_;
  //cudnnTensorDescriptor_t x_desc_;
  //cudnnTensorDescriptor_t y_desc_;
  vector<shared_ptr<Var<V>>> inputs_;
  vector<VarData<V>> xs_;
  //vector<VarData<V>> ys_;
};

class CudnnDeconv2DSize {
public:
  CudnnDeconv2DSize(Conv2DShape shape, array<size_t, 3> x_shape, size_t batch_size, GPUContext* ctx);

  size_t batch_size;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnFilterDescriptor_t kernel_desc;
  cudnnTensorDescriptor_t bias_desc;
  cudnnTensorDescriptor_t x_desc;
  cudnnTensorDescriptor_t y_desc;

  size_t max_workspace_size;
  cudnnConvolutionBwdDataAlgoPerf_t bwd_input_algo;
  cudnnConvolutionBwdFilterAlgoPerf_t bwd_kernel_algo;
  cudnnConvolutionFwdAlgoPerf_t fwd_algo;
};

template <typename V, typename A, typename B>
class CudnnDeconv2DOp : public virtual Op, public virtual Var<V> {
public:
  CudnnDeconv2DOp(Conv2DShape shape, shared_ptr<Var<V>> input, shared_ptr<Var<A>> kernel, optional<shared_ptr<Var<B>>> bias, function<V(TxnId, NodeId)> alloc_fn, shared_ptr<Context> ctx);
  virtual ~CudnnDeconv2DOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      x_var_->_push(epoch, apply);
      kernel_var_->_push(epoch, apply);
      if (static_cast<bool>(bias_var_)) {
        bias_var_.value()->_push(epoch, apply);
      }
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      if (static_cast<bool>(bias_var_)) {
        bias_var_.value()->_pop(epoch, apply);
      }
      kernel_var_->_pop(epoch, apply);
      x_var_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  Conv2DShape shape_;
  std::unordered_map<size_t, CudnnDeconv2DSize> deconv_sizes_;
  shared_ptr<Var<V>> x_var_;
  shared_ptr<Var<A>> kernel_var_;
  optional<shared_ptr<Var<B>>> bias_var_;
  VarData<V> x_;
  VarData<A> kernel_;
  optional<VarData<B>> bias_;
  VarData<V> y_;
  size_t workspace_size_;
  // TODO: workspace should be shared in context.
  //GPUMemory<uint8_t> workspace_;
};

template <typename V>
class CudnnActivationOp : public virtual Op, public virtual Var<V> {
public:
  static vector<shared_ptr<Var<V>>> MakeParallel(
      Activation activation,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      outputs.push_back(make_shared<CudnnActivationOp<V>>(activation, false, inputs.at(rank), dev_ctx));
    }
    return outputs;
  }

  CudnnActivationOp(Activation activation, bool in_place, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), activation_(activation), in_place_(in_place), input_(input), x_(input->data())
  {
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    //CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
    //CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_desc_));
    if (in_place_) {
      y_ = input->data();
    } else {
      y_ = VarData<V>(input->shape(), input->batch_shape());
    }
  }
  virtual ~CudnnActivationOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  Activation activation_;
  bool in_place_;
  shared_ptr<Var<V>> input_;
  cudnnActivationDescriptor_t act_desc_;
  cudnnTensorDescriptor_t x_desc_;
  //cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t y_desc_;
  //cudnnTensorDescriptor_t dy_desc_;
  VarData<V> x_;
  VarData<V> y_;
};

template <typename V>
shared_ptr<Var<V>> relu(shared_ptr<Var<V>> input, shared_ptr<Context> ctx);

template <typename V>
shared_ptr<Var<V>> relu_in_place(shared_ptr<Var<V>> input, shared_ptr<Context> ctx);

template <typename V>
vector<shared_ptr<Var<V>>> parallel_relu(vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx) {
  return CudnnActivationOp<V>::MakeParallel(kReluAct, inputs, ctx);
}

enum Pooling {
  kAveragePool,
  kMaxPool,
};

class Pool2DShape {
public:
  Pooling pooling;
  array<size_t, 2> axes;
  array<size_t, 2> window;
  array<size_t, 2> stride;
  array<size_t, 2> pad;
  bool halo_padded_input;
};

template <typename V>
class CudnnPool2DOp : public virtual Op, public virtual Var<V> {
public:
  static shared_ptr<Var<V>> Make(
      Pool2DShape shape,
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx)
  {
    return make_shared<CudnnPool2DOp<V>>(shape, input, ctx);
  }

  static vector<shared_ptr<Var<V>>> MakeParallel(
      Pool2DShape shape,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    vector<shared_ptr<Var<V>>> outputs;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      outputs.push_back(make_shared<CudnnPool2DOp<V>>(shape, inputs.at(rank), dev_ctx));
    }
    return outputs;
  }

  CudnnPool2DOp(Pool2DShape shape, shared_ptr<Var<V>> input, shared_ptr<Context> ctx)
    : ctx_(ctx), shape_(shape), input_(input), x_(input->data())
  {
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    //CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
    //CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_desc_));
    y_ = VarData<V>(_get_output_shape(input->shape()), input->batch_shape());
  }
  virtual ~CudnnPool2DOp() {}

  Shape3D _get_output_shape(Shape3D in_shape);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  //virtual void _set_grad_sink(TxnId txn);
  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  Pool2DShape shape_;
  cudnnPoolingDescriptor_t pool_desc_;
  cudnnTensorDescriptor_t x_desc_;
  //cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t y_desc_;
  //cudnnTensorDescriptor_t dy_desc_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<V> y_;
};

template <typename V>
class CudnnDownsample2DOp : public virtual Op, public virtual Var<V> {
public:
  CudnnDownsample2DOp(size_t downsample_factor, shared_ptr<Var<V>> input, function<V(TxnId, NodeId)> alloc_fn, shared_ptr<Context> ctx)
    : ctx_(ctx), downsample_factor_(downsample_factor), input_(input), x_(input->data()), y_(alloc_fn)
  {
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    //CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
    //CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_desc_));
  }
  virtual ~CudnnDownsample2DOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  size_t downsample_factor_;
  cudnnPoolingDescriptor_t pool_desc_;
  cudnnTensorDescriptor_t x_desc_;
  //cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t y_desc_;
  //cudnnTensorDescriptor_t dy_desc_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<V> y_;
};

template <typename V>
class CudnnUpsample2DOp : public virtual Op, public virtual Var<V> {
public:
  CudnnUpsample2DOp(size_t upsample_factor, shared_ptr<Var<V>> input, function<V(TxnId, NodeId)> alloc_fn, shared_ptr<Context> ctx)
    : ctx_(ctx), upsample_factor_(upsample_factor), input_(input), x_(input->data()), y_(alloc_fn)
  {
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    //CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
    //CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_desc_));
  }
  virtual ~CudnnUpsample2DOp() {}

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  size_t upsample_factor_;
  cudnnPoolingDescriptor_t pool_desc_;
  cudnnTensorDescriptor_t x_desc_;
  //cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t y_desc_;
  //cudnnTensorDescriptor_t dy_desc_;
  shared_ptr<Var<V>> input_;
  VarData<V> x_;
  VarData<V> y_;
};

template <typename V, typename B>
class BatchNormOutput {
public:
  shared_ptr<Var<V>> op;
  shared_ptr<Var<B>> batch_mean;
  shared_ptr<Var<B>> batch_variance;
  shared_ptr<Var<B>> fixed_mean;
  shared_ptr<Var<B>> fixed_variance;
  shared_ptr<Var<B>> scale;
  shared_ptr<Var<B>> shift;
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;
};

template <typename V, typename B>
class ParallelBatchNormOutput {
public:
  vector<shared_ptr<Var<V>>> ops;
  vector<VarIdSet> params;
  vector<VarIdSet> grads;
  vector<VarIdSet> batch_stats;
  vector<VarIdSet> fixed_stats;
};

template <typename Switch, typename V, typename B>
class CudnnBatchNorm2DOp : public virtual Op, public virtual Var<V> {
public:
  static BatchNormOutput<V, B> Make(
      size_t channels,
      shared_ptr<Switch> switch_var,
      shared_ptr<Var<V>> input,
      shared_ptr<Context> ctx);

  static ParallelBatchNormOutput<V, B> MakeParallel(
      size_t channels,
      shared_ptr<Switch> switch_var,
      vector<shared_ptr<Var<V>>> inputs,
      shared_ptr<Context> ctx)
  {
    ParallelBatchNormOutput<V, B> output;
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
    for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
      shared_ptr<Context> dev_ctx = mgpu_ctx->device_context(rank);
      auto dev_output = CudnnBatchNorm2DOp<Switch, V, B>::Make(channels, switch_var, inputs.at(rank), dev_ctx);
      output.ops.push_back(dev_output.op);
      output.params.push_back(dev_output.params);
      output.grads.push_back(dev_output.grads);
      output.batch_stats.push_back(dev_output.batch_stats);
      output.fixed_stats.push_back(dev_output.fixed_stats);
    }
    return output;
  }

  CudnnBatchNorm2DOp(
      shared_ptr<Switch> switch_var,
      shared_ptr<Var<V>> input,
      shared_ptr<Var<B>> batch_mean_var,
      shared_ptr<Var<B>> batch_variance_var,
      shared_ptr<Var<B>> fixed_mean_var,
      shared_ptr<Var<B>> fixed_variance_var,
      shared_ptr<Var<B>> scale_var,
      shared_ptr<Var<B>> shift_var,
      shared_ptr<Context> ctx)
    : ctx_(ctx),
      fixed_scale_shift_(false),
      switch_var_(switch_var),
      input_(input),
      batch_mean_var_(batch_mean_var),
      batch_variance_var_(batch_variance_var),
      fixed_mean_var_(fixed_mean_var),
      fixed_variance_var_(fixed_variance_var),
      scale_var_(scale_var),
      shift_var_(shift_var),
      x_(input->data()),
      y_(input->shape(), input->batch_shape()),
      batch_mean_(batch_mean_var->data()),
      batch_variance_(batch_variance_var->data()),
      fixed_mean_(fixed_mean_var->data()),
      fixed_variance_(fixed_variance_var->data()),
      scale_(scale_var->data()),
      shift_(shift_var->data()),
      tmp_mean_(_get_stat_shape(input->shape())),
      tmp_variance_(_get_stat_shape(input->shape()))
  {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    //CUDNN_CHECK(cudnnCreateTensorDescriptor(&dx_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
    //CUDNN_CHECK(cudnnCreateTensorDescriptor(&dy_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&stat_desc_));
  }

  virtual ~CudnnBatchNorm2DOp() {}

  typename B:: Shape _get_stat_shape(typename V:: Shape in_shape);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      //if (!static_cast<bool>(*switch_var_)) {
        batch_mean_var_->_push(epoch, apply);
        batch_variance_var_->_push(epoch, apply);
      //} else {
        fixed_mean_var_->_push(epoch, apply);
        fixed_variance_var_->_push(epoch, apply);
      //}
      if (!fixed_scale_shift_) {
        scale_var_->_push(epoch, apply);
        shift_var_->_push(epoch, apply);
      }
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      if (!fixed_scale_shift_) {
        shift_var_->_pop(epoch, apply);
        scale_var_->_pop(epoch, apply);
      }
      //if (static_cast<bool>(*switch_var_)) {
        fixed_variance_var_->_pop(epoch, apply);
        fixed_mean_var_->_pop(epoch, apply);
      //} else {
        batch_variance_var_->_pop(epoch, apply);
        batch_mean_var_->_pop(epoch, apply);
      //}
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    y_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename V:: Shape shape() {
    return y_.shape();
  }

  virtual typename V:: Shape offset() {
    return y_.offset();
  }

  virtual typename V:: Shape stride() {
    return y_.stride();
  }

  virtual size_t batch_shape() {
    return y_.batch_shape();
  }

  virtual VarData<V> data() {
    return y_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(y_.val().var());
    vs.push_back(y_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  bool fixed_scale_shift_;
  cudnnTensorDescriptor_t x_desc_;
  //cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t y_desc_;
  //cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t stat_desc_;
  shared_ptr<Switch> switch_var_;
  shared_ptr<Var<V>> input_;
  shared_ptr<Var<B>> batch_mean_var_;
  shared_ptr<Var<B>> batch_variance_var_;
  shared_ptr<Var<B>> fixed_mean_var_;
  shared_ptr<Var<B>> fixed_variance_var_;
  shared_ptr<Var<B>> scale_var_;
  shared_ptr<Var<B>> shift_var_;
  VarData<V> x_;
  VarData<V> y_;
  VarData<B> batch_mean_;
  VarData<B> batch_variance_;
  VarData<B> fixed_mean_;
  VarData<B> fixed_variance_;
  VarData<B> scale_;
  VarData<B> shift_;
  VarData<B> tmp_mean_;
  VarData<B> tmp_variance_;
};

template <typename Switch, typename V, typename B>
BatchNormOutput<V, B> batch_norm2d(
    size_t channels,
    shared_ptr<Switch> switch_var,
    shared_ptr<Var<V>> input,
    shared_ptr<Context> ctx);

template <typename V, typename Label, typename Loss>
class CudnnSoftmaxNLLLossOp : public virtual Op, public virtual Var<Loss> {
public:
  CudnnSoftmaxNLLLossOp(shared_ptr<Var<V>> input, shared_ptr<Var<Label>> label, shared_ptr<Context> ctx)
    : ctx_(ctx), input_(input), label_var_(label), x_(input->data()), y_(input->shape(), input->batch_shape()), label_(label->data()), loss_(_get_output_shape(input->shape()), input->batch_shape())
  {
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
  }

  virtual ~CudnnSoftmaxNLLLossOp() {}

  typename Loss:: Shape _get_output_shape(typename V:: Shape in_shape);

  virtual void _push(EpochId epoch, function<void(Op* op)> apply) {
    if (1 == stack_.push(epoch)) {
      input_->_push(epoch, apply);
      label_var_->_push(epoch, apply);
      apply(this);
    }
  }

  virtual void _pop(EpochId epoch, function<void(Op* op)> apply) {
    if (stack_.degree(epoch) == stack_.pop(epoch)) {
      apply(this);
      label_var_->_pop(epoch, apply);
      input_->_pop(epoch, apply);
    }
  }

  virtual void _persist(TxnId txn, VarIdSet& vars) {
    loss_.rollover_all(txn, vars);
  }

  virtual void _forward(TxnId txn);
  virtual void _backward(TxnId txn);

  virtual typename Loss:: Shape shape() {
    return loss_.shape();
  }

  virtual typename Loss:: Shape offset() {
    return loss_.offset();
  }

  virtual typename Loss:: Shape stride() {
    return loss_.stride();
  }

  virtual size_t batch_shape() {
    return loss_.batch_shape();
  }

  virtual VarData<Loss> data() {
    return loss_.clone();
  }

  virtual vector<VarId> vars() {
    vector<VarId> vs;
    vs.push_back(loss_.val().var());
    vs.push_back(loss_.grad().var());
    return vs;
  }

private:
  shared_ptr<Context> ctx_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  shared_ptr<Var<V>> input_;
  shared_ptr<Var<Label>> label_var_;
  VarData<V> x_;
  VarData<V> y_;
  VarData<Label> label_;
  VarData<Loss> loss_;
};

template <typename V, typename Label, typename Loss>
shared_ptr<Var<Loss>> softmax_nll_loss(shared_ptr<Var<V>> input, shared_ptr<Var<Label>> label, shared_ptr<Context> ctx) {
  return make_shared<CudnnSoftmaxNLLLossOp<V, Label, Loss>>(input, label, ctx);
}

} // namespace arraydiff

#endif
