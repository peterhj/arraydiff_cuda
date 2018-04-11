#ifndef ARRAYDIFF_OPTIMIZE_HH
#define ARRAYDIFF_OPTIMIZE_HH

#include "arraydiff/context.hh"
#include "arraydiff/op.hh"

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>

namespace arraydiff {

using std::function;
using std::make_pair;
using std::pair;
using std::shared_ptr;

template <typename T>
class Schedule {
public:
  Schedule() {}
  virtual ~Schedule() {}

  virtual T at(size_t index) = 0;
};

template <typename T>
class ConstantSchedule : public virtual Schedule<T> {
public:
  explicit ConstantSchedule(T constant)
    : constant_(constant) {}
  virtual ~ConstantSchedule() {}

  virtual T at(size_t index) {
    (void)index;
    return constant_;
  }

private:
  T constant_;
};

template <typename T>
class PiecewiseStepSchedule : public virtual Schedule<T> {
public:
  explicit PiecewiseStepSchedule(T init)
    : init_(init), pieces_() {}
  PiecewiseStepSchedule(T init, size_t step1, T val1)
    : init_(init), pieces_()
  {
    pieces_.push_back(make_pair(step1, val1));
  }
  PiecewiseStepSchedule(T init, size_t step1, T val1, size_t step2, T val2)
    : init_(init), pieces_()
  {
    pieces_.push_back(make_pair(step1, val1));
    pieces_.push_back(make_pair(step2, val2));
  }
  PiecewiseStepSchedule(T init, size_t step1, T val1, size_t step2, T val2, size_t step3, T val3)
    : init_(init), pieces_()
  {
    pieces_.push_back(make_pair(step1, val1));
    pieces_.push_back(make_pair(step2, val2));
    pieces_.push_back(make_pair(step3, val3));
  }
  PiecewiseStepSchedule(T init, vector<pair<size_t, T>> pieces)
    : init_(init), pieces_(pieces) {}
  virtual ~PiecewiseStepSchedule() {}

  virtual T at(size_t index) {
    if (0UL == pieces_.size()) {
      return init_;
    }
    for (size_t k = 0UL; k < pieces_.size(); ++k) {
      if (index < pieces_[k].first) {
        if (k == 0UL) {
          return init_;
        } else {
          return pieces_[k-1UL].second;
        }
      }
    }
    return pieces_[pieces_.size()-1UL].second;
  }

private:
  T init_;
  vector<pair<size_t, T>> pieces_;
};

template <typename T>
class PiecewiseLinearSchedule : public virtual Schedule<T> {
public:
  explicit PiecewiseLinearSchedule(T init)
    : init_(init), pieces_() {}
  PiecewiseLinearSchedule(T init, size_t step1, T val1)
    : init_(init), pieces_()
  {
    pieces_.push_back(make_pair(step1, val1));
  }
  virtual ~PiecewiseLinearSchedule() {}

  virtual T at(size_t index) {
    if (0UL == pieces_.size()) {
      return init_;
    }
    for (size_t k = 0UL; k < pieces_.size(); ++k) {
      if (index < pieces_[k].first) {
        if (k == 0UL) {
          T alpha = static_cast<T>(index) / static_cast<T>(pieces_[k].first);
          return init_ + alpha * (pieces_[k].second - init_);
        } else {
          T alpha = static_cast<T>(index - pieces_[k-1UL].first) / static_cast<T>(pieces_[k].first - pieces_[k-1UL].first);
          return pieces_[k-1UL].second + alpha * (pieces_[k].second - pieces_[k-1UL].second);
        }
      }
    }
    return pieces_[pieces_.size()-1UL].second;
  }

private:
  T init_;
  vector<pair<size_t, T>> pieces_;
};

class SGDConfig {
public:
  size_t minibatch_size;
  size_t compute_batch_size;
  shared_ptr<Schedule<double>> step_size;
  shared_ptr<Schedule<double>> momentum;
  bool nesterov;
  double l2_reg;
  double batch_norm_rate;
};

template <typename T>
class SGDOptimizer {
public:
  SGDOptimizer(SGDConfig cfg, shared_ptr<Op> sum_loss, VarIdSet params, VarIdSet grads, VarIdSet batch_stats, VarIdSet fixed_stats)
    : cfg_(cfg), sum_loss_(sum_loss), params_(params), grads_(grads), batch_stats_(batch_stats), fixed_stats_(fixed_stats)
  {}
  ~SGDOptimizer() {}

  void reset(shared_ptr<Context> ctx) {
    {
      auto init_txn = TxnId();
      sum_loss_->persist(init_txn, params_);
      sum_loss_->persist(init_txn, fixed_stats_);
      param_dim_ = sum_loss_->store_val(init_txn, params_, 0, NULL);
      stat_dim_ = sum_loss_->store_val(init_txn, fixed_stats_, 0, NULL);
      std::clog << "DEBUG: SGD: param dim: " << param_dim_ << std::endl;
      std::clog << "DEBUG: SGD: stat dim:  " << stat_dim_ << std::endl;
    }
    iter_count_ = 0;
    GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
    param_ = GPUMemory<T>::Zeros(param_dim_, gpu_ctx);
    batch_sum_grad_ = GPUMemory<T>::Zeros(param_dim_, gpu_ctx);
    sum_grad_ = GPUMemory<T>::Zeros(param_dim_, gpu_ctx);
    momentum_grad_ = GPUMemory<T>::Zeros(param_dim_, gpu_ctx);
    batch_stat_ = GPUMemory<T>::Zeros(stat_dim_, gpu_ctx);
    avg_stat_ = GPUMemory<T>::Zeros(stat_dim_, gpu_ctx);
    running_stat_ = GPUMemory<T>::Zeros(stat_dim_, gpu_ctx);
  }

  void step(function<void(TxnId, size_t, shared_ptr<Op>)>& compute_batch_fn, shared_ptr<Context> ctx) {
    size_t num_batches = (cfg_.minibatch_size + cfg_.compute_batch_size - 1) / cfg_.compute_batch_size;
    for (size_t batch_nr = 0; batch_nr < num_batches; ++batch_nr) {
      auto batch_txn = TxnId();
      size_t actual_batch_size = std::min(cfg_.compute_batch_size, cfg_.minibatch_size - batch_nr * cfg_.compute_batch_size);
      (compute_batch_fn)(batch_txn, actual_batch_size, sum_loss_);
      if (0 == batch_nr) {
        {
          any param_writer(param_);
          size_t write_offset = sum_loss_->store_val(batch_txn, params_, 0, &param_writer);
          assert(write_offset == param_dim_);
        }
        {
          any stat_writer(running_stat_);
          size_t write_offset = sum_loss_->store_val(batch_txn, fixed_stats_, 0, &stat_writer);
          assert(write_offset == stat_dim_);
        }
      }
      {
        any batch_sum_grad_writer(batch_sum_grad_);
        size_t write_offset = sum_loss_->store_grad(batch_txn, grads_, 0, &batch_sum_grad_writer);
        assert(write_offset == param_dim_);
      }
      {
        any batch_stat_writer(batch_stat_);
        size_t write_offset = sum_loss_->store_val(batch_txn, batch_stats_, 0, &batch_stat_writer);
        assert(write_offset == stat_dim_);
      }
      GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
      auto conn = gpu_ctx->conn();
      if (0 == batch_nr) {
        sum_grad_.copy(batch_sum_grad_, param_dim_, 0, 0, conn);
        avg_stat_.copy(batch_stat_, stat_dim_, 0, 0, conn);
      } else {
        sum_grad_.add(batch_sum_grad_, param_dim_, 0, 0, conn);
        float alpha = 1.0f / static_cast<float>(batch_nr + 1);
        avg_stat_.add_online(alpha, batch_stat_, stat_dim_, 0, 0, conn);
      }
    }

    {
      GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
      auto conn = gpu_ctx->conn();
      if (cfg_.l2_reg != 0.0) {
        sum_grad_.add_scaled(cfg_.l2_reg, param_, 1.0f, param_dim_, 0, 0, conn);
      }
      float alpha = static_cast<float>(cfg_.step_size->at(iter_count_));
      float mu = static_cast<float>(cfg_.momentum->at(iter_count_));
      momentum_grad_.add_scaled(1.0f, sum_grad_, mu, param_dim_, 0, 0, conn);
      if (cfg_.nesterov) {
        float next_alpha = static_cast<float>(cfg_.step_size->at(iter_count_ + 1));
        float next_mu = static_cast<float>(cfg_.momentum->at(iter_count_ + 1));
        param_.add_scaled(-alpha, sum_grad_, 1.0f, param_dim_, 0, 0, conn);
        param_.add_scaled(-next_alpha * next_mu, momentum_grad_, 1.0f, param_dim_, 0, 0, conn);
      } else {
        param_.add_scaled(-alpha, momentum_grad_, 1.0f, param_dim_, 0, 0, conn);
      }
      const float epsilon = cfg_.batch_norm_rate;
      running_stat_.add_online(epsilon, avg_stat_, stat_dim_, 0, 0, conn);
    }

    auto update_txn = TxnId();
    {
      any param_reader(param_);
      size_t read_offset = sum_loss_->load_val(update_txn, params_, 0, &param_reader);
      assert(read_offset == param_dim_);
    }
    {
      any stat_reader(running_stat_);
      size_t read_offset = sum_loss_->load_val(update_txn, fixed_stats_, 0, &stat_reader);
      assert(read_offset == stat_dim_);
    }

    ++iter_count_;
  }

private:
  SGDConfig cfg_;
  size_t iter_count_;
  shared_ptr<Op> sum_loss_;
  VarIdSet params_;
  VarIdSet grads_;
  VarIdSet batch_stats_;
  VarIdSet fixed_stats_;
  size_t param_dim_;
  size_t stat_dim_;
  GPUMemory<T> param_;
  GPUMemory<T> batch_sum_grad_;
  GPUMemory<T> sum_grad_;
  GPUMemory<T> momentum_grad_;
  GPUMemory<T> batch_stat_;
  GPUMemory<T> avg_stat_;
  GPUMemory<T> running_stat_;
};

template <typename T>
class ParallelSGDOptimizer {
public:
  ParallelSGDOptimizer(SGDConfig cfg, shared_ptr<Op> sum_loss, vector<VarIdSet> params, vector<VarIdSet> grads, vector<VarIdSet> batch_stats, vector<VarIdSet> fixed_stats)
    : num_ranks_(0), cfg_(cfg), sum_loss_(sum_loss), params_(params), grads_(grads), batch_stats_(batch_stats), fixed_stats_(fixed_stats)
  {}
  ~ParallelSGDOptimizer() {}

  void reload_param(shared_ptr<Context> parallel_ctx) {
    auto update_txn = TxnId();
    for (size_t rank = 0; rank < num_ranks_; ++rank) {
      any param_reader(param_.at(rank));
      size_t read_offset = sum_loss_->load_val(update_txn, params_.at(rank), 0, &param_reader);
      assert(read_offset == param_dim_);
    }
  }

  void reload_lagged_param(shared_ptr<Context> parallel_ctx) {
    {
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(parallel_ctx.get());

      mgpu_ctx->sync();

      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        NCCL_CHECK(ncclGroupStart());
      }
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        auto dev_ctx = mgpu_ctx->device_context(rank);
        auto conn = dev_ctx->conn();
        NCCL_CHECK(ncclBcast(
            lagged_param_.at(rank).dptr(),
            param_dim_,
            ncclFloat,
            0,
            mgpu_ctx->nccl_comm(rank),
            conn.stream()));
      }
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        NCCL_CHECK(ncclGroupEnd());
      }

      mgpu_ctx->sync();
    }

    auto update_txn = TxnId();
    for (size_t rank = 0; rank < num_ranks_; ++rank) {
      any param_reader(lagged_param_.at(rank));
      size_t read_offset = sum_loss_->load_val(update_txn, params_.at(rank), 0, &param_reader);
      assert(read_offset == param_dim_);
    }
  }

  void reset(shared_ptr<Context> parallel_ctx) {
    MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(parallel_ctx.get());
    num_ranks_ = mgpu_ctx->num_ranks();
    {
      auto init_txn = TxnId();
      for (size_t rank = 0; rank < num_ranks_; ++rank) {
        sum_loss_->persist(init_txn, params_.at(rank));
        sum_loss_->persist(init_txn, fixed_stats_.at(rank));
      }
      for (size_t rank = 0; rank < num_ranks_; ++rank) {
        if (0 == rank) {
          param_dim_ = sum_loss_->store_val(init_txn, params_.at(rank), 0, NULL);
          stat_dim_ = sum_loss_->store_val(init_txn, fixed_stats_.at(rank), 0, NULL);
          std::clog << "DEBUG: SGD: param dim: " << param_dim_ << std::endl;
          std::clog << "DEBUG: SGD: stat dim:  " << stat_dim_ << std::endl;
        } else {
          assert(param_dim_ == sum_loss_->store_val(init_txn, params_.at(rank), 0, NULL));
          assert(stat_dim_ == sum_loss_->store_val(init_txn, fixed_stats_.at(rank), 0, NULL));
        }
      }
    }
    iter_count_ = 0;
    lagged_param_.clear();
    param_.clear();
    batch_sum_grad_.clear();
    sum_grad_.clear();
    stat_.clear();
    batch_stat_.clear();
    avg_stat_.clear();
    for (size_t rank = 0; rank < num_ranks_; ++rank) {
      auto dev_ctx = mgpu_ctx->device_context(rank);
      lagged_param_.push_back(GPUMemory<T>::Zeros(param_dim_, dev_ctx.get()));
      param_.push_back(GPUMemory<T>::Zeros(param_dim_, dev_ctx.get()));
      batch_sum_grad_.push_back(GPUMemory<T>::Zeros(param_dim_, dev_ctx.get()));
      sum_grad_.push_back(GPUMemory<T>::Zeros(param_dim_, dev_ctx.get()));
      stat_.push_back(GPUMemory<T>::Zeros(stat_dim_, dev_ctx.get()));
      batch_stat_.push_back(GPUMemory<T>::Zeros(stat_dim_, dev_ctx.get()));
      avg_stat_.push_back(GPUMemory<T>::Zeros(stat_dim_, dev_ctx.get()));
      running_stat_.push_back(GPUMemory<T>::Zeros(stat_dim_, dev_ctx.get()));
      if (0 == rank) {
        momentum_grad_ = GPUMemory<T>::Zeros(param_dim_, dev_ctx.get());
        //running_stat_ = GPUMemory<T>::Zeros(stat_dim_, dev_ctx.get());
      }
    }
    {
      auto init_txn = TxnId();
      for (size_t rank = 0; rank < num_ranks_; ++rank) {
        sum_loss_->persist(init_txn, params_.at(rank));
        sum_loss_->persist(init_txn, fixed_stats_.at(rank));
      }
      {
        any param_writer(param_.at(0));
        size_t write_offset = sum_loss_->store_val(init_txn, params_.at(0), 0, &param_writer);
        assert(write_offset == param_dim_);
      }
      {
        auto dev_ctx = mgpu_ctx->device_context(0);
        auto conn = dev_ctx->conn();
        lagged_param_.at(0).copy(param_.at(0), param_dim_, 0, 0, conn);
      }
      for (size_t rank = 1; rank < num_ranks_; ++rank) {
        auto dev_ctx = mgpu_ctx->device_context(rank);
        {
          auto conn = dev_ctx->conn();
          lagged_param_.at(rank).copy(param_.at(0), param_dim_, 0, 0, conn);
          param_.at(rank).copy(param_.at(0), param_dim_, 0, 0, conn);
        }
        dev_ctx->sync();
        any param_reader(param_.at(rank));
        size_t read_offset = sum_loss_->load_val(init_txn, params_.at(rank), 0, &param_reader);
        assert(read_offset == param_dim_);
      }
    }
  }

  void step(function<void(TxnId, size_t, shared_ptr<Op>)>& compute_batch_fn, shared_ptr<Context> parallel_ctx) {
    size_t num_batches = (cfg_.minibatch_size + cfg_.compute_batch_size - 1) / cfg_.compute_batch_size;
    for (size_t batch_nr = 0; batch_nr < num_batches; ++batch_nr) {
      auto batch_txn = TxnId();
      size_t actual_batch_size = std::min(cfg_.compute_batch_size, cfg_.minibatch_size - batch_nr * cfg_.compute_batch_size);
      (compute_batch_fn)(batch_txn, actual_batch_size, sum_loss_);
      if (0 == batch_nr) {
        for (size_t rank = 0; rank < num_ranks_; ++rank) {
          any param_writer(param_.at(rank));
          size_t write_offset = sum_loss_->store_val(batch_txn, params_.at(rank), 0, &param_writer);
          assert(write_offset == param_dim_);
        }
        /*for (size_t rank = 0; rank < num_ranks_; ++rank) {
          any stat_writer(stat_.at(rank));
          size_t write_offset = sum_loss_->store_val(batch_txn, fixed_stats_.at(rank), 0, &stat_writer);
          assert(write_offset == stat_dim_);
        }*/
      }
      for (size_t rank = 0; rank < num_ranks_; ++rank) {
        any batch_sum_grad_writer(batch_sum_grad_.at(rank));
        size_t write_offset = sum_loss_->store_grad(batch_txn, grads_.at(rank), 0, &batch_sum_grad_writer);
        assert(write_offset == param_dim_);
      }
      for (size_t rank = 0; rank < num_ranks_; ++rank) {
        any batch_stat_writer(batch_stat_.at(rank));
        size_t write_offset = sum_loss_->store_val(batch_txn, batch_stats_.at(rank), 0, &batch_stat_writer);
        assert(write_offset == stat_dim_);
      }
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(parallel_ctx.get());
      for (size_t rank = 0; rank < num_ranks_; ++rank) {
        auto dev_ctx = mgpu_ctx->device_context(rank);
        auto conn = dev_ctx->conn();
        if (0 == batch_nr) {
          sum_grad_.at(rank).copy(batch_sum_grad_.at(rank), param_dim_, 0, 0, conn);
          avg_stat_.at(rank).copy(batch_stat_.at(rank), stat_dim_, 0, 0, conn);
        } else {
          sum_grad_.at(rank).add(batch_sum_grad_.at(rank), param_dim_, 0, 0, conn);
          float alpha = 1.0f / static_cast<float>(batch_nr + 1);
          avg_stat_.at(rank).add_online(alpha, batch_stat_.at(rank), stat_dim_, 0, 0, conn);
        }
      }
    }

    {
      MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(parallel_ctx.get());

      mgpu_ctx->sync();

      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        NCCL_CHECK(ncclGroupStart());
      }
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        auto dev_ctx = mgpu_ctx->device_context(rank);
        auto conn = dev_ctx->conn();
        NCCL_CHECK(ncclReduce(
            sum_grad_.at(rank).dptr(),
            sum_grad_.at(rank).dptr(),
            param_dim_,
            ncclFloat,
            ncclSum,
            0,
            mgpu_ctx->nccl_comm(rank),
            conn.stream()));
      }
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        NCCL_CHECK(ncclGroupEnd());
      }

      /*for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        NCCL_CHECK(ncclGroupStart());
      }
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        auto dev_ctx = mgpu_ctx->device_context(rank);
        auto conn = dev_ctx->conn();
        NCCL_CHECK(ncclReduce(
            avg_stat_.at(rank).dptr(),
            avg_stat_.at(rank).dptr(),
            stat_dim_,
            ncclFloat,
            ncclSum,
            0,
            mgpu_ctx->nccl_comm(rank),
            conn.stream()));
      }
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        NCCL_CHECK(ncclGroupEnd());
      }*/

      mgpu_ctx->sync();

      {
        auto root_dev_ctx = mgpu_ctx->device_context(0);
        auto conn = root_dev_ctx->conn();

        if (cfg_.l2_reg != 0.0) {
          sum_grad_.at(0).add_scaled(cfg_.l2_reg, param_.at(0), 1.0f, param_dim_, 0, 0, conn);
        }

        float mu = static_cast<float>(cfg_.momentum->at(iter_count_));
        float alpha = static_cast<float>(cfg_.step_size->at(iter_count_));
        momentum_grad_.add_scaled(1.0f, sum_grad_.at(0), mu, param_dim_, 0, 0, conn);
        if (cfg_.nesterov) {
          float next_alpha = static_cast<float>(cfg_.step_size->at(iter_count_ + 1));
          param_.at(0).add_scaled(-alpha, sum_grad_.at(0), 1.0f, param_dim_, 0, 0, conn);
          lagged_param_.at(0).copy(param_.at(0), param_dim_, 0, 0, conn);
          param_.at(0).add_scaled(-next_alpha * mu, momentum_grad_, 1.0f, param_dim_, 0, 0, conn);
        } else {
          param_.at(0).add_scaled(-alpha, momentum_grad_, 1.0f, param_dim_, 0, 0, conn);
          lagged_param_.at(0).copy(param_.at(0), param_dim_, 0, 0, conn);
        }
      }

      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        auto dev_ctx = mgpu_ctx->device_context(rank);
        auto conn = dev_ctx->conn();

        const float epsilon = cfg_.batch_norm_rate;
        running_stat_.at(rank).add_online(epsilon, avg_stat_.at(rank), stat_dim_, 0, 0, conn);
        stat_.at(rank).copy(running_stat_.at(rank), stat_dim_, 0, 0, conn);
      }

      mgpu_ctx->sync();

      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        NCCL_CHECK(ncclGroupStart());
      }
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        auto dev_ctx = mgpu_ctx->device_context(rank);
        auto conn = dev_ctx->conn();
        NCCL_CHECK(ncclBcast(
            param_.at(rank).dptr(),
            param_dim_,
            ncclFloat,
            0,
            mgpu_ctx->nccl_comm(rank),
            conn.stream()));
      }
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        NCCL_CHECK(ncclGroupEnd());
      }

      /*for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        NCCL_CHECK(ncclGroupStart());
      }
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        auto dev_ctx = mgpu_ctx->device_context(rank);
        auto conn = dev_ctx->conn();
        NCCL_CHECK(ncclBcast(
            stat_.at(rank).dptr(),
            stat_dim_,
            ncclFloat,
            0,
            mgpu_ctx->nccl_comm(rank),
            conn.stream()));
      }
      for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
        NCCL_CHECK(ncclGroupEnd());
      }*/

      mgpu_ctx->sync();
    }

    auto update_txn = TxnId();
    for (size_t rank = 0; rank < num_ranks_; ++rank) {
      any param_reader(param_.at(rank));
      size_t read_offset = sum_loss_->load_val(update_txn, params_.at(rank), 0, &param_reader);
      assert(read_offset == param_dim_);
    }
    for (size_t rank = 0; rank < num_ranks_; ++rank) {
      any stat_reader(stat_.at(rank));
      size_t read_offset = sum_loss_->load_val(update_txn, fixed_stats_.at(rank), 0, &stat_reader);
      assert(read_offset == stat_dim_);
    }

    ++iter_count_;
  }

private:
  size_t num_ranks_;
  SGDConfig cfg_;
  size_t iter_count_;
  shared_ptr<Op> sum_loss_;
  vector<VarIdSet> params_;
  vector<VarIdSet> grads_;
  vector<VarIdSet> batch_stats_;
  vector<VarIdSet> fixed_stats_;
  size_t param_dim_;
  size_t stat_dim_;
  vector<GPUMemory<T>> lagged_param_;
  vector<GPUMemory<T>> param_;
  vector<GPUMemory<T>> batch_sum_grad_;
  vector<GPUMemory<T>> sum_grad_;
  GPUMemory<T> momentum_grad_;
  vector<GPUMemory<T>> stat_;
  vector<GPUMemory<T>> batch_stat_;
  vector<GPUMemory<T>> avg_stat_;
  vector<GPUMemory<T>> running_stat_;
};

template <typename T>
class DataParallelSGDOptimizer {
public:

private:
  size_t num_ranks_;
  SGDConfig cfg_;
  size_t iter_count_;
  vector<shared_ptr<Op>> sum_loss_;
  vector<VarIdSet> params_;
  vector<VarIdSet> grads_;
  vector<VarIdSet> batch_stats_;
  vector<VarIdSet> fixed_stats_;
  size_t param_dim_;
  size_t stat_dim_;
  vector<GPUMemory<T>> param_;
  vector<GPUMemory<T>> batch_sum_grad_;
  vector<GPUMemory<T>> sum_grad_;
  GPUMemory<T> momentum_grad_;
  vector<GPUMemory<T>> stat_;
  vector<GPUMemory<T>> batch_stat_;
  vector<GPUMemory<T>> avg_stat_;
  GPUMemory<T> running_stat_;
};

} // namespace arraydiff

#endif
