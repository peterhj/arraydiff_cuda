#ifndef ARRAYDIFF_OPTIMIZE_MPI_HH
#define ARRAYDIFF_OPTIMIZE_MPI_HH

#include "arraydiff/context.hh"
#include "arraydiff/op.hh"
#include "arraydiff/optimize.hh"

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <utility>

#define MPI_CHECK(e) (assert(MPI_SUCCESS == (e)))

namespace arraydiff {

template <typename T>
class MPIDataParallelSGDOptimizer {
public:
  MPIDataParallelSGDOptimizer(SGDConfig cfg, shared_ptr<Op> sum_loss, VarIdSet params, VarIdSet grads, VarIdSet batch_stats, VarIdSet fixed_stats)
    : cfg_(cfg), sum_loss_(sum_loss), params_(params), grads_(grads), batch_stats_(batch_stats), fixed_stats_(fixed_stats)
  {
  }
  ~MPIDataParallelSGDOptimizer() {
  }

  size_t worker_rank() const {
    return worker_rank_;
  }

  void reset(shared_ptr<Context> ctx) {
    {
      int comm_rank = 0;
      int comm_size = 0;
      MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
      MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));
      worker_rank_ = comm_rank;
      num_workers_ = comm_size;
      std::clog << "DEBUG: SGD: rank: " << worker_rank_ << " size: " << num_workers_ << std::endl;
    }
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
    recv_sum_grad_ = GPUMemory<T>::Zeros(param_dim_, gpu_ctx);
    momentum_grad_ = GPUMemory<T>::Zeros(param_dim_, gpu_ctx);
    batch_stat_ = GPUMemory<T>::Zeros(stat_dim_, gpu_ctx);
    avg_stat_ = GPUMemory<T>::Zeros(stat_dim_, gpu_ctx);
    running_stat_ = GPUMemory<T>::Zeros(stat_dim_, gpu_ctx);
    {
      if (0 == worker_rank_) {
        auto init_txn = TxnId();
        sum_loss_->persist(init_txn, params_);
        sum_loss_->persist(init_txn, fixed_stats_);
        {
          any param_writer(param_);
          size_t write_offset = sum_loss_->store_val(init_txn, params_, 0, &param_writer);
          assert(write_offset == param_dim_);
        }
      }
      gpu_ctx->sync();
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      MPI_CHECK(MPI_Bcast(
          param_.dptr(),
          param_dim_,
          MPI_FLOAT,
          0,
          MPI_COMM_WORLD));
      if (0 != worker_rank_) {
        auto init_txn = TxnId();
        sum_loss_->persist(init_txn, fixed_stats_);
        {
          any param_reader(param_);
          size_t read_offset = sum_loss_->load_val(init_txn, params_, 0, &param_reader);
          assert(read_offset == param_dim_);
        }
      }
      gpu_ctx->sync();
    }
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
      gpu_ctx->sync();

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      MPI_CHECK(MPI_Allreduce(
          sum_grad_.dptr(),
          recv_sum_grad_.dptr(),
          param_dim_,
          MPI_FLOAT,
          MPI_SUM,
          MPI_COMM_WORLD));

      gpu_ctx->sync();
    }

    {
      GPUContext* gpu_ctx = dynamic_cast<GPUContext*>(ctx.get());
      auto conn = gpu_ctx->conn();
      if (cfg_.l2_reg != 0.0) {
        recv_sum_grad_.add_scaled(cfg_.l2_reg, param_, 1.0f, param_dim_, 0, 0, conn);
      }
      float mu = static_cast<float>(cfg_.momentum->at(iter_count_));
      float alpha = static_cast<float>(cfg_.step_size->at(iter_count_));
      momentum_grad_.add_scaled(1.0f, recv_sum_grad_, mu, param_dim_, 0, 0, conn);
      if (cfg_.nesterov) {
        param_.add_scaled(-alpha, recv_sum_grad_, 1.0f, param_dim_, 0, 0, conn);
        param_.add_scaled(-alpha * mu, momentum_grad_, 1.0f, param_dim_, 0, 0, conn);
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
  size_t worker_rank_;
  size_t num_workers_;
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
  GPUMemory<T> recv_sum_grad_;
  GPUMemory<T> momentum_grad_;
  GPUMemory<T> batch_stat_;
  GPUMemory<T> avg_stat_;
  GPUMemory<T> running_stat_;
};

} // namespace arraydiff

#endif
