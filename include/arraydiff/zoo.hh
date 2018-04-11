#ifndef ARRAYDIFF_ZOO_HH
#define ARRAYDIFF_ZOO_HH

#include "arraydiff/array_gpu.hh"
#include "arraydiff/op.hh"

#include <functional>
#include <memory>
#include <random>

namespace arraydiff {

using std::function;
using std::shared_ptr;

template <typename V>
class ForwardOp {
public:
  //ForwardOp() {}

  shared_ptr<Var<V>> op;
  VarIdSet inputs;
};

template <typename V>
class ParamOp {
public:
  shared_ptr<Var<V>> op;
  VarIdSet params;
  VarIdSet grads;
};

template <typename V>
class ParallelParamOp {
public:
  vector<shared_ptr<Var<V>>> ops;
  vector<VarIdSet> params;
  vector<VarIdSet> grads;
};

template <typename V>
class BatchNormParamOp {
public:
  shared_ptr<Var<V>> op;
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;
};

template <typename V>
class ParallelBatchNormParamOp {
public:
  vector<shared_ptr<Var<V>>> ops;
  vector<VarIdSet> params;
  vector<VarIdSet> grads;
  vector<VarIdSet> batch_stats;
  vector<VarIdSet> fixed_stats;
};

template <typename V, typename A>
ParamOp<V> linear(size_t filters, shared_ptr<Var<V>> input, function<void(A*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, shared_ptr<Context> ctx);

template <typename V, typename A>
ParallelParamOp<V> parallel_linear(size_t filters, vector<shared_ptr<Var<V>>> inputs, function<void(A*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, shared_ptr<Context> ctx);

template <typename V, typename A>
ParamOp<V> conv2d(Conv2DShape shape, shared_ptr<Var<V>> input, function<void(A*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, bool enable_bias, shared_ptr<Context> ctx);

template <typename V, typename A>
ParallelParamOp<V> parallel_conv2d(Conv2DShape shape, vector<shared_ptr<Var<V>>> inputs, function<void(A*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, bool enable_bias, shared_ptr<Context> ctx);

template <typename V, typename A>
BatchNormParamOp<V> batch_norm_conv2d(Conv2DShape shape, shared_ptr<bool> switch_var, shared_ptr<Var<V>> input, function<void(A*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, shared_ptr<Context> ctx);

template <typename V, typename A>
ParallelBatchNormParamOp<V> parallel_batch_norm_conv2d(Conv2DShape shape, shared_ptr<bool> switch_var, vector<shared_ptr<Var<V>>> inputs, function<void(A*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, shared_ptr<Context> ctx);

template <typename V>
BatchNormParamOp<V> residual_conv2d(Conv2DShape shape, shared_ptr<bool> switch_var, shared_ptr<Var<V>> input, shared_ptr<Context> ctx);

template <typename V>
BatchNormParamOp<V> proj_residual_conv2d(Conv2DShape shape, size_t stride, shared_ptr<bool> switch_var, shared_ptr<Var<V>> input, shared_ptr<Context> ctx);

template <typename V>
BatchNormParamOp<V> residual_v2_conv2d(Conv2DShape shape, shared_ptr<bool> switch_var, shared_ptr<Var<V>> input, shared_ptr<Context> ctx);

template <typename V>
BatchNormParamOp<V> proj_residual_v2_conv2d(Conv2DShape shape, size_t stride, shared_ptr<bool> switch_var, shared_ptr<Var<V>> input, shared_ptr<Context> ctx);

template <typename V>
BatchNormParamOp<V> bottleneck_residual_v2_conv2d(Conv2DShape shape, size_t bottleneck, shared_ptr<bool> switch_var, shared_ptr<Var<V>> input, shared_ptr<Context> ctx);

template <typename V>
BatchNormParamOp<V> proj_bottleneck_residual_v2_conv2d(Conv2DShape shape, size_t stride, size_t bottleneck, shared_ptr<bool> switch_var, shared_ptr<Var<V>> input, shared_ptr<Context> ctx);

template <typename V>
ParallelBatchNormParamOp<V> spatial_parallel_residual_v2_conv2d(Conv2DShape shape, shared_ptr<bool> switch_var, vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx);

template <typename V>
ParallelBatchNormParamOp<V> spatial_parallel_proj_residual_v2_conv2d(Conv2DShape shape, size_t stride, shared_ptr<bool> switch_var, vector<shared_ptr<Var<V>>> input, shared_ptr<Context> ctx);

template <typename V>
ParallelBatchNormParamOp<V> spatial_parallel_bottleneck_residual_v2_conv2d(Conv2DShape shape, size_t bottleneck, shared_ptr<bool> switch_var, vector<shared_ptr<Var<V>>> inputs, shared_ptr<Context> ctx);

template <typename V>
ParallelBatchNormParamOp<V> spatial_parallel_proj_bottleneck_residual_v2_conv2d(Conv2DShape shape, size_t bottleneck, size_t stride, shared_ptr<bool> switch_var, vector<shared_ptr<Var<V>>> input, shared_ptr<Context> ctx);

template <typename Logit, typename Loss, typename SumLoss>
class CategoricalNLLLoss {
public:
  shared_ptr<Var<Logit>> logit;
  shared_ptr<Var<Loss>> loss;
  shared_ptr<Var<SumLoss>> sum_loss;
  VarIdSet inputs;
  VarIdSet labels;
  VarIdSet params;
  VarIdSet grads;
};

template <typename Logit, typename Loss, typename SumLoss>
class CategoricalNLLBatchNormLoss {
public:
  shared_ptr<bool> batch_norm_use_fixed;
  shared_ptr<Var<Logit>> op;
  shared_ptr<Var<Loss>> loss;
  shared_ptr<Var<SumLoss>> sum_loss;
  VarIdSet inputs;
  VarIdSet labels;
  VarIdSet logits;
  VarIdSet losses;
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;
  VarIdSet _preproc_inputs;
  VarIdSet _hidden_convs;
  VarIdSet _hidden_fcs;
};

template <typename Logit, typename Loss, typename SumLoss>
class ParallelCategoricalNLLBatchNormLoss {
public:
  shared_ptr<bool> batch_norm_use_fixed;
  shared_ptr<Var<Logit>> op;
  shared_ptr<Var<Loss>> loss;
  shared_ptr<Var<SumLoss>> sum_loss;
  VarIdSet inputs;
  VarIdSet labels;
  VarIdSet logits;
  VarIdSet losses;
  //vector<VarIdSet> shared_params;
  //vector<VarIdSet> shared_grads;
  vector<VarIdSet> dev_params;
  vector<VarIdSet> dev_grads;
  vector<VarIdSet> dev_batch_stats;
  vector<VarIdSet> dev_fixed_stats;
};

template <typename T>
class ImagenetPreproc {
public:
  static ForwardOp<GPUBatchArray3D<T>> Make(size_t input_w, size_t input_h, size_t batch_size, shared_ptr<Context> ctx);
};

class ConvNetConfig {
public:
  //size_t input_c;
  size_t input_w;
  size_t input_h;
  size_t batch_size;
  size_t minibatch_size;
};

template <typename T>
class ResNet {
public:
  static CategoricalNLLBatchNormLoss<GPUBatchArray1D<T>, GPUBatchArray0D<T>, GPUArray0D<T>> MakeTestNet(const ConvNetConfig& config, shared_ptr<Context> ctx);
  static CategoricalNLLBatchNormLoss<GPUBatchArray1D<T>, GPUBatchArray0D<T>, GPUArray0D<T>> MakeResNet18(const ConvNetConfig& config, shared_ptr<Context> ctx);
  static ParallelCategoricalNLLBatchNormLoss<GPUBatchArray1D<T>, GPUBatchArray0D<T>, GPUArray0D<T>> MakeParallelResNet18(const ConvNetConfig& config, shared_ptr<Context> ctx);
  static CategoricalNLLBatchNormLoss<GPUBatchArray1D<T>, GPUBatchArray0D<T>, GPUArray0D<T>> MakeResNet50(const ConvNetConfig& config, shared_ptr<Context> ctx);
  static ParallelCategoricalNLLBatchNormLoss<GPUBatchArray1D<T>, GPUBatchArray0D<T>, GPUArray0D<T>> MakeParallelResNet50(const ConvNetConfig& config, shared_ptr<Context> ctx);
};

template <typename T>
class SpatialParallelResNet {
public:
  //static shared_ptr<Var<GPUBatchArray1D<T>>> Make(const ConvNetConfig& config, shared_ptr<Context> ctx);
  static CategoricalNLLBatchNormLoss<GPUBatchArray1D<T>, GPUBatchArray0D<T>, GPUArray0D<T>> Make(const ConvNetConfig& config, shared_ptr<Context> ctx);
};

} // namespace arraydiff

#endif
