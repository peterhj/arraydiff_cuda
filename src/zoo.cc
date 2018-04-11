#include "arraydiff/zoo.hh"
#include "arraydiff/array_abstract.hh"
#include "arraydiff/array_gpu.hh"
#include "arraydiff/context.hh"
#include "arraydiff/op.hh"

#include <experimental/optional>
#include <functional>
#include <iostream>
#include <memory>

namespace arraydiff {

using std::experimental::make_optional;
using std::experimental::nullopt;
using std::experimental::nullopt_t;
using std::function;
using std::make_shared;
using std::shared_ptr;

void parallel_vars_insert_all(vector<VarIdSet>& dst_vars, const vector<VarIdSet>& src_vars) {
  assert(dst_vars.size() == src_vars.size());
  for (size_t rank = 0; rank < dst_vars.size(); ++rank) {
    dst_vars.at(rank).insert_all(src_vars.at(rank));
  }
}

template <>
ParamOp<GPUBatchArray1D<float>> linear(size_t filters, shared_ptr<Var<GPUBatchArray1D<float>>> input, function<void(GPUArray2D<float>*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, shared_ptr<Context> ctx) {
  VarIdSet params;
  VarIdSet grads;

  size_t in_filters = input->shape().shape;
  auto kernel_src = array_src<GPUArray2D<float>, Shape2D>(Shape2D(filters, in_filters), ctx);
  auto kernel = initialize<GPUArray2D<float>>(
      kernel_init,
      kernel_src,
      ctx);
  params.insert(kernel->vars().at(0));
  grads.insert(kernel->vars().at(1));

  auto bias_src = array_src<GPUArray1D<float>, Shape1D>(Shape1D(filters), ctx);
  auto bias = initialize<GPUArray1D<float>>(
      //[](auto val, auto ctx) { zeros_init_fn(val, ctx); },
      zeros_init_gen<GPUArray1D<float>>(),
      bias_src,
      ctx);
  params.insert(bias->vars().at(0));
  grads.insert(bias->vars().at(1));

  auto op = make_shared<CublasLinearOp<GPUBatchArray1D<float>, GPUArray2D<float>, GPUArray1D<float>>>(filters, input, kernel, bias, ctx);

  ParamOp<GPUBatchArray1D<float>> param_op = {
    .op = op,
    .params = params,
    .grads = grads,
  };
  return param_op;
}

template <>
ParallelParamOp<GPUBatchArray1D<float>> parallel_linear(size_t filters, vector<shared_ptr<Var<GPUBatchArray1D<float>>>> inputs, function<void(GPUArray2D<float>*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, shared_ptr<Context> ctx) {
  vector<shared_ptr<Var<GPUBatchArray1D<float>>>> ops;
  vector<VarIdSet> params;
  vector<VarIdSet> grads;

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    VarIdSet rank_params;
    VarIdSet rank_grads;

    auto dev_ctx = mgpu_ctx->device_context(rank);

    auto fc1 = linear<GPUBatchArray1D<float>, GPUArray2D<float>>(filters, inputs.at(rank), kernel_init, dev_ctx);
    rank_params.insert_all(fc1.params);
    rank_grads.insert_all(fc1.grads);

    ops.push_back(fc1.op);
    params.push_back(rank_params);
    grads.push_back(rank_grads);
  }

  ParallelParamOp<GPUBatchArray1D<float>> param_op = {
    .ops = ops,
    .params = params,
    .grads = grads,
  };
  return param_op;
}

template <>
ParamOp<GPUBatchArray3D<float>> conv2d(Conv2DShape shape, shared_ptr<Var<GPUBatchArray3D<float>>> input, function<void(GPUArray4D<float>*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, bool enable_bias, shared_ptr<Context> ctx) {
  VarIdSet params;
  VarIdSet grads;

  size_t in_filters = input->shape().shape.at(2);
  std::clog << "DEBUG: conv2d: kernel shape: "
    << shape.kernel.at(0) << " "
    << shape.kernel.at(1) << " "
    << in_filters << " "
    << shape.filters << std::endl;
  auto kernel_src = array_src<GPUArray4D<float>, Shape4D>(Shape4D(shape.kernel.at(0), shape.kernel.at(1), in_filters, shape.filters), ctx);
  auto kernel = initialize<GPUArray4D<float>>(
      kernel_init,
      kernel_src,
      ctx);
  params.insert(kernel->vars().at(0));
  grads.insert(kernel->vars().at(1));

  shared_ptr<Var<GPUBatchArray3D<float>>> op;
  if (enable_bias) {
    auto bias_src = array_src<GPUArray1D<float>, Shape1D>(Shape1D(shape.filters), ctx);
    auto bias = initialize<GPUArray1D<float>>(
        //[](auto val, auto ctx) { zeros_init_fn(val, ctx); },
        zeros_init_gen<GPUArray1D<float>>(),
        bias_src,
        ctx);
    params.insert(bias->vars().at(0));
    grads.insert(bias->vars().at(1));
    op = make_shared<CudnnConv2DOp<GPUBatchArray3D<float>, GPUArray4D<float>, GPUArray1D<float>>>(shape, input, kernel, bias, ctx);
  } else {
    op = make_shared<CudnnConv2DOp<GPUBatchArray3D<float>, GPUArray4D<float>, GPUArray1D<float>>>(shape, input, kernel, ctx);
  }

  ParamOp<GPUBatchArray3D<float>> param_op = {
    .op = op,
    .params = params,
    .grads = grads,
  };
  return param_op;
}

template <>
ParallelParamOp<GPUBatchArray3D<float>> parallel_conv2d(Conv2DShape shape, vector<shared_ptr<Var<GPUBatchArray3D<float>>>> inputs, function<void(GPUArray4D<float>*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, bool enable_bias, shared_ptr<Context> ctx) {
  vector<shared_ptr<Var<GPUBatchArray3D<float>>>> ops;
  vector<VarIdSet> params;
  vector<VarIdSet> grads;

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    VarIdSet rank_params;
    VarIdSet rank_grads;

    auto dev_ctx = mgpu_ctx->device_context(rank);

    auto conv1 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(shape, inputs.at(rank), kernel_init, enable_bias, dev_ctx);
    rank_params.insert_all(conv1.params);
    rank_grads.insert_all(conv1.grads);

    ops.push_back(conv1.op);
    params.push_back(rank_params);
    grads.push_back(rank_grads);
  }

  ParallelParamOp<GPUBatchArray3D<float>> param_op = {
    .ops = ops,
    .params = params,
    .grads = grads,
  };
  return param_op;
}

template <>
BatchNormParamOp<GPUBatchArray3D<float>> batch_norm_conv2d(Conv2DShape shape, shared_ptr<bool> switch_var, shared_ptr<Var<GPUBatchArray3D<float>>> input, function<void(GPUArray4D<float>*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, shared_ptr<Context> ctx) {
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  auto conv1 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(shape, input, kernel_init, false, ctx);
  params.insert_all(conv1.params);
  grads.insert_all(conv1.grads);

  auto bnorm1 = batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(shape.filters, switch_var, conv1.op, ctx);
  params.insert_all(bnorm1.params);
  grads.insert_all(bnorm1.grads);
  batch_stats.insert_all(bnorm1.batch_stats);
  fixed_stats.insert_all(bnorm1.fixed_stats);

  BatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .op = bnorm1.op,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
ParallelBatchNormParamOp<GPUBatchArray3D<float>> parallel_batch_norm_conv2d(Conv2DShape shape, shared_ptr<bool> switch_var, vector<shared_ptr<Var<GPUBatchArray3D<float>>>> inputs, function<void(GPUArray4D<float>*, std::mt19937_64*, shared_ptr<Context> ctx)> kernel_init, shared_ptr<Context> ctx) {
  vector<shared_ptr<Var<GPUBatchArray3D<float>>>> ops;
  vector<VarIdSet> params;
  vector<VarIdSet> grads;
  vector<VarIdSet> batch_stats;
  vector<VarIdSet> fixed_stats;

  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
  for (size_t rank = 0; rank < mgpu_ctx->num_ranks(); ++rank) {
    VarIdSet rank_params;
    VarIdSet rank_grads;
    VarIdSet rank_batch_stats;
    VarIdSet rank_fixed_stats;

    auto dev_ctx = mgpu_ctx->device_context(rank);

    auto conv1 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(shape, inputs.at(rank), kernel_init, false, dev_ctx);
    rank_params.insert_all(conv1.params);
    rank_grads.insert_all(conv1.grads);

    auto bnorm1 = batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(shape.filters, switch_var, conv1.op, dev_ctx);
    rank_params.insert_all(bnorm1.params);
    rank_grads.insert_all(bnorm1.grads);
    rank_batch_stats.insert_all(bnorm1.batch_stats);
    rank_fixed_stats.insert_all(bnorm1.fixed_stats);

    ops.push_back(bnorm1.op);
    params.push_back(rank_params);
    grads.push_back(rank_grads);
    batch_stats.push_back(rank_batch_stats);
    fixed_stats.push_back(rank_fixed_stats);
  }

  ParallelBatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .ops = ops,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
BatchNormParamOp<GPUBatchArray3D<float>> residual_conv2d(Conv2DShape shape, shared_ptr<bool> switch_var, shared_ptr<Var<GPUBatchArray3D<float>>> input, shared_ptr<Context> ctx) {
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  auto conv1 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(shape, switch_var, input, kaiming_init_gen<GPUArray4D<float>>(input->shape().shape.at(2) * shape.kernel.at(0) * shape.kernel.at(1), shape.filters), ctx);
  params.insert_all(conv1.params);
  grads.insert_all(conv1.grads);
  batch_stats.insert_all(conv1.batch_stats);
  fixed_stats.insert_all(conv1.fixed_stats);

  auto relu1 = relu<GPUBatchArray3D<float>>(conv1.op, ctx);
  //auto relu1 = relu_in_place<GPUBatchArray3D<float>>(conv1.op, ctx);

  auto conv2 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(shape, switch_var, relu1, kaiming_init_gen<GPUArray4D<float>>(shape.filters * shape.kernel.at(0) * shape.kernel.at(1), shape.filters), ctx);
  params.insert_all(conv2.params);
  grads.insert_all(conv2.grads);
  batch_stats.insert_all(conv2.batch_stats);
  fixed_stats.insert_all(conv2.fixed_stats);

  vector<shared_ptr<Var<GPUBatchArray3D<float>>>> join_inputs;
  join_inputs.push_back(input);
  join_inputs.push_back(conv2.op);
  auto join = sum<GPUBatchArray3D<float>>(join_inputs, ctx);
  //auto join = sum_in_place_0<GPUBatchArray3D<float>>(join_inputs, ctx);

  auto relu2 = relu<GPUBatchArray3D<float>>(join, ctx);
  //auto relu2 = relu_in_place<GPUBatchArray3D<float>>(join, ctx);

  BatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .op = relu2,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
BatchNormParamOp<GPUBatchArray3D<float>> proj_residual_conv2d(Conv2DShape shape, size_t stride, shared_ptr<bool> switch_var, shared_ptr<Var<GPUBatchArray3D<float>>> input, shared_ptr<Context> ctx) {
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  auto proj_shape = shape;
  proj_shape.kernel = {1, 1};
  proj_shape.stride = {stride, stride};
  proj_shape.pad = {0, 0};

  auto proj_1x1 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(proj_shape, switch_var, input, kaiming_init_gen<GPUArray4D<float>>(input->shape().shape.at(2), shape.filters), ctx);
  params.insert_all(proj_1x1.params);
  grads.insert_all(proj_1x1.grads);
  batch_stats.insert_all(proj_1x1.batch_stats);
  fixed_stats.insert_all(proj_1x1.fixed_stats);

  auto conv1_shape = shape;
  conv1_shape.stride = {stride, stride};

  auto conv1 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, switch_var, input, kaiming_init_gen<GPUArray4D<float>>(input->shape().shape.at(2) * shape.kernel.at(0) * shape.kernel.at(1), shape.filters), ctx);
  params.insert_all(conv1.params);
  grads.insert_all(conv1.grads);
  batch_stats.insert_all(conv1.batch_stats);
  fixed_stats.insert_all(conv1.fixed_stats);

  auto relu1 = relu<GPUBatchArray3D<float>>(conv1.op, ctx);
  //auto relu1 = relu_in_place<GPUBatchArray3D<float>>(conv1.op, ctx);

  auto conv2 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(shape, switch_var, relu1, kaiming_init_gen<GPUArray4D<float>>(shape.filters * shape.kernel.at(0) * shape.kernel.at(1), shape.filters), ctx);
  params.insert_all(conv2.params);
  grads.insert_all(conv2.grads);
  batch_stats.insert_all(conv2.batch_stats);
  fixed_stats.insert_all(conv2.fixed_stats);

  vector<shared_ptr<Var<GPUBatchArray3D<float>>>> join_inputs;
  join_inputs.push_back(proj_1x1.op);
  join_inputs.push_back(conv2.op);
  auto join = sum<GPUBatchArray3D<float>>(join_inputs, ctx);
  //auto join = sum_in_place_0<GPUBatchArray3D<float>>(join_inputs, ctx);

  auto relu2 = relu<GPUBatchArray3D<float>>(join, ctx);
  //auto relu2 = relu_in_place<GPUBatchArray3D<float>>(join, ctx);

  BatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .op = relu2,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
BatchNormParamOp<GPUBatchArray3D<float>> residual_v2_conv2d(Conv2DShape shape, shared_ptr<bool> switch_var, shared_ptr<Var<GPUBatchArray3D<float>>> input, shared_ptr<Context> ctx) {
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  assert(input->shape().shape.at(2) == shape.filters);

  auto bnorm0 = batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(shape.filters, switch_var, input, ctx);
  params.insert_all(bnorm0.params);
  grads.insert_all(bnorm0.grads);
  batch_stats.insert_all(bnorm0.batch_stats);
  fixed_stats.insert_all(bnorm0.fixed_stats);

  auto relu0 = relu<GPUBatchArray3D<float>>(bnorm0.op, ctx);
  //auto relu0 = relu_in_place<GPUBatchArray3D<float>>(bnorm0.op, ctx);

  auto conv1 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(shape, switch_var, relu0, kaiming_init_gen<GPUArray4D<float>>(input->shape().shape.at(2) * shape.kernel.at(0) * shape.kernel.at(1), shape.filters), ctx);
  params.insert_all(conv1.params);
  grads.insert_all(conv1.grads);
  batch_stats.insert_all(conv1.batch_stats);
  fixed_stats.insert_all(conv1.fixed_stats);

  auto relu1 = relu<GPUBatchArray3D<float>>(conv1.op, ctx);
  //auto relu1 = relu_in_place<GPUBatchArray3D<float>>(conv1.op, ctx);

  auto conv2 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(shape, relu1, kaiming_init_gen<GPUArray4D<float>>(shape.filters * shape.kernel.at(0) * shape.kernel.at(1), shape.filters), false, ctx);
  params.insert_all(conv2.params);
  grads.insert_all(conv2.grads);

  vector<shared_ptr<Var<GPUBatchArray3D<float>>>> join_inputs;
  join_inputs.push_back(input);
  join_inputs.push_back(conv2.op);
  auto join = sum<GPUBatchArray3D<float>>(join_inputs, ctx);
  //auto join = sum_in_place_0<GPUBatchArray3D<float>>(join_inputs, ctx);

  BatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .op = join,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
BatchNormParamOp<GPUBatchArray3D<float>> proj_residual_v2_conv2d(Conv2DShape shape, size_t stride, shared_ptr<bool> switch_var, shared_ptr<Var<GPUBatchArray3D<float>>> input, shared_ptr<Context> ctx) {
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  auto bnorm0 = batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(shape.filters, switch_var, input, ctx);
  params.insert_all(bnorm0.params);
  grads.insert_all(bnorm0.grads);
  batch_stats.insert_all(bnorm0.batch_stats);
  fixed_stats.insert_all(bnorm0.fixed_stats);

  auto relu0 = relu<GPUBatchArray3D<float>>(bnorm0.op, ctx);
  //auto relu0 = relu_in_place<GPUBatchArray3D<float>>(bnorm0.op, ctx);

  auto proj_shape = shape;
  proj_shape.kernel = {1, 1};
  proj_shape.stride = {stride, stride};
  proj_shape.pad = {0, 0};

  //auto proj_1x1 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(proj_shape, switch_var, relu0, kaiming_init_gen<GPUArray4D<float>>(input->shape().shape.at(2), shape.filters), ctx);
  auto proj_1x1 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(proj_shape, relu0, kaiming_init_gen<GPUArray4D<float>>(input->shape().shape.at(2), shape.filters), false, ctx);
  params.insert_all(proj_1x1.params);
  grads.insert_all(proj_1x1.grads);
  //batch_stats.insert_all(proj_1x1.batch_stats);
  //fixed_stats.insert_all(proj_1x1.fixed_stats);

  auto conv1_shape = shape;
  conv1_shape.stride = {stride, stride};

  auto conv1 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, switch_var, relu0, kaiming_init_gen<GPUArray4D<float>>(input->shape().shape.at(2) * shape.kernel.at(0) * shape.kernel.at(1), shape.filters), ctx);
  params.insert_all(conv1.params);
  grads.insert_all(conv1.grads);
  batch_stats.insert_all(conv1.batch_stats);
  fixed_stats.insert_all(conv1.fixed_stats);

  auto relu1 = relu<GPUBatchArray3D<float>>(conv1.op, ctx);
  //auto relu1 = relu_in_place<GPUBatchArray3D<float>>(conv1.op, ctx);

  auto conv2 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(shape, relu1, kaiming_init_gen<GPUArray4D<float>>(shape.filters * shape.kernel.at(0) * shape.kernel.at(1), shape.filters), false, ctx);
  params.insert_all(conv2.params);
  grads.insert_all(conv2.grads);

  vector<shared_ptr<Var<GPUBatchArray3D<float>>>> join_inputs;
  join_inputs.push_back(proj_1x1.op);
  join_inputs.push_back(conv2.op);
  auto join = sum<GPUBatchArray3D<float>>(join_inputs, ctx);
  //auto join = sum_in_place_0<GPUBatchArray3D<float>>(join_inputs, ctx);

  BatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .op = join,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
BatchNormParamOp<GPUBatchArray3D<float>> bottleneck_residual_v2_conv2d(Conv2DShape shape, size_t bottleneck, shared_ptr<bool> switch_var, shared_ptr<Var<GPUBatchArray3D<float>>> input, shared_ptr<Context> ctx) {
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  assert(input->shape().shape.at(2) == bottleneck * shape.filters);

  auto bnorm0 = batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(input->shape().shape.at(2), switch_var, input, ctx);
  params.insert_all(bnorm0.params);
  grads.insert_all(bnorm0.grads);
  batch_stats.insert_all(bnorm0.batch_stats);
  fixed_stats.insert_all(bnorm0.fixed_stats);

  auto relu0 = relu<GPUBatchArray3D<float>>(bnorm0.op, ctx);

  Conv2DShape conv1_shape = shape;
  conv1_shape.kernel = {1, 1};
  conv1_shape.stride = {1, 1};
  conv1_shape.pad = {0, 0};

  auto conv1 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, switch_var, relu0, kaiming_init_gen<GPUArray4D<float>>(input->shape().shape.at(2) * conv1_shape.kernel.at(0) * conv1_shape.kernel.at(1), conv1_shape.filters), ctx);
  params.insert_all(conv1.params);
  grads.insert_all(conv1.grads);
  batch_stats.insert_all(conv1.batch_stats);
  fixed_stats.insert_all(conv1.fixed_stats);

  auto relu1 = relu<GPUBatchArray3D<float>>(conv1.op, ctx);

  Conv2DShape conv2_shape = shape;
  conv2_shape.stride = {1, 1};

  auto conv2 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv2_shape, switch_var, relu1, kaiming_init_gen<GPUArray4D<float>>(conv1_shape.filters * conv2_shape.kernel.at(0) * conv2_shape.kernel.at(1), conv2_shape.filters), ctx);
  params.insert_all(conv2.params);
  grads.insert_all(conv2.grads);
  batch_stats.insert_all(conv2.batch_stats);
  fixed_stats.insert_all(conv2.fixed_stats);

  auto relu2 = relu<GPUBatchArray3D<float>>(conv2.op, ctx);

  Conv2DShape conv3_shape = shape;
  conv3_shape.filters = bottleneck * shape.filters;
  conv3_shape.kernel = {1, 1};
  conv3_shape.stride = {1, 1};
  conv3_shape.pad = {0, 0};

  auto conv3 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv3_shape, relu2, kaiming_init_gen<GPUArray4D<float>>(conv2_shape.filters * conv3_shape.kernel.at(0) * conv3_shape.kernel.at(1), conv3_shape.filters), false, ctx);
  params.insert_all(conv3.params);
  grads.insert_all(conv3.grads);

  vector<shared_ptr<Var<GPUBatchArray3D<float>>>> join_inputs;
  join_inputs.push_back(input);
  join_inputs.push_back(conv3.op);
  auto join = sum<GPUBatchArray3D<float>>(join_inputs, ctx);

  BatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .op = join,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
BatchNormParamOp<GPUBatchArray3D<float>> proj_bottleneck_residual_v2_conv2d(Conv2DShape shape, size_t bottleneck, size_t stride, shared_ptr<bool> switch_var, shared_ptr<Var<GPUBatchArray3D<float>>> input, shared_ptr<Context> ctx) {
  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  auto bnorm0 = batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(input->shape().shape.at(2), switch_var, input, ctx);
  params.insert_all(bnorm0.params);
  grads.insert_all(bnorm0.grads);
  batch_stats.insert_all(bnorm0.batch_stats);
  fixed_stats.insert_all(bnorm0.fixed_stats);

  auto relu0 = relu<GPUBatchArray3D<float>>(bnorm0.op, ctx);

  Conv2DShape proj_shape = shape;
  proj_shape.filters = bottleneck * shape.filters;
  proj_shape.kernel = {1, 1};
  proj_shape.stride = {stride, stride};
  proj_shape.pad = {0, 0};

  auto proj = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(proj_shape, relu0, kaiming_init_gen<GPUArray4D<float>>(input->shape().shape.at(2), proj_shape.filters), false, ctx);
  params.insert_all(proj.params);
  grads.insert_all(proj.grads);

  Conv2DShape conv1_shape = shape;
  conv1_shape.kernel = {1, 1};
  conv1_shape.stride = {1, 1};
  conv1_shape.pad = {0, 0};

  auto conv1 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, switch_var, relu0, kaiming_init_gen<GPUArray4D<float>>(input->shape().shape.at(2), conv1_shape.filters), ctx);
  params.insert_all(conv1.params);
  grads.insert_all(conv1.grads);
  batch_stats.insert_all(conv1.batch_stats);
  fixed_stats.insert_all(conv1.fixed_stats);

  auto relu1 = relu<GPUBatchArray3D<float>>(conv1.op, ctx);

  Conv2DShape conv2_shape = shape;
  conv2_shape.stride = {stride, stride};

  auto conv2 = batch_norm_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv2_shape, switch_var, relu1, kaiming_init_gen<GPUArray4D<float>>(conv1_shape.filters * conv2_shape.kernel.at(0) * conv2_shape.kernel.at(1), conv2_shape.filters), ctx);
  params.insert_all(conv2.params);
  grads.insert_all(conv2.grads);
  batch_stats.insert_all(conv2.batch_stats);
  fixed_stats.insert_all(conv2.fixed_stats);

  auto relu2 = relu<GPUBatchArray3D<float>>(conv2.op, ctx);

  Conv2DShape conv3_shape = shape;
  conv3_shape.filters = bottleneck * shape.filters;
  conv3_shape.kernel = {1, 1};
  conv3_shape.stride = {1, 1};
  conv3_shape.pad = {0, 0};

  auto conv3 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv3_shape, relu2, kaiming_init_gen<GPUArray4D<float>>(conv2_shape.filters, conv3_shape.filters), false, ctx);
  params.insert_all(conv3.params);
  grads.insert_all(conv3.grads);

  vector<shared_ptr<Var<GPUBatchArray3D<float>>>> join_inputs;
  join_inputs.push_back(proj.op);
  join_inputs.push_back(conv3.op);
  auto join = sum<GPUBatchArray3D<float>>(join_inputs, ctx);

  BatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .op = join,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
ParallelBatchNormParamOp<GPUBatchArray3D<float>> spatial_parallel_residual_v2_conv2d(Conv2DShape shape, shared_ptr<bool> switch_var, vector<shared_ptr<Var<GPUBatchArray3D<float>>>> inputs, shared_ptr<Context> ctx) {
  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
  size_t num_ranks = mgpu_ctx->num_ranks();

  vector<VarIdSet> params(num_ranks);
  vector<VarIdSet> grads(num_ranks);
  vector<VarIdSet> batch_stats(num_ranks);
  vector<VarIdSet> fixed_stats(num_ranks);

  std::clog << "DEBUG: building spatial parallel residual v2 conv2d..." << std::endl;

  assert(shape.filters == inputs.at(0)->shape().shape.at(2));

  assert(0 == params.at(0).size());
  assert(0 == grads.at(0).size());
  assert(0 == batch_stats.at(0).size());
  assert(0 == fixed_stats.at(0).size());

  auto bnorm0 = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(inputs.at(0)->shape().shape.at(2), switch_var, inputs, ctx);
  parallel_vars_insert_all(params, bnorm0.params);
  parallel_vars_insert_all(grads, bnorm0.grads);
  parallel_vars_insert_all(batch_stats, bnorm0.batch_stats);
  parallel_vars_insert_all(fixed_stats, bnorm0.fixed_stats);

  auto relu0  = parallel_relu<GPUBatchArray3D<float>>(bnorm0.ops, ctx);

  auto pad1   = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(1, relu0, ctx);
  auto exch1  = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(1, pad1, ctx);

  Conv2DShape conv1_shape = shape;
  conv1_shape.halo_padded_input = true;

  auto conv1_init = kaiming_init_gen<GPUArray4D<float>>(inputs.at(0)->shape().shape.at(2) * shape.kernel.at(0) * shape.kernel.at(1), shape.filters);
  auto conv1  = parallel_conv2d<GPUBatchArray3D<float>>(conv1_shape, exch1, conv1_init, false, ctx);
  parallel_vars_insert_all(params, conv1.params);
  parallel_vars_insert_all(grads, conv1.grads);

  auto bnorm1 = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(shape.filters, switch_var, conv1.ops, ctx);
  parallel_vars_insert_all(params, bnorm1.params);
  parallel_vars_insert_all(grads, bnorm1.grads);
  parallel_vars_insert_all(batch_stats, bnorm1.batch_stats);
  parallel_vars_insert_all(fixed_stats, bnorm1.fixed_stats);

  auto relu1  = parallel_relu<GPUBatchArray3D<float>>(bnorm1.ops, ctx);

  auto pad2   = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(1, relu1, ctx);
  auto exch2  = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(1, pad2, ctx);

  Conv2DShape conv2_shape = shape;
  conv2_shape.halo_padded_input = true;

  auto conv2_init = kaiming_init_gen<GPUArray4D<float>>(shape.filters * shape.kernel.at(0) * shape.kernel.at(1), shape.filters);
  auto conv2  = parallel_conv2d<GPUBatchArray3D<float>>(conv2_shape, exch2, conv2_init, false, ctx);
  parallel_vars_insert_all(params, conv2.params);
  parallel_vars_insert_all(grads, conv2.grads);

  vector<vector<shared_ptr<Var<GPUBatchArray3D<float>>>>> join_inputs;
  join_inputs.push_back(inputs);
  join_inputs.push_back(conv2.ops);
  auto join   = parallel_sum<GPUBatchArray3D<float>>(join_inputs, ctx);

  assert(6 == params.at(0).size());
  assert(6 == grads.at(0).size());
  assert(4 == batch_stats.at(0).size());
  assert(4 == fixed_stats.at(0).size());

  ParallelBatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .ops = join,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
ParallelBatchNormParamOp<GPUBatchArray3D<float>> spatial_parallel_proj_residual_v2_conv2d(Conv2DShape shape, size_t stride, shared_ptr<bool> switch_var, vector<shared_ptr<Var<GPUBatchArray3D<float>>>> inputs, shared_ptr<Context> ctx) {
  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
  size_t num_ranks = mgpu_ctx->num_ranks();

  vector<VarIdSet> params(num_ranks);
  vector<VarIdSet> grads(num_ranks);
  vector<VarIdSet> batch_stats(num_ranks);
  vector<VarIdSet> fixed_stats(num_ranks);

  std::clog << "DEBUG: building spatial parallel proj residual v2 conv2d..." << std::endl;
  assert(0 == params.at(0).size());
  assert(0 == grads.at(0).size());
  assert(0 == batch_stats.at(0).size());
  assert(0 == fixed_stats.at(0).size());

  auto bnorm0 = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(inputs.at(0)->shape().shape.at(2), switch_var, inputs, ctx);
  parallel_vars_insert_all(params, bnorm0.params);
  parallel_vars_insert_all(grads, bnorm0.grads);
  parallel_vars_insert_all(batch_stats, bnorm0.batch_stats);
  parallel_vars_insert_all(fixed_stats, bnorm0.fixed_stats);

  auto relu0  = parallel_relu<GPUBatchArray3D<float>>(bnorm0.ops, ctx);

  Conv2DShape proj_shape = shape;
  proj_shape.kernel = {1, 1};
  proj_shape.stride = {stride, stride};
  proj_shape.pad = {0, 0};
  proj_shape.halo_padded_input = false;

  auto proj_init = kaiming_init_gen<GPUArray4D<float>>(inputs.at(0)->shape().shape.at(2), shape.filters);
  auto proj   = parallel_conv2d<GPUBatchArray3D<float>>(proj_shape, relu0, proj_init, false, ctx);
  parallel_vars_insert_all(params, proj.params);
  parallel_vars_insert_all(grads, proj.grads);

  auto pad1   = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(1, relu0, ctx);
  auto exch1  = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(1, pad1, ctx);

  Conv2DShape conv1_shape = shape;
  conv1_shape.stride = {stride, stride};
  conv1_shape.halo_padded_input = true;

  auto conv1_init = kaiming_init_gen<GPUArray4D<float>>(inputs.at(0)->shape().shape.at(2) * shape.kernel.at(0) * shape.kernel.at(1), shape.filters);
  auto conv1  = parallel_conv2d<GPUBatchArray3D<float>>(conv1_shape, exch1, conv1_init, false, ctx);
  parallel_vars_insert_all(params, conv1.params);
  parallel_vars_insert_all(grads, conv1.grads);

  auto bnorm1 = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(shape.filters, switch_var, conv1.ops, ctx);
  parallel_vars_insert_all(params, bnorm1.params);
  parallel_vars_insert_all(grads, bnorm1.grads);
  parallel_vars_insert_all(batch_stats, bnorm1.batch_stats);
  parallel_vars_insert_all(fixed_stats, bnorm1.fixed_stats);

  auto relu1  = parallel_relu<GPUBatchArray3D<float>>(bnorm1.ops, ctx);

  auto pad2   = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(1, relu1, ctx);
  auto exch2  = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(1, pad2, ctx);

  Conv2DShape conv2_shape = shape;
  conv2_shape.halo_padded_input = true;

  auto conv2_init = kaiming_init_gen<GPUArray4D<float>>(shape.filters * shape.kernel.at(0) * shape.kernel.at(1), shape.filters);
  auto conv2  = parallel_conv2d<GPUBatchArray3D<float>>(conv2_shape, exch2, conv2_init, false, ctx);
  parallel_vars_insert_all(params, conv2.params);
  parallel_vars_insert_all(grads, conv2.grads);

  vector<vector<shared_ptr<Var<GPUBatchArray3D<float>>>>> join_inputs;
  join_inputs.push_back(proj.ops);
  join_inputs.push_back(conv2.ops);
  auto join   = parallel_sum<GPUBatchArray3D<float>>(join_inputs, ctx);

  assert(7 == params.at(0).size());
  assert(7 == grads.at(0).size());
  assert(4 == batch_stats.at(0).size());
  assert(4 == fixed_stats.at(0).size());

  ParallelBatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .ops = join,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
ParallelBatchNormParamOp<GPUBatchArray3D<float>> spatial_parallel_bottleneck_residual_v2_conv2d(Conv2DShape shape, size_t bottleneck, shared_ptr<bool> switch_var, vector<shared_ptr<Var<GPUBatchArray3D<float>>>> inputs, shared_ptr<Context> ctx) {
  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
  size_t num_ranks = mgpu_ctx->num_ranks();

  vector<VarIdSet> params(num_ranks);
  vector<VarIdSet> grads(num_ranks);
  vector<VarIdSet> batch_stats(num_ranks);
  vector<VarIdSet> fixed_stats(num_ranks);

  std::clog << "DEBUG: building spatial parallel bottleneck residual v2 conv2d..." << std::endl;

  assert(inputs.at(0)->shape().shape.at(0) == bottleneck * shape.filters);

  assert(0 == params.at(0).size());
  assert(0 == grads.at(0).size());
  assert(0 == batch_stats.at(0).size());
  assert(0 == fixed_stats.at(0).size());

  auto bnorm0 = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(inputs.at(0)->shape().shape.at(2), switch_var, inputs, ctx);
  parallel_vars_insert_all(params, bnorm0.params);
  parallel_vars_insert_all(grads, bnorm0.grads);
  parallel_vars_insert_all(batch_stats, bnorm0.batch_stats);
  parallel_vars_insert_all(fixed_stats, bnorm0.fixed_stats);

  auto relu0  = parallel_relu<GPUBatchArray3D<float>>(bnorm0.ops, ctx);

  Conv2DShape conv1_shape = shape;
  conv1_shape.kernel = {1, 1};
  conv1_shape.stride = {1, 1};
  conv1_shape.pad = {0, 0};
  conv1_shape.halo_padded_input = false;

  auto conv1_init = kaiming_init_gen<GPUArray4D<float>>(inputs.at(0)->shape().shape.at(2), conv1_shape.filters);
  auto conv1  = parallel_conv2d<GPUBatchArray3D<float>>(conv1_shape, relu0, conv1_init, false, ctx);
  parallel_vars_insert_all(params, conv1.params);
  parallel_vars_insert_all(grads, conv1.grads);

  auto bnorm1 = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(conv1_shape.filters, switch_var, conv1.ops, ctx);
  parallel_vars_insert_all(params, bnorm1.params);
  parallel_vars_insert_all(grads, bnorm1.grads);
  parallel_vars_insert_all(batch_stats, bnorm1.batch_stats);
  parallel_vars_insert_all(fixed_stats, bnorm1.fixed_stats);

  auto relu1  = parallel_relu<GPUBatchArray3D<float>>(bnorm1.ops, ctx);

  Conv2DShape conv2_shape = shape;
  conv2_shape.stride = {1, 1};
  conv2_shape.halo_padded_input = true;
  assert(conv2_shape.pad.at(0) == conv2_shape.pad.at(1));

  auto pad2   = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(conv2_shape.pad.at(0), relu1, ctx);
  auto exch2  = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(conv2_shape.pad.at(0), pad2, ctx);

  auto conv2_init = kaiming_init_gen<GPUArray4D<float>>(conv1_shape.filters * conv2_shape.kernel.at(0) * conv2_shape.kernel.at(1), conv2_shape.filters);
  auto conv2  = parallel_conv2d<GPUBatchArray3D<float>>(conv2_shape, exch2, conv2_init, false, ctx);
  parallel_vars_insert_all(params, conv2.params);
  parallel_vars_insert_all(grads, conv2.grads);

  auto bnorm2 = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(conv2_shape.filters, switch_var, conv2.ops, ctx);
  parallel_vars_insert_all(params, bnorm2.params);
  parallel_vars_insert_all(grads, bnorm2.grads);
  parallel_vars_insert_all(batch_stats, bnorm2.batch_stats);
  parallel_vars_insert_all(fixed_stats, bnorm2.fixed_stats);

  auto relu2  = parallel_relu<GPUBatchArray3D<float>>(bnorm2.ops, ctx);

  Conv2DShape conv3_shape = shape;
  conv3_shape.filters = bottleneck * shape.filters;
  conv3_shape.kernel = {1, 1};
  conv3_shape.stride = {1, 1};
  conv3_shape.pad = {0, 0};
  conv3_shape.halo_padded_input = false;

  auto conv3_init = kaiming_init_gen<GPUArray4D<float>>(conv2_shape.filters, conv3_shape.filters);
  auto conv3  = parallel_conv2d<GPUBatchArray3D<float>>(conv3_shape, relu2, conv3_init, false, ctx);
  parallel_vars_insert_all(params, conv3.params);
  parallel_vars_insert_all(grads, conv3.grads);

  vector<vector<shared_ptr<Var<GPUBatchArray3D<float>>>>> join_inputs;
  join_inputs.push_back(inputs);
  join_inputs.push_back(conv3.ops);
  auto join   = parallel_sum<GPUBatchArray3D<float>>(join_inputs, ctx);

  assert(9 == params.at(0).size());
  assert(9 == grads.at(0).size());
  assert(6 == batch_stats.at(0).size());
  assert(6 == fixed_stats.at(0).size());

  ParallelBatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .ops = join,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
ParallelBatchNormParamOp<GPUBatchArray3D<float>> spatial_parallel_proj_bottleneck_residual_v2_conv2d(Conv2DShape shape, size_t bottleneck, size_t stride, shared_ptr<bool> switch_var, vector<shared_ptr<Var<GPUBatchArray3D<float>>>> inputs, shared_ptr<Context> ctx) {
  MultiGPUContext* mgpu_ctx = dynamic_cast<MultiGPUContext*>(ctx.get());
  size_t num_ranks = mgpu_ctx->num_ranks();

  vector<VarIdSet> params(num_ranks);
  vector<VarIdSet> grads(num_ranks);
  vector<VarIdSet> batch_stats(num_ranks);
  vector<VarIdSet> fixed_stats(num_ranks);

  std::clog << "DEBUG: building spatial parallel bottleneck residual v2 conv2d..." << std::endl;

  assert(0 == params.at(0).size());
  assert(0 == grads.at(0).size());
  assert(0 == batch_stats.at(0).size());
  assert(0 == fixed_stats.at(0).size());

  auto bnorm0 = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(inputs.at(0)->shape().shape.at(2), switch_var, inputs, ctx);
  parallel_vars_insert_all(params, bnorm0.params);
  parallel_vars_insert_all(grads, bnorm0.grads);
  parallel_vars_insert_all(batch_stats, bnorm0.batch_stats);
  parallel_vars_insert_all(fixed_stats, bnorm0.fixed_stats);

  auto relu0  = parallel_relu<GPUBatchArray3D<float>>(bnorm0.ops, ctx);

  Conv2DShape proj_shape = shape;
  proj_shape.filters = bottleneck * shape.filters;
  proj_shape.kernel = {1, 1};
  proj_shape.stride = {stride, stride};
  proj_shape.pad = {0, 0};
  proj_shape.halo_padded_input = false;

  auto proj_init = kaiming_init_gen<GPUArray4D<float>>(inputs.at(0)->shape().shape.at(2), proj_shape.filters);
  auto proj   = parallel_conv2d<GPUBatchArray3D<float>>(proj_shape, relu0, proj_init, false, ctx);
  parallel_vars_insert_all(params, proj.params);
  parallel_vars_insert_all(grads, proj.grads);

  Conv2DShape conv1_shape = shape;
  conv1_shape.kernel = {1, 1};
  conv1_shape.stride = {1, 1};
  conv1_shape.pad = {0, 0};
  conv1_shape.halo_padded_input = false;

  auto conv1_init = kaiming_init_gen<GPUArray4D<float>>(inputs.at(0)->shape().shape.at(2), conv1_shape.filters);
  auto conv1  = parallel_conv2d<GPUBatchArray3D<float>>(conv1_shape, relu0, conv1_init, false, ctx);
  parallel_vars_insert_all(params, conv1.params);
  parallel_vars_insert_all(grads, conv1.grads);

  auto bnorm1 = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(conv1_shape.filters, switch_var, conv1.ops, ctx);
  parallel_vars_insert_all(params, bnorm1.params);
  parallel_vars_insert_all(grads, bnorm1.grads);
  parallel_vars_insert_all(batch_stats, bnorm1.batch_stats);
  parallel_vars_insert_all(fixed_stats, bnorm1.fixed_stats);

  auto relu1  = parallel_relu<GPUBatchArray3D<float>>(bnorm1.ops, ctx);

  Conv2DShape conv2_shape = shape;
  conv2_shape.stride = {1, 1};
  conv2_shape.halo_padded_input = true;
  assert(conv2_shape.pad.at(0) == conv2_shape.pad.at(1));

  auto pad2   = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(conv2_shape.pad.at(0), relu1, ctx);
  auto exch2  = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(conv2_shape.pad.at(0), pad2, ctx);

  auto conv2_init = kaiming_init_gen<GPUArray4D<float>>(conv1_shape.filters * conv2_shape.kernel.at(0) * conv2_shape.kernel.at(1), conv2_shape.filters);
  auto conv2  = parallel_conv2d<GPUBatchArray3D<float>>(conv2_shape, exch2, conv2_init, false, ctx);
  parallel_vars_insert_all(params, conv2.params);
  parallel_vars_insert_all(grads, conv2.grads);

  auto bnorm2 = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(conv2_shape.filters, switch_var, conv2.ops, ctx);
  parallel_vars_insert_all(params, bnorm2.params);
  parallel_vars_insert_all(grads, bnorm2.grads);
  parallel_vars_insert_all(batch_stats, bnorm2.batch_stats);
  parallel_vars_insert_all(fixed_stats, bnorm2.fixed_stats);

  auto relu2  = parallel_relu<GPUBatchArray3D<float>>(bnorm2.ops, ctx);

  Conv2DShape conv3_shape = shape;
  conv3_shape.filters = bottleneck * shape.filters;
  conv3_shape.kernel = {1, 1};
  conv3_shape.stride = {1, 1};
  conv3_shape.pad = {0, 0};
  conv3_shape.halo_padded_input = false;

  auto conv3_init = kaiming_init_gen<GPUArray4D<float>>(conv2_shape.filters, conv3_shape.filters);
  auto conv3  = parallel_conv2d<GPUBatchArray3D<float>>(conv3_shape, relu2, conv3_init, false, ctx);
  parallel_vars_insert_all(params, conv3.params);
  parallel_vars_insert_all(grads, conv3.grads);

  vector<vector<shared_ptr<Var<GPUBatchArray3D<float>>>>> join_inputs;
  join_inputs.push_back(proj.ops);
  join_inputs.push_back(conv3.ops);
  auto join   = parallel_sum<GPUBatchArray3D<float>>(join_inputs, ctx);

  assert(10 == params.at(0).size());
  assert(10 == grads.at(0).size());
  assert(6 == batch_stats.at(0).size());
  assert(6 == fixed_stats.at(0).size());

  ParallelBatchNormParamOp<GPUBatchArray3D<float>> param_op = {
    .ops = join,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return param_op;
}

template <>
ForwardOp<GPUBatchArray3D<float>> ImagenetPreproc<float>::Make(size_t input_w, size_t input_h, size_t batch_size, shared_ptr<Context> ctx) {
  Shape3D raw_input_dim(3UL, input_w, input_h);
  auto raw_input = batch_array_src<GPUBatchArray3D<uint8_t>, Shape3D>(raw_input_dim, batch_size, ctx);
  auto inputs = VarIdSet::Singleton(raw_input->vars().at(0));
  auto cast_input = make_shared<CastOp<GPUBatchArray3D<uint8_t>, GPUBatchArray3D<float>>>(raw_input, ctx);
  auto transpose_input = make_shared<TransposeNHWCToNCHWOp<GPUBatchArray3D<float>, Shape3D>>(cast_input, ctx);
  shared_ptr<Var<GPUBatchArray3D<float>>> y = transpose_input;
  Conv2DShape conv1_shape = {
    .filters = 64,
    .axes = {0, 1},
    .kernel = {7, 7},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {3, 3},
  };
  //auto conv1 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, y, ctx);
  //y = conv1.op;
  ForwardOp<GPUBatchArray3D<float>> fwd = {
    .op = y,
    .inputs = inputs,
  };
  return fwd;
}

template <>
CategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> ResNet<float>::MakeTestNet(const ConvNetConfig& cfg, shared_ptr<Context> ctx) {
  std::clog << "DEBUG: making testnet..." << std::endl;

  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  std::clog << "DEBUG: make batch norm switch var..." << std::endl;
  //auto switch_var = val_src<bool>(false, ctx);
  auto switch_var = make_shared<bool>(false);

  std::clog << "DEBUG: make input preproc..." << std::endl;

  Shape3D raw_input_dim(3UL, cfg.input_w, cfg.input_h);
  Shape3D transpose_input_dim(cfg.input_w, cfg.input_h, 3UL);

  auto raw_input = batch_array_src<GPUBatchArray3D<uint8_t>, Shape3D>(raw_input_dim, cfg.batch_size, ctx);
  VarIdSet inputs = VarIdSet::Singleton(raw_input->vars().at(0));

  auto label = batch_array_src<GPUBatchArray0D<uint32_t>, Shape0D>(Shape0D(), cfg.batch_size, ctx);
  VarIdSet labels = VarIdSet::Singleton(label->vars().at(0));

  shared_ptr<Var<GPUBatchArray3D<float>>> cast_input = make_shared<CastOp<GPUBatchArray3D<uint8_t>, GPUBatchArray3D<float>>>(raw_input, ctx);
  auto cast_input_io = io_var(cast_input, ctx);

  shared_ptr<Var<GPUBatchArray3D<float>>> scale_input = make_shared<ScaleOp<float, GPUBatchArray3D<float>>>(1.0f / 255.0f, cast_input_io, ctx);
  auto scale_input_io = io_var(scale_input, ctx);

  shared_ptr<Var<GPUBatchArray3D<float>>> transpose_input = make_shared<TransposeNHWCToNCHWOp<GPUBatchArray3D<float>, Shape3D>>(scale_input_io, ctx);
  auto transpose_input_io = io_var(transpose_input, ctx);

  //VarIdSet _preproc_inputs = VarIdSet::Singleton(cast_input_io->vars().at(0));
  //VarIdSet _preproc_inputs = VarIdSet::Singleton(scale_input_io->vars().at(0));
  VarIdSet _preproc_inputs = VarIdSet::Singleton(transpose_input_io->vars().at(0));

  std::clog << "DEBUG: make conv1..." << std::endl;

  shared_ptr<Var<GPUBatchArray3D<float>>> y = transpose_input_io;

  // FIXME
  //assert(0 && "unimplemented");

  Conv2DShape conv1_shape = {
    .filters = 64,
    .axes = {0, 1},
    .kernel = {11, 11},
    .dilate = {1, 1},
    .stride = {2, 2},
    .pad = {5, 5},
  };

  auto conv1 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, y, xavier_init_gen<GPUArray4D<float>>(3UL * conv1_shape.kernel.at(0) * conv1_shape.kernel.at(1), conv1_shape.filters), true, ctx);
  params.insert_all(conv1.params);
  grads.insert_all(conv1.grads);
  //batch_stats.insert_all(conv1.batch_stats);
  //fixed_stats.insert_all(conv1.fixed_stats);
  y = io_var(conv1.op, ctx);

  //shared_ptr<Var<GPUBatchArray3D<float>>> target = y;
  //VarIdSet _hidden_convs = VarIdSet::Singleton(y->vars().at(0));

  /*std::clog << "DEBUG: make pool1..." << std::endl;

  Pool2DShape pool1_shape = {
    .pooling = kMaxPool,
    .axes = {0, 1},
    .window = {3, 3},
    .stride = {2, 2},
    .pad = {1, 1},
  };
  y = make_shared<CudnnPool2DOp<GPUBatchArray3D<float>>>(pool1_shape, y, ctx);*/

  Conv2DShape res2_shape = {
    .filters = 128,
    .axes = {0, 1},
    .kernel = {3, 3},
    //.kernel = {5, 5},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    //.pad = {2, 2},
  };

  std::clog << "DEBUG: make res2_1..." << std::endl;

  auto res2_1 = proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, 1, switch_var, y, ctx);
  params.insert_all(res2_1.params);
  grads.insert_all(res2_1.grads);
  batch_stats.insert_all(res2_1.batch_stats);
  fixed_stats.insert_all(res2_1.fixed_stats);
  y = io_var(res2_1.op, ctx);

  std::clog << "DEBUG: make res2_2..." << std::endl;

  auto res2_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, switch_var, y, ctx);
  params.insert_all(res2_2.params);
  grads.insert_all(res2_2.grads);
  batch_stats.insert_all(res2_2.batch_stats);
  fixed_stats.insert_all(res2_2.fixed_stats);
  y = res2_2.op;

  Conv2DShape res3_shape = {
    .filters = 256,
    .axes = {0, 1},
    .kernel = {3, 3},
    //.kernel = {5, 5},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    //.pad = {2, 2},
  };

  std::clog << "DEBUG: make res3_1..." << std::endl;

  auto res3_1 = proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, 2, switch_var, y, ctx);
  params.insert_all(res3_1.params);
  grads.insert_all(res3_1.grads);
  batch_stats.insert_all(res3_1.batch_stats);
  fixed_stats.insert_all(res3_1.fixed_stats);
  y = res3_1.op;

  std::clog << "DEBUG: make res3_2..." << std::endl;

  auto res3_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, switch_var, y, ctx);
  params.insert_all(res3_2.params);
  grads.insert_all(res3_2.grads);
  batch_stats.insert_all(res3_2.batch_stats);
  fixed_stats.insert_all(res3_2.fixed_stats);
  y = res3_2.op;

  Conv2DShape res4_shape = {
    .filters = 512,
    .axes = {0, 1},
    .kernel = {3, 3},
    //.kernel = {5, 5},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    //.pad = {2, 2},
  };

  std::clog << "DEBUG: make res4_1..." << std::endl;

  auto res4_1 = proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 2, switch_var, y, ctx);
  params.insert_all(res4_1.params);
  grads.insert_all(res4_1.grads);
  batch_stats.insert_all(res4_1.batch_stats);
  fixed_stats.insert_all(res4_1.fixed_stats);
  y = res4_1.op;

  std::clog << "DEBUG: make res4_2..." << std::endl;

  auto res4_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, switch_var, y, ctx);
  params.insert_all(res4_2.params);
  grads.insert_all(res4_2.grads);
  batch_stats.insert_all(res4_2.batch_stats);
  fixed_stats.insert_all(res4_2.fixed_stats);
  y = res4_2.op;

  Conv2DShape res5_shape = {
    .filters = 512,
    .axes = {0, 1},
    .kernel = {3, 3},
    //.kernel = {5, 5},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    //.pad = {2, 2},
  };

  std::clog << "DEBUG: make res5_1..." << std::endl;

  auto res5_1 = proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, 2, switch_var, y, ctx);
  params.insert_all(res5_1.params);
  grads.insert_all(res5_1.grads);
  batch_stats.insert_all(res5_1.batch_stats);
  fixed_stats.insert_all(res5_1.fixed_stats);
  y = res5_1.op;

  std::clog << "DEBUG: make res5_2..." << std::endl;

  auto res5_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, switch_var, y, ctx);
  params.insert_all(res5_2.params);
  grads.insert_all(res5_2.grads);
  batch_stats.insert_all(res5_2.batch_stats);
  fixed_stats.insert_all(res5_2.fixed_stats);
  y = res5_2.op;

  std::clog << "DEBUG: make global pool..." << std::endl;

  auto bnorm_final = batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(512, switch_var, y, ctx);
  params.insert_all(bnorm_final.params);
  grads.insert_all(bnorm_final.grads);
  batch_stats.insert_all(bnorm_final.batch_stats);
  fixed_stats.insert_all(bnorm_final.fixed_stats);
  y = bnorm_final.op;

  y = relu<GPUBatchArray3D<float>>(y, ctx);
  //y = relu_in_place<GPUBatchArray3D<float>>(y, ctx);

  Pool2DShape global_pool_shape = {
    .pooling = kAveragePool,
    .axes = {0, 1},
    .window = {14, 14},
    .stride = {14, 14},
    //.window = {28, 28},
    //.stride = {28, 28},
    .pad = {0, 0},
  };
  y = make_shared<CudnnPool2DOp<GPUBatchArray3D<float>>>(global_pool_shape, y, ctx);

  y = io_var(y, ctx);

  shared_ptr<Var<GPUBatchArray3D<float>>> target = y;
  VarIdSet _hidden_convs = VarIdSet::Singleton(y->vars().at(0));

  std::clog << "DEBUG: package loss..." << std::endl;

  shared_ptr<Var<GPUBatchArray1D<float>>> z = flatten_nchw<GPUBatchArray3D<float>, GPUBatchArray1D<float>>(y, ctx);

  //z = io_var(z, ctx);
  //shared_ptr<Var<GPUBatchArray1D<float>>> full_target = z;
  //VarIdSet _hidden_fcs = VarIdSet::Singleton(z->vars().at(0));

  auto fc1 = linear<GPUBatchArray1D<float>, GPUArray2D<float>>(1000, z, xavier_init_gen<GPUArray2D<float>>(512, 1000), ctx);
  params.insert_all(fc1.params);
  grads.insert_all(fc1.grads);
  z = io_var(fc1.op, ctx);
  VarIdSet logits = VarIdSet::Singleton(z->vars().at(0));

  shared_ptr<Var<GPUBatchArray1D<float>>> full_target = z;
  VarIdSet _hidden_fcs = VarIdSet::Singleton(z->vars().at(0));

  auto nll_loss = io_var(softmax_nll_loss<GPUBatchArray1D<float>, GPUBatchArray0D<uint32_t>, GPUBatchArray0D<float>>(z, label, ctx), ctx);
  VarIdSet losses = VarIdSet::Singleton(nll_loss->vars().at(0));

  auto sum_loss = batch_sum<GPUBatchArray0D<float>, GPUArray0D<float>>(nll_loss, ctx);

  CategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> loss = {
    .batch_norm_use_fixed = switch_var,
    // FIXME
    //.op = target,
    //.op = full_target,
    .op = z,
    .loss = nll_loss,
    .sum_loss = sum_loss,
    .inputs = inputs,
    .labels = labels,
    .logits = logits,
    .losses = losses,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
    ._preproc_inputs = _preproc_inputs,
    ._hidden_convs = _hidden_convs,
    ._hidden_fcs = _hidden_fcs,
  };
  return loss;
}

template <>
CategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> ResNet<float>::MakeResNet18(const ConvNetConfig& cfg, shared_ptr<Context> ctx) {
  std::clog << "DEBUG: making resnet18..." << std::endl;

  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  std::clog << "DEBUG: make batch norm switch var..." << std::endl;
  //auto switch_var = val_src<bool>(false, ctx);
  auto switch_var = make_shared<bool>(false);

  std::clog << "DEBUG: make input preproc..." << std::endl;

  Shape3D raw_input_dim(3UL, cfg.input_w, cfg.input_h);
  Shape3D transpose_input_dim(cfg.input_w, cfg.input_h, 3UL);

  auto raw_input = batch_array_src<GPUBatchArray3D<float>, Shape3D>(raw_input_dim, cfg.batch_size, ctx);
  VarIdSet inputs = VarIdSet::Singleton(raw_input->vars().at(0));

  auto label = batch_array_src<GPUBatchArray0D<uint32_t>, Shape0D>(Shape0D(), cfg.batch_size, ctx);
  VarIdSet labels = VarIdSet::Singleton(label->vars().at(0));

  //auto cast_input = make_shared<CastOp<GPUBatchArray3D<uint8_t>, GPUBatchArray3D<float>>>(raw_input, ctx);
  auto transpose_input = make_shared<TransposeNHWCToNCHWOp<GPUBatchArray3D<float>, Shape3D>>(raw_input, ctx);
  auto scale_input = make_shared<ScaleOp<float, GPUBatchArray3D<float>>>(1.0f / 255.0f, transpose_input, ctx);

  std::clog << "DEBUG: make conv1..." << std::endl;

  shared_ptr<Var<GPUBatchArray3D<float>>> y = transpose_input;

  //y = make_shared<HaloPad2DOp<GPUBatchArray3D<float>>>(3, y, ctx);
  //y = make_shared<HaloUnpad2DOp<GPUBatchArray3D<float>>>(3, y, ctx);

  Conv2DShape conv1_shape = {
    .filters = 64,
    .axes = {0, 1},
    .kernel = {7, 7},
    .dilate = {1, 1},
    .stride = {2, 2},
    .pad = {3, 3},
  };

  auto conv1 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, y, xavier_init_gen<GPUArray4D<float>>(3UL * conv1_shape.kernel.at(0) * conv1_shape.kernel.at(1), conv1_shape.filters), true, ctx);
  params.insert_all(conv1.params);
  grads.insert_all(conv1.grads);
  //batch_stats.insert_all(conv1.batch_stats);
  //fixed_stats.insert_all(conv1.fixed_stats);
  y = conv1.op;

  //y = make_shared<HaloPad2DOp<GPUBatchArray3D<float>>>(1, y, ctx);
  //y = make_shared<HaloUnpad2DOp<GPUBatchArray3D<float>>>(1, y, ctx);

  std::clog << "DEBUG: make pool1..." << std::endl;

  Pool2DShape pool1_shape = {
    .pooling = kMaxPool,
    .axes = {0, 1},
    .window = {3, 3},
    .stride = {2, 2},
    .pad = {1, 1},
  };
  y = make_shared<CudnnPool2DOp<GPUBatchArray3D<float>>>(pool1_shape, y, ctx);

  Conv2DShape res2_shape = {
    .filters = 64,
    .axes = {0, 1},
    .kernel = {3, 3},
    //.kernel = {5, 5},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    //.pad = {2, 2},
  };

  std::clog << "DEBUG: make res2_1..." << std::endl;

  auto res2_1 = residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, switch_var, y, ctx);
  params.insert_all(res2_1.params);
  grads.insert_all(res2_1.grads);
  batch_stats.insert_all(res2_1.batch_stats);
  fixed_stats.insert_all(res2_1.fixed_stats);
  y = res2_1.op;

  std::clog << "DEBUG: make res2_2..." << std::endl;

  auto res2_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, switch_var, y, ctx);
  params.insert_all(res2_2.params);
  grads.insert_all(res2_2.grads);
  batch_stats.insert_all(res2_2.batch_stats);
  fixed_stats.insert_all(res2_2.fixed_stats);
  y = res2_2.op;

  Conv2DShape res3_shape = {
    .filters = 128,
    .axes = {0, 1},
    .kernel = {3, 3},
    //.kernel = {5, 5},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    //.pad = {2, 2},
  };

  std::clog << "DEBUG: make res3_1..." << std::endl;

  auto res3_1 = proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, 2, switch_var, y, ctx);
  params.insert_all(res3_1.params);
  grads.insert_all(res3_1.grads);
  batch_stats.insert_all(res3_1.batch_stats);
  fixed_stats.insert_all(res3_1.fixed_stats);
  y = res3_1.op;

  std::clog << "DEBUG: make res3_2..." << std::endl;

  auto res3_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, switch_var, y, ctx);
  params.insert_all(res3_2.params);
  grads.insert_all(res3_2.grads);
  batch_stats.insert_all(res3_2.batch_stats);
  fixed_stats.insert_all(res3_2.fixed_stats);
  y = res3_2.op;

  Conv2DShape res4_shape = {
    .filters = 256,
    .axes = {0, 1},
    .kernel = {3, 3},
    //.kernel = {5, 5},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    //.pad = {2, 2},
  };

  std::clog << "DEBUG: make res4_1..." << std::endl;

  auto res4_1 = proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 2, switch_var, y, ctx);
  params.insert_all(res4_1.params);
  grads.insert_all(res4_1.grads);
  batch_stats.insert_all(res4_1.batch_stats);
  fixed_stats.insert_all(res4_1.fixed_stats);
  y = res4_1.op;

  std::clog << "DEBUG: make res4_2..." << std::endl;

  auto res4_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, switch_var, y, ctx);
  params.insert_all(res4_2.params);
  grads.insert_all(res4_2.grads);
  batch_stats.insert_all(res4_2.batch_stats);
  fixed_stats.insert_all(res4_2.fixed_stats);
  y = res4_2.op;

  Conv2DShape res5_shape = {
    .filters = 512,
    .axes = {0, 1},
    .kernel = {3, 3},
    //.kernel = {5, 5},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    //.pad = {2, 2},
  };

  std::clog << "DEBUG: make res5_1..." << std::endl;

  auto res5_1 = proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, 2, switch_var, y, ctx);
  params.insert_all(res5_1.params);
  grads.insert_all(res5_1.grads);
  batch_stats.insert_all(res5_1.batch_stats);
  fixed_stats.insert_all(res5_1.fixed_stats);
  y = res5_1.op;

  std::clog << "DEBUG: make res5_2..." << std::endl;

  auto res5_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, switch_var, y, ctx);
  params.insert_all(res5_2.params);
  grads.insert_all(res5_2.grads);
  batch_stats.insert_all(res5_2.batch_stats);
  fixed_stats.insert_all(res5_2.fixed_stats);
  y = res5_2.op;

  std::clog << "DEBUG: make global pool..." << std::endl;

  auto bnorm_final = batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(512, switch_var, y, ctx);
  params.insert_all(bnorm_final.params);
  grads.insert_all(bnorm_final.grads);
  batch_stats.insert_all(bnorm_final.batch_stats);
  fixed_stats.insert_all(bnorm_final.fixed_stats);
  y = bnorm_final.op;

  y = relu<GPUBatchArray3D<float>>(y, ctx);
  //y = relu_in_place<GPUBatchArray3D<float>>(y, ctx);

  Pool2DShape global_pool_shape = {
    .pooling = kAveragePool,
    .axes = {0, 1},
    .window = {7, 7},
    .stride = {7, 7},
    //.window = {14, 14},
    //.stride = {14, 14},
    //.window = {28, 28},
    //.stride = {28, 28},
    .pad = {0, 0},
  };
  y = make_shared<CudnnPool2DOp<GPUBatchArray3D<float>>>(global_pool_shape, y, ctx);

  std::clog << "DEBUG: package loss..." << std::endl;

  shared_ptr<Var<GPUBatchArray1D<float>>> z = flatten_nchw<GPUBatchArray3D<float>, GPUBatchArray1D<float>>(y, ctx);

  auto fc1 = linear<GPUBatchArray1D<float>, GPUArray2D<float>>(1000, z, xavier_init_gen<GPUArray2D<float>>(512, 1000), ctx);
  params.insert_all(fc1.params);
  grads.insert_all(fc1.grads);
  z = io_var(fc1.op, ctx);
  VarIdSet logits = VarIdSet::Singleton(z->vars().at(0));

  auto nll_loss = io_var(softmax_nll_loss<GPUBatchArray1D<float>, GPUBatchArray0D<uint32_t>, GPUBatchArray0D<float>>(z, label, ctx), ctx);
  VarIdSet losses = VarIdSet::Singleton(nll_loss->vars().at(0));

  auto nll_loss_scaled = ScaleOp<float, GPUBatchArray0D<float>>::Make(1.0f / static_cast<float>(cfg.minibatch_size), nll_loss, ctx);
  auto sum_loss = batch_sum<GPUBatchArray0D<float>, GPUArray0D<float>>(nll_loss_scaled, ctx);

  CategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> loss = {
    .batch_norm_use_fixed = switch_var,
    .op = z,
    .loss = nll_loss,
    .sum_loss = sum_loss,
    .inputs = inputs,
    .labels = labels,
    .logits = logits,
    .losses = losses,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return loss;
}

template <>
ParallelCategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> ResNet<float>::MakeParallelResNet18(const ConvNetConfig& cfg, shared_ptr<Context> parallel_ctx) {
  std::clog << "DEBUG: making parallel resnet18..." << std::endl;

  const bool enable_hmma = false;

  size_t num_ranks = dynamic_cast<MultiGPUContext*>(parallel_ctx.get())->num_ranks();
  shared_ptr<Context> root_dev_ctx = dynamic_cast<MultiGPUContext*>(parallel_ctx.get())->device_context(0);
  shared_ptr<Context> ctx = root_dev_ctx;

  /*VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;*/

  vector<VarIdSet> shared_params(num_ranks);
  vector<VarIdSet> shared_grads(num_ranks);
  vector<VarIdSet> dev_params(num_ranks);
  vector<VarIdSet> dev_grads(num_ranks);
  vector<VarIdSet> dev_batch_stats(num_ranks);
  vector<VarIdSet> dev_fixed_stats(num_ranks);

  std::clog << "DEBUG: make batch norm switch var..." << std::endl;
  auto switch_var = make_shared<bool>(false);

  std::clog << "DEBUG: make input preproc..." << std::endl;

  Shape3D raw_input_dim(3UL, cfg.input_w, cfg.input_h);
  Shape3D transpose_input_dim(cfg.input_w, cfg.input_h, 3UL);

  //auto raw_input = batch_array_src<GPUBatchArray3D<uint8_t>, Shape3D>(raw_input_dim, cfg.batch_size, root_dev_ctx);
  auto raw_input = batch_array_src<GPUBatchArray3D<float>, Shape3D>(raw_input_dim, cfg.batch_size, root_dev_ctx);
  VarIdSet inputs = VarIdSet::Singleton(raw_input->vars().at(0));

  auto label = batch_array_src<GPUBatchArray0D<uint32_t>, Shape0D>(Shape0D(), cfg.batch_size, root_dev_ctx);
  VarIdSet labels = VarIdSet::Singleton(label->vars().at(0));

  //auto cast_input = CastOp<GPUBatchArray3D<uint8_t>, GPUBatchArray3D<float>>::Make(raw_input, root_dev_ctx);
  //auto transpose_input = TransposeNHWCToNCHWOp<GPUBatchArray3D<float>, Shape3D>::Make(cast_input, root_dev_ctx);
  auto transpose_input = TransposeNHWCToNCHWOp<GPUBatchArray3D<float>, Shape3D>::Make(raw_input, root_dev_ctx);
  auto scale_input = ScaleOp<float, GPUBatchArray3D<float>>::Make(1.0f / 255.0f, transpose_input, root_dev_ctx);

  std::clog << "DEBUG: make conv1..." << std::endl;

  vector<shared_ptr<Var<GPUBatchArray3D<float>>>> par_ys;
  shared_ptr<Var<GPUBatchArray3D<float>>> y = scale_input;

  par_ys = BroadcastOp<GPUBatchArray3D<float>>::MakeParallel(0, y, parallel_ctx);
  par_ys = SpatialScatter2DOp<GPUBatchArray3D<float>>::MakeParallel(par_ys, parallel_ctx);

  par_ys = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(3, par_ys, parallel_ctx);
  par_ys = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(3, par_ys, parallel_ctx);

  Conv2DShape conv1_shape = {
    .filters = 64,
    .axes = {0, 1},
    .kernel = {7, 7},
    .dilate = {1, 1},
    .stride = {1, 1},
    //.stride = {2, 2},
    .pad = {3, 3},
    .halo_padded_input = true,
    .enable_hmma = enable_hmma,
  };

  // TODO
  //auto conv1 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, y, xavier_init_gen<GPUArray4D<float>>(3UL * conv1_shape.kernel.at(0) * conv1_shape.kernel.at(1), conv1_shape.filters), ctx);
  //params.insert_all(conv1.params);
  //grads.insert_all(conv1.grads);
  //y = conv1.op;
  //par_ys.clear();
  //par_ys.push_back(y);

  auto conv1 = parallel_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, par_ys, xavier_init_gen<GPUArray4D<float>>(3UL * conv1_shape.kernel.at(0) * conv1_shape.kernel.at(1), conv1_shape.filters), true, parallel_ctx);
  /*params.insert_all(conv1.params.at(0));
  grads.insert_all(conv1.grads.at(0));*/
  parallel_vars_insert_all(dev_params, conv1.params);
  parallel_vars_insert_all(dev_grads, conv1.grads);
  par_ys = conv1.ops;
  //y = par_ys.at(0);

  /*par_ys = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(1, par_ys, parallel_ctx);
  par_ys = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(1, par_ys, parallel_ctx);
  par_ys = HaloUnpad2DOp<GPUBatchArray3D<float>>::MakeParallel(1, par_ys, parallel_ctx);*/

  std::clog << "DEBUG: make pool1..." << std::endl;

  par_ys = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(1, par_ys, parallel_ctx);
  par_ys = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(1, par_ys, parallel_ctx);

  Pool2DShape pool1_shape = {
    .pooling = kMaxPool,
    .axes = {0, 1},
    .window = {3, 3},
    .stride = {2, 2},
    .pad = {1, 1},
    .halo_padded_input = true,
  };
  par_ys = CudnnPool2DOp<GPUBatchArray3D<float>>::MakeParallel(pool1_shape, par_ys, parallel_ctx);
  //y = par_ys.at(0);

  Conv2DShape res2_shape = {
    .filters = 64,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    .halo_padded_input = true,
    .enable_hmma = enable_hmma,
  };

  std::clog << "DEBUG: make res2_1..." << std::endl;

  /*auto res2_1 = residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, switch_var, y, ctx);
  params.insert_all(res2_1.params);
  grads.insert_all(res2_1.grads);
  batch_stats.insert_all(res2_1.batch_stats);
  fixed_stats.insert_all(res2_1.fixed_stats);
  y = res2_1.op;*/
  auto res2_1 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, switch_var, par_ys, parallel_ctx);
  /*params.insert_all(res2_1.params.at(0));
  grads.insert_all(res2_1.grads.at(0));
  batch_stats.insert_all(res2_1.batch_stats.at(0));
  fixed_stats.insert_all(res2_1.fixed_stats.at(0));*/
  parallel_vars_insert_all(dev_params, res2_1.params);
  parallel_vars_insert_all(dev_grads, res2_1.grads);
  parallel_vars_insert_all(dev_batch_stats, res2_1.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res2_1.fixed_stats);
  par_ys = res2_1.ops;
  //y = par_ys.at(0);

  std::clog << "DEBUG: make res2_2..." << std::endl;

  /*auto res2_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, switch_var, y, ctx);
  params.insert_all(res2_2.params);
  grads.insert_all(res2_2.grads);
  batch_stats.insert_all(res2_2.batch_stats);
  fixed_stats.insert_all(res2_2.fixed_stats);
  y = res2_2.op;*/
  auto res2_2 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, switch_var, par_ys, parallel_ctx);
  /*params.insert_all(res2_2.params.at(0));
  grads.insert_all(res2_2.grads.at(0));
  batch_stats.insert_all(res2_2.batch_stats.at(0));
  fixed_stats.insert_all(res2_2.fixed_stats.at(0));*/
  parallel_vars_insert_all(dev_params, res2_2.params);
  parallel_vars_insert_all(dev_grads, res2_2.grads);
  parallel_vars_insert_all(dev_batch_stats, res2_2.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res2_2.fixed_stats);
  par_ys = res2_2.ops;
  //y = par_ys.at(0);

  Conv2DShape res3_shape = {
    .filters = 128,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    .halo_padded_input = true,
    .enable_hmma = enable_hmma,
  };

  std::clog << "DEBUG: make res3_1..." << std::endl;

  /*auto res3_1 = proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, 2, switch_var, y, ctx);
  params.insert_all(res3_1.params);
  grads.insert_all(res3_1.grads);
  batch_stats.insert_all(res3_1.batch_stats);
  fixed_stats.insert_all(res3_1.fixed_stats);
  y = res3_1.op;*/
  auto res3_1 = spatial_parallel_proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, 2, switch_var, par_ys, parallel_ctx);
  /*params.insert_all(res3_1.params.at(0));
  grads.insert_all(res3_1.grads.at(0));
  batch_stats.insert_all(res3_1.batch_stats.at(0));
  fixed_stats.insert_all(res3_1.fixed_stats.at(0));*/
  parallel_vars_insert_all(dev_params, res3_1.params);
  parallel_vars_insert_all(dev_grads, res3_1.grads);
  parallel_vars_insert_all(dev_batch_stats, res3_1.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res3_1.fixed_stats);
  par_ys = res3_1.ops;
  //y = par_ys.at(0);

  std::clog << "DEBUG: make res3_2..." << std::endl;

  /*auto res3_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, switch_var, y, ctx);
  params.insert_all(res3_2.params);
  grads.insert_all(res3_2.grads);
  batch_stats.insert_all(res3_2.batch_stats);
  fixed_stats.insert_all(res3_2.fixed_stats);
  y = res3_2.op;*/
  auto res3_2 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, switch_var, par_ys, parallel_ctx);
  /*params.insert_all(res3_2.params.at(0));
  grads.insert_all(res3_2.grads.at(0));
  batch_stats.insert_all(res3_2.batch_stats.at(0));
  fixed_stats.insert_all(res3_2.fixed_stats.at(0));*/
  parallel_vars_insert_all(dev_params, res3_2.params);
  parallel_vars_insert_all(dev_grads, res3_2.grads);
  parallel_vars_insert_all(dev_batch_stats, res3_2.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res3_2.fixed_stats);
  par_ys = res3_2.ops;
  //y = par_ys.at(0);

  Conv2DShape res4_shape = {
    .filters = 256,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    .halo_padded_input = true,
    .enable_hmma = enable_hmma,
  };

  std::clog << "DEBUG: make res4_1..." << std::endl;

  /*auto res4_1 = proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 2, switch_var, y, ctx);
  params.insert_all(res4_1.params);
  grads.insert_all(res4_1.grads);
  batch_stats.insert_all(res4_1.batch_stats);
  fixed_stats.insert_all(res4_1.fixed_stats);
  y = res4_1.op;*/
  auto res4_1 = spatial_parallel_proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 2, switch_var, par_ys, parallel_ctx);
  /*params.insert_all(res4_1.params.at(0));
  grads.insert_all(res4_1.grads.at(0));
  batch_stats.insert_all(res4_1.batch_stats.at(0));
  fixed_stats.insert_all(res4_1.fixed_stats.at(0));*/
  parallel_vars_insert_all(dev_params, res4_1.params);
  parallel_vars_insert_all(dev_grads, res4_1.grads);
  parallel_vars_insert_all(dev_batch_stats, res4_1.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res4_1.fixed_stats);
  par_ys = res4_1.ops;
  //y = par_ys.at(0);

  std::clog << "DEBUG: make res4_2..." << std::endl;

  /*auto res4_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, switch_var, y, ctx);
  params.insert_all(res4_2.params);
  grads.insert_all(res4_2.grads);
  batch_stats.insert_all(res4_2.batch_stats);
  fixed_stats.insert_all(res4_2.fixed_stats);
  y = res4_2.op;*/
  auto res4_2 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, switch_var, par_ys, parallel_ctx);
  /*params.insert_all(res4_2.params.at(0));
  grads.insert_all(res4_2.grads.at(0));
  batch_stats.insert_all(res4_2.batch_stats.at(0));
  fixed_stats.insert_all(res4_2.fixed_stats.at(0));*/
  parallel_vars_insert_all(dev_params, res4_2.params);
  parallel_vars_insert_all(dev_grads, res4_2.grads);
  parallel_vars_insert_all(dev_batch_stats, res4_2.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res4_2.fixed_stats);
  par_ys = res4_2.ops;
  //y = par_ys.at(0);

  Conv2DShape res5_shape = {
    //.filters = 256,
    .filters = 512,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    .halo_padded_input = true,
    .enable_hmma = enable_hmma,
  };

  std::clog << "DEBUG: make res5_1..." << std::endl;

  /*auto res5_1 = proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, 2, switch_var, y, ctx);
  params.insert_all(res5_1.params);
  grads.insert_all(res5_1.grads);
  batch_stats.insert_all(res5_1.batch_stats);
  fixed_stats.insert_all(res5_1.fixed_stats);
  y = res5_1.op;*/
  auto res5_1 = spatial_parallel_proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, 2, switch_var, par_ys, parallel_ctx);
  /*params.insert_all(res5_1.params.at(0));
  grads.insert_all(res5_1.grads.at(0));
  batch_stats.insert_all(res5_1.batch_stats.at(0));
  fixed_stats.insert_all(res5_1.fixed_stats.at(0));*/
  parallel_vars_insert_all(dev_params, res5_1.params);
  parallel_vars_insert_all(dev_grads, res5_1.grads);
  parallel_vars_insert_all(dev_batch_stats, res5_1.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res5_1.fixed_stats);
  par_ys = res5_1.ops;
  //y = par_ys.at(0);

  if (false) {
    std::clog << "DEBUG: make res5_2..." << std::endl;

    /*auto res5_2 = residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, switch_var, y, ctx);
    params.insert_all(res5_2.params);
    grads.insert_all(res5_2.grads);
    batch_stats.insert_all(res5_2.batch_stats);
    fixed_stats.insert_all(res5_2.fixed_stats);
    y = res5_2.op;*/
    auto res5_2 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, switch_var, par_ys, parallel_ctx);
    /*params.insert_all(res5_2.params.at(0));
    grads.insert_all(res5_2.grads.at(0));
    batch_stats.insert_all(res5_2.batch_stats.at(0));
    fixed_stats.insert_all(res5_2.fixed_stats.at(0));*/
    parallel_vars_insert_all(dev_params, res5_2.params);
    parallel_vars_insert_all(dev_grads, res5_2.grads);
    parallel_vars_insert_all(dev_batch_stats, res5_2.batch_stats);
    parallel_vars_insert_all(dev_fixed_stats, res5_2.fixed_stats);
    par_ys = res5_2.ops;
    //y = par_ys.at(0);
  }

  if (true) {
    Conv2DShape res6_shape = {
      .filters = 512,
      .axes = {0, 1},
      .kernel = {3, 3},
      .dilate = {1, 1},
      .stride = {1, 1},
      .pad = {1, 1},
      .halo_padded_input = true,
      .enable_hmma = enable_hmma,
    };

    std::clog << "DEBUG: make res6_1..." << std::endl;

    auto res6_1 = spatial_parallel_proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res6_shape, 2, switch_var, par_ys, parallel_ctx);
    parallel_vars_insert_all(dev_params, res6_1.params);
    parallel_vars_insert_all(dev_grads, res6_1.grads);
    parallel_vars_insert_all(dev_batch_stats, res6_1.batch_stats);
    parallel_vars_insert_all(dev_fixed_stats, res6_1.fixed_stats);
    par_ys = res6_1.ops;

    if (false) {
      std::clog << "DEBUG: make res6_2..." << std::endl;

      auto res6_2 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res6_shape, switch_var, par_ys, parallel_ctx);
      parallel_vars_insert_all(dev_params, res6_2.params);
      parallel_vars_insert_all(dev_grads, res6_2.grads);
      parallel_vars_insert_all(dev_batch_stats, res6_2.batch_stats);
      parallel_vars_insert_all(dev_fixed_stats, res6_2.fixed_stats);
      par_ys = res6_2.ops;
    }
  }

  std::clog << "DEBUG: make global pool..." << std::endl;

  /*auto bnorm_final = batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(512, switch_var, y, ctx);
  params.insert_all(bnorm_final.params);
  grads.insert_all(bnorm_final.grads);
  batch_stats.insert_all(bnorm_final.batch_stats);
  fixed_stats.insert_all(bnorm_final.fixed_stats);
  y = bnorm_final.op;

  y = relu<GPUBatchArray3D<float>>(y, ctx);*/

  auto bnorm_final = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(512, switch_var, par_ys, parallel_ctx);
  /*params.insert_all(bnorm_final.params.at(0));
  grads.insert_all(bnorm_final.grads.at(0));
  batch_stats.insert_all(bnorm_final.batch_stats.at(0));
  fixed_stats.insert_all(bnorm_final.fixed_stats.at(0));*/
  parallel_vars_insert_all(dev_params, bnorm_final.params);
  parallel_vars_insert_all(dev_grads, bnorm_final.grads);
  parallel_vars_insert_all(dev_batch_stats, bnorm_final.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, bnorm_final.fixed_stats);
  par_ys = bnorm_final.ops;

  auto relu_final = parallel_relu<GPUBatchArray3D<float>>(par_ys, parallel_ctx);
  par_ys = relu_final;

  par_ys = SpatialGather2DOp<GPUBatchArray3D<float>>::MakeParallel(par_ys, parallel_ctx);
  par_ys = AllreduceOp<GPUBatchArray3D<float>>::MakeParallel(par_ys, parallel_ctx);

  Pool2DShape global_pool_shape = {
    .pooling = kAveragePool,
    .axes = {0, 1},
    //.window = {7, 7},
    //.stride = {7, 7},
    .window = {8, 8},
    .stride = {8, 8},
    .pad = {0, 0},
  };
  par_ys = CudnnPool2DOp<GPUBatchArray3D<float>>::MakeParallel(global_pool_shape, par_ys, parallel_ctx);

  std::clog << "DEBUG: package loss..." << std::endl;

  vector<shared_ptr<Var<GPUBatchArray1D<float>>>> par_zs;
  par_zs = parallel_flatten_nchw<GPUBatchArray3D<float>, GPUBatchArray1D<float>>(par_ys, parallel_ctx);

  auto fc1 = parallel_linear<GPUBatchArray1D<float>, GPUArray2D<float>>(1000, par_zs, xavier_init_gen<GPUArray2D<float>>(512, 1000), parallel_ctx);
  /*params.insert_all(fc1.params.at(0));
  grads.insert_all(fc1.grads.at(0));*/
  parallel_vars_insert_all(dev_params, fc1.params);
  parallel_vars_insert_all(dev_grads, fc1.grads);
  par_zs = fc1.ops;

  shared_ptr<Var<GPUBatchArray1D<float>>> z;
  z = ReduceOp<GPUBatchArray1D<float>>::MakeParallel(0, par_zs, parallel_ctx);
  z = ScaleOp<float, GPUBatchArray1D<float>>::Make(1.0f / static_cast<float>(num_ranks), z, ctx);

  z = io_var(z, ctx);
  VarIdSet logits = VarIdSet::Singleton(z->vars().at(0));

  auto nll_loss = io_var(softmax_nll_loss<GPUBatchArray1D<float>, GPUBatchArray0D<uint32_t>, GPUBatchArray0D<float>>(z, label, ctx), ctx);
  VarIdSet losses = VarIdSet::Singleton(nll_loss->vars().at(0));

  auto nll_loss_scaled = ScaleOp<float, GPUBatchArray0D<float>>::Make(1.0f / static_cast<float>(cfg.minibatch_size), nll_loss, ctx);
  auto sum_loss = batch_sum<GPUBatchArray0D<float>, GPUArray0D<float>>(nll_loss_scaled, ctx);

  /*CategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> loss = {
    .batch_norm_use_fixed = switch_var,
    .op = z,
    .loss = nll_loss,
    .sum_loss = sum_loss,
    .inputs = inputs,
    .labels = labels,
    .logits = logits,
    .losses = losses,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };*/
  ParallelCategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> loss = {
    .batch_norm_use_fixed = switch_var,
    .op = z,
    .loss = nll_loss,
    .sum_loss = sum_loss,
    .inputs = inputs,
    .labels = labels,
    .logits = logits,
    .losses = losses,
    .dev_params = dev_params,
    .dev_grads = dev_grads,
    .dev_batch_stats = dev_batch_stats,
    .dev_fixed_stats = dev_fixed_stats,
  };
  return loss;
}

template <>
CategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> ResNet<float>::MakeResNet50(const ConvNetConfig& cfg, shared_ptr<Context> ctx) {
  std::clog << "DEBUG: making resnet50..." << std::endl;

  VarIdSet params;
  VarIdSet grads;
  VarIdSet batch_stats;
  VarIdSet fixed_stats;

  std::clog << "DEBUG: make batch norm switch var..." << std::endl;
  //auto switch_var = val_src<bool>(false, ctx);
  auto switch_var = make_shared<bool>(false);

  std::clog << "DEBUG: make input preproc..." << std::endl;

  Shape3D raw_input_dim(3UL, cfg.input_w, cfg.input_h);
  Shape3D transpose_input_dim(cfg.input_w, cfg.input_h, 3UL);

  auto raw_input = batch_array_src<GPUBatchArray3D<float>, Shape3D>(raw_input_dim, cfg.batch_size, ctx);
  VarIdSet inputs = VarIdSet::Singleton(raw_input->vars().at(0));

  auto label = batch_array_src<GPUBatchArray0D<uint32_t>, Shape0D>(Shape0D(), cfg.batch_size, ctx);
  VarIdSet labels = VarIdSet::Singleton(label->vars().at(0));

  auto transpose_input = make_shared<TransposeNHWCToNCHWOp<GPUBatchArray3D<float>, Shape3D>>(raw_input, ctx);
  auto scale_input = make_shared<ScaleOp<float, GPUBatchArray3D<float>>>(1.0f / 255.0f, transpose_input, ctx);

  std::clog << "DEBUG: make conv1..." << std::endl;

  shared_ptr<Var<GPUBatchArray3D<float>>> y = transpose_input;

  Conv2DShape conv1_shape = {
    .filters = 64,
    .axes = {0, 1},
    .kernel = {7, 7},
    .dilate = {1, 1},
    .stride = {2, 2},
    .pad = {3, 3},
  };

  auto conv1 = conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, y, xavier_init_gen<GPUArray4D<float>>(3UL * conv1_shape.kernel.at(0) * conv1_shape.kernel.at(1), conv1_shape.filters), true, ctx);
  params.insert_all(conv1.params);
  grads.insert_all(conv1.grads);
  y = conv1.op;

  std::clog << "DEBUG: make pool1..." << std::endl;

  Pool2DShape pool1_shape = {
    .pooling = kMaxPool,
    .axes = {0, 1},
    .window = {3, 3},
    .stride = {2, 2},
    .pad = {1, 1},
  };
  y = make_shared<CudnnPool2DOp<GPUBatchArray3D<float>>>(pool1_shape, y, ctx);

  Conv2DShape res2_shape = {
    .filters = 64,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
  };

  std::clog << "DEBUG: make res2_1..." << std::endl;

  auto res2_1 = proj_bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, 4, 1, switch_var, y, ctx);
  params.insert_all(res2_1.params);
  grads.insert_all(res2_1.grads);
  batch_stats.insert_all(res2_1.batch_stats);
  fixed_stats.insert_all(res2_1.fixed_stats);
  y = res2_1.op;

  std::clog << "DEBUG: make res2_2..." << std::endl;

  auto res2_2 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, 4, switch_var, y, ctx);
  params.insert_all(res2_2.params);
  grads.insert_all(res2_2.grads);
  batch_stats.insert_all(res2_2.batch_stats);
  fixed_stats.insert_all(res2_2.fixed_stats);
  y = res2_2.op;

  std::clog << "DEBUG: make res2_3..." << std::endl;

  auto res2_3 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, 4, switch_var, y, ctx);
  params.insert_all(res2_3.params);
  grads.insert_all(res2_3.grads);
  batch_stats.insert_all(res2_3.batch_stats);
  fixed_stats.insert_all(res2_3.fixed_stats);
  y = res2_3.op;

  Conv2DShape res3_shape = {
    .filters = 128,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
  };

  std::clog << "DEBUG: make res3_1..." << std::endl;

  auto res3_1 = proj_bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, 4, 2, switch_var, y, ctx);
  params.insert_all(res3_1.params);
  grads.insert_all(res3_1.grads);
  batch_stats.insert_all(res3_1.batch_stats);
  fixed_stats.insert_all(res3_1.fixed_stats);
  y = res3_1.op;

  std::clog << "DEBUG: make res3_2..." << std::endl;

  auto res3_2 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, 4, switch_var, y, ctx);
  params.insert_all(res3_2.params);
  grads.insert_all(res3_2.grads);
  batch_stats.insert_all(res3_2.batch_stats);
  fixed_stats.insert_all(res3_2.fixed_stats);
  y = res3_2.op;

  std::clog << "DEBUG: make res3_3..." << std::endl;

  auto res3_3 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, 4, switch_var, y, ctx);
  params.insert_all(res3_3.params);
  grads.insert_all(res3_3.grads);
  batch_stats.insert_all(res3_3.batch_stats);
  fixed_stats.insert_all(res3_3.fixed_stats);
  y = res3_3.op;

  std::clog << "DEBUG: make res3_4..." << std::endl;

  auto res3_4 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, 4, switch_var, y, ctx);
  params.insert_all(res3_4.params);
  grads.insert_all(res3_4.grads);
  batch_stats.insert_all(res3_4.batch_stats);
  fixed_stats.insert_all(res3_4.fixed_stats);
  y = res3_4.op;

  Conv2DShape res4_shape = {
    .filters = 256,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
  };

  std::clog << "DEBUG: make res4_1..." << std::endl;

  auto res4_1 = proj_bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 4, 2, switch_var, y, ctx);
  params.insert_all(res4_1.params);
  grads.insert_all(res4_1.grads);
  batch_stats.insert_all(res4_1.batch_stats);
  fixed_stats.insert_all(res4_1.fixed_stats);
  y = res4_1.op;

  std::clog << "DEBUG: make res4_2..." << std::endl;

  auto res4_2 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 4, switch_var, y, ctx);
  params.insert_all(res4_2.params);
  grads.insert_all(res4_2.grads);
  batch_stats.insert_all(res4_2.batch_stats);
  fixed_stats.insert_all(res4_2.fixed_stats);
  y = res4_2.op;

  std::clog << "DEBUG: make res4_3..." << std::endl;

  auto res4_3 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 4, switch_var, y, ctx);
  params.insert_all(res4_3.params);
  grads.insert_all(res4_3.grads);
  batch_stats.insert_all(res4_3.batch_stats);
  fixed_stats.insert_all(res4_3.fixed_stats);
  y = res4_3.op;

  std::clog << "DEBUG: make res4_4..." << std::endl;

  auto res4_4 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 4, switch_var, y, ctx);
  params.insert_all(res4_4.params);
  grads.insert_all(res4_4.grads);
  batch_stats.insert_all(res4_4.batch_stats);
  fixed_stats.insert_all(res4_4.fixed_stats);
  y = res4_4.op;

  std::clog << "DEBUG: make res4_5..." << std::endl;

  auto res4_5 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 4, switch_var, y, ctx);
  params.insert_all(res4_5.params);
  grads.insert_all(res4_5.grads);
  batch_stats.insert_all(res4_5.batch_stats);
  fixed_stats.insert_all(res4_5.fixed_stats);
  y = res4_5.op;

  std::clog << "DEBUG: make res4_6..." << std::endl;

  auto res4_6 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 4, switch_var, y, ctx);
  params.insert_all(res4_6.params);
  grads.insert_all(res4_6.grads);
  batch_stats.insert_all(res4_6.batch_stats);
  fixed_stats.insert_all(res4_6.fixed_stats);
  y = res4_6.op;

  Conv2DShape res5_shape = {
    .filters = 512,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
  };

  std::clog << "DEBUG: make res5_1..." << std::endl;

  auto res5_1 = proj_bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, 4, 2, switch_var, y, ctx);
  params.insert_all(res5_1.params);
  grads.insert_all(res5_1.grads);
  batch_stats.insert_all(res5_1.batch_stats);
  fixed_stats.insert_all(res5_1.fixed_stats);
  y = res5_1.op;

  std::clog << "DEBUG: make res5_2..." << std::endl;

  auto res5_2 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, 4, switch_var, y, ctx);
  params.insert_all(res5_2.params);
  grads.insert_all(res5_2.grads);
  batch_stats.insert_all(res5_2.batch_stats);
  fixed_stats.insert_all(res5_2.fixed_stats);
  y = res5_2.op;

  std::clog << "DEBUG: make res5_3..." << std::endl;

  auto res5_3 = bottleneck_residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, 4, switch_var, y, ctx);
  params.insert_all(res5_3.params);
  grads.insert_all(res5_3.grads);
  batch_stats.insert_all(res5_3.batch_stats);
  fixed_stats.insert_all(res5_3.fixed_stats);
  y = res5_3.op;

  std::clog << "DEBUG: make global pool..." << std::endl;

  auto bnorm_final = batch_norm2d<bool, GPUBatchArray3D<float>, GPUArray1D<float>>(2048, switch_var, y, ctx);
  params.insert_all(bnorm_final.params);
  grads.insert_all(bnorm_final.grads);
  batch_stats.insert_all(bnorm_final.batch_stats);
  fixed_stats.insert_all(bnorm_final.fixed_stats);
  y = bnorm_final.op;

  y = relu<GPUBatchArray3D<float>>(y, ctx);

  Pool2DShape global_pool_shape = {
    .pooling = kAveragePool,
    .axes = {0, 1},
    .window = {7, 7},
    .stride = {7, 7},
    .pad = {0, 0},
  };
  y = make_shared<CudnnPool2DOp<GPUBatchArray3D<float>>>(global_pool_shape, y, ctx);

  std::clog << "DEBUG: package loss..." << std::endl;

  shared_ptr<Var<GPUBatchArray1D<float>>> z = flatten_nchw<GPUBatchArray3D<float>, GPUBatchArray1D<float>>(y, ctx);

  auto fc1 = linear<GPUBatchArray1D<float>, GPUArray2D<float>>(1000, z, xavier_init_gen<GPUArray2D<float>>(2048, 1000), ctx);
  params.insert_all(fc1.params);
  grads.insert_all(fc1.grads);
  z = io_var(fc1.op, ctx);
  VarIdSet logits = VarIdSet::Singleton(z->vars().at(0));

  auto nll_loss = io_var(softmax_nll_loss<GPUBatchArray1D<float>, GPUBatchArray0D<uint32_t>, GPUBatchArray0D<float>>(z, label, ctx), ctx);
  VarIdSet losses = VarIdSet::Singleton(nll_loss->vars().at(0));

  auto nll_loss_scaled = ScaleOp<float, GPUBatchArray0D<float>>::Make(1.0f / static_cast<float>(cfg.minibatch_size), nll_loss, ctx);
  auto sum_loss = batch_sum<GPUBatchArray0D<float>, GPUArray0D<float>>(nll_loss_scaled, ctx);

  CategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> loss = {
    .batch_norm_use_fixed = switch_var,
    .op = z,
    .loss = nll_loss,
    .sum_loss = sum_loss,
    .inputs = inputs,
    .labels = labels,
    .logits = logits,
    .losses = losses,
    .params = params,
    .grads = grads,
    .batch_stats = batch_stats,
    .fixed_stats = fixed_stats,
  };
  return loss;
}

template <>
ParallelCategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> ResNet<float>::MakeParallelResNet50(const ConvNetConfig& cfg, shared_ptr<Context> parallel_ctx) {
  std::clog << "DEBUG: making parallel resnet18..." << std::endl;

  const bool enable_hmma = false;

  size_t num_ranks = dynamic_cast<MultiGPUContext*>(parallel_ctx.get())->num_ranks();
  shared_ptr<Context> root_dev_ctx = dynamic_cast<MultiGPUContext*>(parallel_ctx.get())->device_context(0);
  shared_ptr<Context> ctx = root_dev_ctx;

  vector<VarIdSet> shared_params(num_ranks);
  vector<VarIdSet> shared_grads(num_ranks);
  vector<VarIdSet> dev_params(num_ranks);
  vector<VarIdSet> dev_grads(num_ranks);
  vector<VarIdSet> dev_batch_stats(num_ranks);
  vector<VarIdSet> dev_fixed_stats(num_ranks);

  std::clog << "DEBUG: make batch norm switch var..." << std::endl;
  auto switch_var = make_shared<bool>(false);

  std::clog << "DEBUG: make input preproc..." << std::endl;

  Shape3D raw_input_dim(3UL, cfg.input_w, cfg.input_h);

  auto raw_input = batch_array_src<GPUBatchArray3D<float>, Shape3D>(raw_input_dim, cfg.batch_size, root_dev_ctx);
  VarIdSet inputs = VarIdSet::Singleton(raw_input->vars().at(0));

  auto label = batch_array_src<GPUBatchArray0D<uint32_t>, Shape0D>(Shape0D(), cfg.batch_size, root_dev_ctx);
  VarIdSet labels = VarIdSet::Singleton(label->vars().at(0));

  auto transpose_input = TransposeNHWCToNCHWOp<GPUBatchArray3D<float>, Shape3D>::Make(raw_input, root_dev_ctx);
  auto scale_input = ScaleOp<float, GPUBatchArray3D<float>>::Make(1.0f / 255.0f, transpose_input, root_dev_ctx);

  std::clog << "DEBUG: make conv1..." << std::endl;

  vector<shared_ptr<Var<GPUBatchArray3D<float>>>> par_ys;
  shared_ptr<Var<GPUBatchArray3D<float>>> y = scale_input;

  par_ys = BroadcastOp<GPUBatchArray3D<float>>::MakeParallel(0, y, parallel_ctx);
  par_ys = SpatialScatter2DOp<GPUBatchArray3D<float>>::MakeParallel(par_ys, parallel_ctx);

  par_ys = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(3, par_ys, parallel_ctx);
  par_ys = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(3, par_ys, parallel_ctx);

  Conv2DShape conv1_shape = {
    .filters = 64,
    .axes = {0, 1},
    .kernel = {7, 7},
    .dilate = {1, 1},
    .stride = {2, 2},
    .pad = {3, 3},
    .halo_padded_input = true,
    .enable_hmma = enable_hmma,
  };

  auto conv1 = parallel_conv2d<GPUBatchArray3D<float>, GPUArray4D<float>>(conv1_shape, par_ys, xavier_init_gen<GPUArray4D<float>>(3UL * conv1_shape.kernel.at(0) * conv1_shape.kernel.at(1), conv1_shape.filters), true, parallel_ctx);
  parallel_vars_insert_all(dev_params, conv1.params);
  parallel_vars_insert_all(dev_grads, conv1.grads);
  par_ys = conv1.ops;

  std::clog << "DEBUG: make pool1..." << std::endl;

  par_ys = HaloPad2DOp<GPUBatchArray3D<float>>::MakeParallel(1, par_ys, parallel_ctx);
  par_ys = HaloExch2DOp<GPUBatchArray3D<float>>::MakeParallel(1, par_ys, parallel_ctx);

  Pool2DShape pool1_shape = {
    .pooling = kMaxPool,
    .axes = {0, 1},
    .window = {3, 3},
    .stride = {2, 2},
    .pad = {1, 1},
    .halo_padded_input = true,
  };
  par_ys = CudnnPool2DOp<GPUBatchArray3D<float>>::MakeParallel(pool1_shape, par_ys, parallel_ctx);

  Conv2DShape res2_shape = {
    .filters = 64,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    .halo_padded_input = true,
    .enable_hmma = enable_hmma,
  };

  std::clog << "DEBUG: make res2_1..." << std::endl;

  auto res2_1 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, switch_var, par_ys, parallel_ctx);
  parallel_vars_insert_all(dev_params, res2_1.params);
  parallel_vars_insert_all(dev_grads, res2_1.grads);
  parallel_vars_insert_all(dev_batch_stats, res2_1.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res2_1.fixed_stats);
  par_ys = res2_1.ops;

  std::clog << "DEBUG: make res2_2..." << std::endl;

  auto res2_2 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res2_shape, switch_var, par_ys, parallel_ctx);
  parallel_vars_insert_all(dev_params, res2_2.params);
  parallel_vars_insert_all(dev_grads, res2_2.grads);
  parallel_vars_insert_all(dev_batch_stats, res2_2.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res2_2.fixed_stats);
  par_ys = res2_2.ops;

  Conv2DShape res3_shape = {
    .filters = 128,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    .halo_padded_input = true,
    .enable_hmma = enable_hmma,
  };

  std::clog << "DEBUG: make res3_1..." << std::endl;

  auto res3_1 = spatial_parallel_proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, 2, switch_var, par_ys, parallel_ctx);
  parallel_vars_insert_all(dev_params, res3_1.params);
  parallel_vars_insert_all(dev_grads, res3_1.grads);
  parallel_vars_insert_all(dev_batch_stats, res3_1.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res3_1.fixed_stats);
  par_ys = res3_1.ops;

  std::clog << "DEBUG: make res3_2..." << std::endl;

  auto res3_2 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res3_shape, switch_var, par_ys, parallel_ctx);
  parallel_vars_insert_all(dev_params, res3_2.params);
  parallel_vars_insert_all(dev_grads, res3_2.grads);
  parallel_vars_insert_all(dev_batch_stats, res3_2.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res3_2.fixed_stats);
  par_ys = res3_2.ops;

  Conv2DShape res4_shape = {
    .filters = 256,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    .halo_padded_input = true,
    .enable_hmma = enable_hmma,
  };

  std::clog << "DEBUG: make res4_1..." << std::endl;

  auto res4_1 = spatial_parallel_proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, 2, switch_var, par_ys, parallel_ctx);
  parallel_vars_insert_all(dev_params, res4_1.params);
  parallel_vars_insert_all(dev_grads, res4_1.grads);
  parallel_vars_insert_all(dev_batch_stats, res4_1.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res4_1.fixed_stats);
  par_ys = res4_1.ops;

  std::clog << "DEBUG: make res4_2..." << std::endl;

  auto res4_2 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res4_shape, switch_var, par_ys, parallel_ctx);
  parallel_vars_insert_all(dev_params, res4_2.params);
  parallel_vars_insert_all(dev_grads, res4_2.grads);
  parallel_vars_insert_all(dev_batch_stats, res4_2.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res4_2.fixed_stats);
  par_ys = res4_2.ops;

  Conv2DShape res5_shape = {
    .filters = 512,
    .axes = {0, 1},
    .kernel = {3, 3},
    .dilate = {1, 1},
    .stride = {1, 1},
    .pad = {1, 1},
    .halo_padded_input = true,
    .enable_hmma = enable_hmma,
  };

  std::clog << "DEBUG: make res5_1..." << std::endl;

  auto res5_1 = spatial_parallel_proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, 2, switch_var, par_ys, parallel_ctx);
  parallel_vars_insert_all(dev_params, res5_1.params);
  parallel_vars_insert_all(dev_grads, res5_1.grads);
  parallel_vars_insert_all(dev_batch_stats, res5_1.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res5_1.fixed_stats);
  par_ys = res5_1.ops;

  std::clog << "DEBUG: make res5_2..." << std::endl;

  auto res5_2 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res5_shape, switch_var, par_ys, parallel_ctx);
  parallel_vars_insert_all(dev_params, res5_2.params);
  parallel_vars_insert_all(dev_grads, res5_2.grads);
  parallel_vars_insert_all(dev_batch_stats, res5_2.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, res5_2.fixed_stats);
  par_ys = res5_2.ops;

  if (false) {
    Conv2DShape res6_shape = {
      .filters = 512,
      .axes = {0, 1},
      .kernel = {3, 3},
      .dilate = {1, 1},
      .stride = {1, 1},
      .pad = {1, 1},
      .halo_padded_input = true,
      .enable_hmma = enable_hmma,
    };

    std::clog << "DEBUG: make res6_1..." << std::endl;

    auto res6_1 = spatial_parallel_proj_residual_v2_conv2d<GPUBatchArray3D<float>>(res6_shape, 2, switch_var, par_ys, parallel_ctx);
    parallel_vars_insert_all(dev_params, res6_1.params);
    parallel_vars_insert_all(dev_grads, res6_1.grads);
    parallel_vars_insert_all(dev_batch_stats, res6_1.batch_stats);
    parallel_vars_insert_all(dev_fixed_stats, res6_1.fixed_stats);
    par_ys = res6_1.ops;

    std::clog << "DEBUG: make res6_2..." << std::endl;

    auto res6_2 = spatial_parallel_residual_v2_conv2d<GPUBatchArray3D<float>>(res6_shape, switch_var, par_ys, parallel_ctx);
    parallel_vars_insert_all(dev_params, res6_2.params);
    parallel_vars_insert_all(dev_grads, res6_2.grads);
    parallel_vars_insert_all(dev_batch_stats, res6_2.batch_stats);
    parallel_vars_insert_all(dev_fixed_stats, res6_2.fixed_stats);
    par_ys = res6_2.ops;
  }

  std::clog << "DEBUG: make global pool..." << std::endl;

  auto bnorm_final = CudnnBatchNorm2DOp<bool, GPUBatchArray3D<float>, GPUArray1D<float>>::MakeParallel(512, switch_var, par_ys, parallel_ctx);
  parallel_vars_insert_all(dev_params, bnorm_final.params);
  parallel_vars_insert_all(dev_grads, bnorm_final.grads);
  parallel_vars_insert_all(dev_batch_stats, bnorm_final.batch_stats);
  parallel_vars_insert_all(dev_fixed_stats, bnorm_final.fixed_stats);
  par_ys = bnorm_final.ops;

  auto relu_final = parallel_relu<GPUBatchArray3D<float>>(par_ys, parallel_ctx);
  par_ys = relu_final;

  par_ys = SpatialGather2DOp<GPUBatchArray3D<float>>::MakeParallel(par_ys, parallel_ctx);
  par_ys = AllreduceOp<GPUBatchArray3D<float>>::MakeParallel(par_ys, parallel_ctx);

  Pool2DShape global_pool_shape = {
    .pooling = kAveragePool,
    .axes = {0, 1},
    .window = {8, 8},
    .stride = {8, 8},
    .pad = {0, 0},
  };
  par_ys = CudnnPool2DOp<GPUBatchArray3D<float>>::MakeParallel(global_pool_shape, par_ys, parallel_ctx);

  std::clog << "DEBUG: package loss..." << std::endl;

  vector<shared_ptr<Var<GPUBatchArray1D<float>>>> par_zs;
  par_zs = parallel_flatten_nchw<GPUBatchArray3D<float>, GPUBatchArray1D<float>>(par_ys, parallel_ctx);

  auto fc1 = parallel_linear<GPUBatchArray1D<float>, GPUArray2D<float>>(1000, par_zs, xavier_init_gen<GPUArray2D<float>>(512, 1000), parallel_ctx);
  parallel_vars_insert_all(dev_params, fc1.params);
  parallel_vars_insert_all(dev_grads, fc1.grads);
  par_zs = fc1.ops;

  shared_ptr<Var<GPUBatchArray1D<float>>> z;
  z = ReduceOp<GPUBatchArray1D<float>>::MakeParallel(0, par_zs, parallel_ctx);
  z = ScaleOp<float, GPUBatchArray1D<float>>::Make(1.0f / static_cast<float>(num_ranks), z, ctx);

  z = io_var(z, ctx);
  VarIdSet logits = VarIdSet::Singleton(z->vars().at(0));

  auto nll_loss = io_var(softmax_nll_loss<GPUBatchArray1D<float>, GPUBatchArray0D<uint32_t>, GPUBatchArray0D<float>>(z, label, ctx), ctx);
  VarIdSet losses = VarIdSet::Singleton(nll_loss->vars().at(0));

  auto nll_loss_scaled = ScaleOp<float, GPUBatchArray0D<float>>::Make(1.0f / static_cast<float>(cfg.minibatch_size), nll_loss, ctx);
  auto sum_loss = batch_sum<GPUBatchArray0D<float>, GPUArray0D<float>>(nll_loss_scaled, ctx);

  ParallelCategoricalNLLBatchNormLoss<GPUBatchArray1D<float>, GPUBatchArray0D<float>, GPUArray0D<float>> loss = {
    .batch_norm_use_fixed = switch_var,
    .op = z,
    .loss = nll_loss,
    .sum_loss = sum_loss,
    .inputs = inputs,
    .labels = labels,
    .logits = logits,
    .losses = losses,
    .dev_params = dev_params,
    .dev_grads = dev_grads,
    .dev_batch_stats = dev_batch_stats,
    .dev_fixed_stats = dev_fixed_stats,
  };
  return loss;
}

} // namespace arraydiff
