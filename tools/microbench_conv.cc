//#include "nvham/comm.hh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <cudnn.h>
#include <nccl.h>

#include <cassert>
#include <chrono>
#include <iostream>

#define CUBLAS_CHECK(e) (assert(CUBLAS_STATUS_SUCCESS == (e)))
#define CUDNN_CHECK(e) (assert(CUDNN_STATUS_SUCCESS == (e)))

int main(int argc, const char** argv) {
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  std::clog << "num devices: " << num_devices << std::endl;
  num_devices = 1;

  size_t KERNEL_SIZE = 3UL;
  //size_t KERNEL_SIZE = 5UL;

  size_t GHOST_ZONE_W = 1UL;
  size_t GHOST_ZONE_H = 1UL;
  //size_t GHOST_ZONE_W = 2UL;
  //size_t GHOST_ZONE_H = 2UL;

  size_t batch_count = 32UL;
  size_t buf_width = 128UL;
  size_t buf_height = 128UL / 2;
  size_t buf_src_width = buf_width + 2 * GHOST_ZONE_W;
  size_t buf_src_height = buf_height + 2 * GHOST_ZONE_H;
  size_t buf_src_channels = 64UL;
  size_t buf_dst_channels = 64UL;
  std::clog << "DEBUG: batch sz:  " << batch_count << std::endl;
  std::clog << "DEBUG: src shape: " << buf_src_channels << " " << buf_src_height << " " << buf_src_width << std::endl;
  std::clog << "DEBUG: dst shape: " << buf_dst_channels << " " << buf_height << " " << buf_width << std::endl;

  size_t dst_height_stride = buf_width + 2 * GHOST_ZONE_W;
  size_t dst_chan_stride = (buf_height + 2 * GHOST_ZONE_H) * dst_height_stride;
  size_t dst_batch_stride = buf_dst_channels * dst_chan_stride;
  size_t dst_offset_inside_halo = (buf_width + GHOST_ZONE_W * 2) * GHOST_ZONE_H + GHOST_ZONE_W;

  size_t filter_len = KERNEL_SIZE * KERNEL_SIZE * buf_src_channels * buf_dst_channels;
  size_t bias_len = buf_dst_channels;
  size_t buf_src_len = batch_count * buf_src_channels * buf_src_height * buf_src_width;
  //size_t buf_dst_len = batch_count * buf_dst_channels * buf_height * buf_width;
  size_t buf_dst_len = batch_count * dst_batch_stride;
  //std::clog << "DEBUG: src size:  " << buf_src_len << std::endl;
  //std::clog << "DEBUG: dst size:  " << buf_dst_len << std::endl;
  size_t flops = 2UL * batch_count * buf_dst_channels * buf_src_channels * buf_height * buf_width * KERNEL_SIZE * KERNEL_SIZE;
  std::clog << "DEBUG: est flops: " << (double)(flops) * 1.0e-9 << " GF" << std::endl;

  cudaStream_t* streams = new cudaStream_t[num_devices];
  float** filter_bufs = new float*[num_devices];
  float** bias_bufs = new float*[num_devices];
  float** src_bufs = new float*[num_devices];
  float** dst_bufs = new float*[num_devices];
  uint8_t **workspace_bufs = new uint8_t*[num_devices];
  for (int dev = 0; dev < num_devices; ++dev) {
    cudaSetDevice(0);
    //cudaSetDevice(dev);
    cudaStreamCreate(&streams[dev]);
    cudaMalloc((void**)(&filter_bufs[dev]), filter_len * sizeof(float));
    cudaMemset(filter_bufs[dev], 0, filter_len * sizeof(float));
    cudaMalloc((void**)(&bias_bufs[dev]), bias_len * sizeof(float));
    cudaMemset(bias_bufs[dev], 0, bias_len * sizeof(float));
    cudaMalloc((void**)(&src_bufs[dev]), buf_src_len * sizeof(float));
    cudaMemset(src_bufs[dev], 0, buf_src_len * sizeof(float));
    cudaMalloc((void**)(&dst_bufs[dev]), buf_dst_len * sizeof(float));
    cudaMemset(dst_bufs[dev], 0, buf_dst_len * sizeof(float));
  }

  cudaSetDevice(0);

  cublasHandle_t cublas_h = NULL;
  CUBLAS_CHECK(cublasCreate(&cublas_h));

  /*cublasXtHandle_t cublasxt_h = NULL;
  CUBLAS_CHECK(cublasXtCreate(&cublasxt_h));
  int device_ids[4] = {0, 1, 2, 3};
  CUBLAS_CHECK(cublasXtDeviceSelect(cublasxt_h, 4, device_ids));
  int cublasxt_block_dim = 0;
  CUBLAS_CHECK(cublasXtGetBlockDim(cublasxt_h, &cublasxt_block_dim));
  std::clog << "DEBUG: cublasxt block dim: " << cublasxt_block_dim << std::endl;*/

  cudnnHandle_t cudnn_h = NULL;
  CUDNN_CHECK(cudnnCreate(&cudnn_h));

  cudnnFilterDescriptor_t filter_desc = NULL;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(
      filter_desc,
      CUDNN_DATA_FLOAT,
      CUDNN_TENSOR_NCHW,
      buf_dst_channels, buf_src_channels, KERNEL_SIZE, KERNEL_SIZE));

  cudnnTensorDescriptor_t bias_desc = NULL;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      bias_desc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      1, buf_dst_channels, 1, 1));

  cudnnTensorDescriptor_t src_desc = NULL;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&src_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      src_desc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batch_count, buf_src_channels, buf_src_height, buf_src_width));

  cudnnTensorDescriptor_t dst_desc = NULL;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&dst_desc));
  /*CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      dst_desc,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batch_count, buf_dst_channels, buf_height, buf_width));*/
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
      dst_desc,
      CUDNN_DATA_FLOAT,
      batch_count, buf_dst_channels, buf_height, buf_width,
      dst_batch_stride, dst_chan_stride, dst_height_stride, 1));

  cudnnConvolutionDescriptor_t conv_desc = NULL;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc,
      //0, 0, // pad
      KERNEL_SIZE / 2 - GHOST_ZONE_H, KERNEL_SIZE / 2 - GHOST_ZONE_W, // pad
      1, 1, // stride
      1, 1, // dilation
      CUDNN_CROSS_CORRELATION,
      CUDNN_DATA_FLOAT));

  cudnnActivationDescriptor_t act_desc = NULL;
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
  CUDNN_CHECK(cudnnSetActivationDescriptor(
      act_desc,
      CUDNN_ACTIVATION_RELU,
      CUDNN_NOT_PROPAGATE_NAN,
      0.0));

  int algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t algo_perf[10];
  CUDNN_CHECK(cudnnSetStream(cudnn_h, streams[0]));
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
      cudnn_h,
      src_desc,
      filter_desc,
      conv_desc,
      dst_desc,
      10,
      &algo_count,
      algo_perf));
  cudaStreamSynchronize(streams[0]);
  //assert(1 == algo_count);

  const int algo_idx = 0;
  std::clog << "DEBUG: conv fwd algos: " << algo_count << std::endl;
  std::clog << "DEBUG: conv fwd algo:  " << algo_perf[algo_idx].algo << std::endl;
  std::clog << "DEBUG: workspace size: " << algo_perf[algo_idx].memory << " bytes" << std::endl;
  for (int dev = 0; dev < num_devices; ++dev) {
    cudaSetDevice(0);
    //cudaSetDevice(dev);
    cudaMalloc((void**)(&workspace_bufs[dev]), algo_perf[algo_idx].memory);
  }

  cudaSetDevice(0);
  cudaDeviceSynchronize();

  cudaSetDevice(0);
  cudaStreamSynchronize(streams[0]);
  cudaStreamSynchronize(streams[1]);
  //cudaStreamSynchronize(streams2[0]);

  int num_trials = 1000;
  double avg_elapsed_ms = 0.0;
  for (int t = 0; t < num_trials; ++t) {
    auto start = std::chrono::steady_clock::now();

    cudaSetDevice(0);
    float alpha = 1.0f;
    float beta1 = 0.0f;
    float beta2 = 1.0f;
    CUDNN_CHECK(cudnnSetStream(cudnn_h, streams[0]));
    CUDNN_CHECK(cudnnConvolutionForward(
    //CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
        cudnn_h,
        &alpha,
        src_desc, src_bufs[0],
        filter_desc, filter_bufs[0],
        conv_desc,
        algo_perf[algo_idx].algo,
        workspace_bufs[0],
        algo_perf[algo_idx].memory,
        &beta1,
        //dst_desc, dst_bufs[0] + dst_offset_inside_halo,
        //bias_desc, bias_bufs[0],
        //act_desc,
        dst_desc, dst_bufs[0] + dst_offset_inside_halo));
    /*CUDNN_CHECK(cudnnAddTensor(
        cudnn_h,
        &alpha,
        bias_desc, bias_bufs[0],
        &beta2,
        dst_desc, dst_bufs[0]));*/
    /*for (int dev = 1; dev < num_devices; ++dev) {
    CUDNN_CHECK(cudnnSetStream(cudnn_h, streams[dev]));
    CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
        cudnn_h,
        &alpha,
        src_desc, src_bufs[dev],
        filter_desc, filter_bufs[dev],
        conv_desc,
        algo_perf[algo_idx].algo,
        workspace_bufs[dev],
        algo_perf[algo_idx].memory,
        &beta1,
        dst_desc, dst_bufs[dev] + dst_offset_inside_halo,
        bias_desc, bias_bufs[dev],
        act_desc,
        dst_desc, dst_bufs[dev] + dst_offset_inside_halo));
    }*/
    cudaStreamSynchronize(streams[0]);
    for (int dev = 1; dev < num_devices; ++dev) {
    cudaStreamSynchronize(streams[dev]);
    }

    auto lap = std::chrono::steady_clock::now();
    auto diff = lap - start;
    avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
  }
  avg_elapsed_ms /= num_trials;

  std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;

  return 0;
}
