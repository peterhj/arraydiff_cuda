#include "arraydiff/context.hh"
#include "arraydiff/spatial.hh"

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>

#define CUDA_CHECK(e) (assert(cudaSuccess == (e)))
#define CUBLAS_CHECK(e) (assert(CUBLAS_STATUS_SUCCESS == (e)))
#define CUDNN_CHECK(e) (assert(CUDNN_STATUS_SUCCESS == (e)))

using namespace arraydiff;

int main(void) {
  int num_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  std::clog << "num devices: " << num_devices << std::endl;
  //num_devices = 1;
  //num_devices = 2;

  const size_t conv_size  = 3UL;

  //const size_t dilation   = 1UL;
  const size_t dilation   = 3UL;

  //const size_t halo_pad   = 0UL;
  //const size_t halo_pad   = 1UL;
  const size_t halo_pad   = 3UL;

  //const size_t image_dim  = 128UL;
  //const size_t image_dim  = 256UL;
  const size_t image_dim  = 512UL;

  const size_t batch_size = 32UL;
  const size_t channels   = 64UL;
  const size_t height     = image_dim / num_devices;
  const size_t width      = image_dim;

  const size_t buf_size   = (width + halo_pad * 2UL) * (height + halo_pad * 2UL) * channels * batch_size;
  const size_t w_size     = conv_size * conv_size * channels * channels;

  const size_t src_width = width + halo_pad * 2UL;
  const size_t src_height = height + halo_pad * 2UL;

  std::clog << "shape:"
      << " conv: " << conv_size
      << " dilate: " << dilation
      << " halo: " << halo_pad
      << " n: " << batch_size
      << " c: " << channels
      << " h: " << height
      << " w: " << width
      << std::endl;

  auto ctx = MultiGPUContext::Make();

  Spatial2DComm* comms = new Spatial2DComm[num_devices];
  Spatial2DComm::InitAll(comms, 1, num_devices);

  cudnnFilterDescriptor_t filter_desc = NULL;
  cudnnTensorDescriptor_t src_desc = NULL;
  cudnnTensorDescriptor_t dst_desc = NULL;
  cudnnConvolutionDescriptor_t conv_desc = NULL;
  cudnnConvolutionFwdAlgo_t conv_fwd_algo;
  cudnnConvolutionBwdFilterAlgo_t conv_bwd_w_algo;
  cudnnConvolutionBwdDataAlgo_t conv_bwd_x_algo;
  size_t conv_fwd_workspace = 0;
  size_t conv_bwd_w_workspace = 0;
  size_t conv_bwd_x_workspace = 0;
  size_t conv_max_workspace = 0;

  {
    auto conn = ctx->device_context(0)->conn();

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        channels, channels, conv_size, conv_size));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&src_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        src_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, channels, src_height, src_width));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dst_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        dst_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size, channels, height, width));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        ((conv_size - 1) * dilation + 1) / 2 - halo_pad, ((conv_size - 1) * dilation + 1) / 2 - halo_pad, // pad
        1, 1, // stride
        dilation, dilation, // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    CUDA_CHECK(cudaStreamSynchronize(conn.stream()));

    int algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t algo_perf[10];
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        conn.cudnn(),
        src_desc,
        filter_desc,
        conv_desc,
        dst_desc,
        10,
        &algo_count,
        algo_perf));
    CUDA_CHECK(cudaStreamSynchronize(conn.stream()));

    int algo_idx = 0;
    for (int i = 0; i < 10; ++i) {
      if (algo_perf[i].memory <= std::numeric_limits<unsigned int>::max()) {
        algo_idx = i;
        break;
      }
    }

    std::clog << "DEBUG: conv fwd algos: " << algo_count << std::endl;
    std::clog << "DEBUG: conv fwd algo:  " << algo_perf[algo_idx].algo << std::endl;
    std::clog << "DEBUG: workspace size: " << algo_perf[algo_idx].memory << " bytes" << std::endl;

    conv_fwd_algo = algo_perf[algo_idx].algo;
    conv_fwd_workspace = algo_perf[algo_idx].memory;
    conv_max_workspace = std::max(conv_fwd_workspace, conv_max_workspace);

    algo_count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t bwd_w_algo_perf[10];
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithm(
        conn.cudnn(),
        src_desc,
        dst_desc,
        conv_desc,
        filter_desc,
        10,
        &algo_count,
        bwd_w_algo_perf));
    CUDA_CHECK(cudaStreamSynchronize(conn.stream()));

    algo_idx = 0;
    for (int i = 0; i < 10; ++i) {
      if (bwd_w_algo_perf[i].memory <= std::numeric_limits<unsigned int>::max()) {
        algo_idx = i;
        break;
      }
    }

    std::clog << "DEBUG: conv bwd w algos: " << algo_count << std::endl;
    std::clog << "DEBUG: conv bwd w algo:  " << bwd_w_algo_perf[algo_idx].algo << std::endl;
    std::clog << "DEBUG: workspace size:   " << bwd_w_algo_perf[algo_idx].memory << " bytes" << std::endl;

    conv_bwd_w_algo = bwd_w_algo_perf[algo_idx].algo;
    conv_bwd_w_workspace = bwd_w_algo_perf[algo_idx].memory;
    conv_max_workspace = std::max(conv_bwd_w_workspace, conv_max_workspace);

    algo_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t bwd_x_algo_perf[10];
    CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithm(
        conn.cudnn(),
        filter_desc,
        dst_desc,
        conv_desc,
        src_desc,
        10,
        &algo_count,
        bwd_x_algo_perf));
    CUDA_CHECK(cudaStreamSynchronize(conn.stream()));

    algo_idx = 0;
    for (int i = 0; i < 10; ++i) {
      if (bwd_x_algo_perf[i].memory <= std::numeric_limits<unsigned int>::max()) {
        algo_idx = i;
        break;
      }
    }

    std::clog << "DEBUG: conv bwd x algos: " << algo_count << std::endl;
    std::clog << "DEBUG: conv bwd x algo:  " << bwd_x_algo_perf[algo_idx].algo << std::endl;
    std::clog << "DEBUG: workspace size:   " << bwd_x_algo_perf[algo_idx].memory << " bytes" << std::endl;

    conv_bwd_x_algo = bwd_x_algo_perf[algo_idx].algo;
    conv_bwd_x_workspace = bwd_x_algo_perf[algo_idx].memory;
    conv_max_workspace = std::max(conv_bwd_x_workspace, conv_max_workspace);
  }

  conv_max_workspace = std::max(4096UL, conv_max_workspace);
  std::clog << "DEBUG: max workspace size:   " << conv_max_workspace << " bytes" << std::endl;

  float** x = (float**)malloc(sizeof(float*) * num_devices);
  float** y = (float**)malloc(sizeof(float*) * num_devices);
  float** w = (float**)malloc(sizeof(float*) * num_devices);
  float** dx = (float**)malloc(sizeof(float*) * num_devices);
  float** dy = (float**)malloc(sizeof(float*) * num_devices);
  float** dw = (float**)malloc(sizeof(float*) * num_devices);
  uint8_t** scratch = (uint8_t**)malloc(sizeof(uint8_t*) * num_devices);

  for (int dev = 0; dev < num_devices; ++dev) {
    auto conn = ctx->device_context(dev)->conn();
    conn.sync();
    CUDA_CHECK(cudaMalloc((void**)&x[dev], buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&y[dev], buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&w[dev], w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dx[dev], buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dy[dev], buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dw[dev], w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&scratch[dev], conv_max_workspace * sizeof(uint8_t)));
    CUDA_CHECK(cudaDeviceSynchronize());
    ctx->device_context(dev)->sync();
  }

  if (halo_pad > 0) {
    const int num_trials = 10000;
    double avg_elapsed_ms = 0.0;

    std::clog << "running: spatial halo exchange (fwd)" << std::endl;

    for (int t = 0; t < num_trials + 1; ++t) {
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        ctx->device_context(dev)->sync();
      }

      auto start = std::chrono::steady_clock::now();

      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        comms[dev].group_start();
        comms[dev].halo_exchange(
            x[dev],
            halo_pad, src_width, height, channels, batch_size,
            SPATIAL2D_TENSOR_NCHW,
            SPATIAL2D_DATA_FLOAT);
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        comms[dev].group_end(conn);
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        ctx->device_context(dev)->sync();
      }

      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      if (t >= 1) {
        avg_elapsed_ms += (std::chrono::duration<double, std::milli>(diff).count() - avg_elapsed_ms) / (static_cast<double>(t));
      }
    }

    size_t work_size = halo_pad * width * channels * batch_size;
    double avg_bandwidth = ((double)(work_size * sizeof(float)) * 1.0e-9) / (avg_elapsed_ms * 1.0e-3);
    std::clog << "  avg wallclock:  " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg bandwidth:  " << avg_bandwidth << " GB/s" << std::endl;
  }

  if (halo_pad > 0) {
    const int num_trials = 10000;
    double avg_elapsed_ms = 0.0;

    std::clog << "running: spatial halo reduce (bwd)" << std::endl;

    for (int t = 0; t < num_trials + 1; ++t) {
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        ctx->device_context(dev)->sync();
      }

      auto start = std::chrono::steady_clock::now();

      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        comms[dev].group_start();
        comms[dev].halo_reduce(
            x[dev],
            halo_pad, src_width, height, channels, batch_size,
            SPATIAL2D_TENSOR_NCHW,
            SPATIAL2D_DATA_FLOAT);
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        comms[dev].group_end(conn);
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        ctx->device_context(dev)->sync();
      }

      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      if (t >= 1) {
        avg_elapsed_ms += (std::chrono::duration<double, std::milli>(diff).count() - avg_elapsed_ms) / (static_cast<double>(t));
      }
    }

    size_t work_size = halo_pad * width * channels * batch_size;
    double avg_bandwidth = ((double)(work_size * sizeof(float)) * 1.0e-9) / (avg_elapsed_ms * 1.0e-3);
    std::clog << "  avg wallclock:  " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg bandwidth:  " << avg_bandwidth << " GB/s" << std::endl;
  }

  {
    const int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    double sum_sq_rms_elapsed_ms = 0.0;

    std::clog << "running: spatial conv (fwd)" << std::endl;

    for (int t = 0; t < num_trials + 1; ++t) {
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        ctx->device_context(dev)->sync();
      }

      auto start = std::chrono::steady_clock::now();

      if (halo_pad > 0) {
        for (int dev = 0; dev < num_devices; ++dev) {
          auto conn = ctx->device_context(dev)->conn();
          comms[dev].group_start();
          comms[dev].halo_exchange(
              x[dev],
              halo_pad, width, height, channels, batch_size,
              SPATIAL2D_TENSOR_NCHW,
              SPATIAL2D_DATA_FLOAT);
        }
        for (int dev = 0; dev < num_devices; ++dev) {
          auto conn = ctx->device_context(dev)->conn();
          comms[dev].group_end(conn);
        }
        for (int dev = 0; dev < num_devices; ++dev) {
          auto conn = ctx->device_context(dev)->conn();
          ctx->device_context(dev)->sync();
        }
      }

      float alpha = 1.0f;
      float beta = 0.0f;
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
        CUDNN_CHECK(cudnnConvolutionForward(
            conn.cudnn(),
            &alpha,
            src_desc, x[dev],
            filter_desc, w[dev],
            conv_desc,
            conv_fwd_algo,
            scratch[dev],
            conv_fwd_workspace,
            &beta,
            dst_desc, y[dev]));
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        ctx->device_context(dev)->sync();
      }

      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      if (t >= 1) {
        double delta_ms = std::chrono::duration<double, std::milli>(diff).count();
        double prev_avg_elapsed_ms = avg_elapsed_ms;
        avg_elapsed_ms += (delta_ms - avg_elapsed_ms) / (static_cast<double>(t));
        sum_sq_rms_elapsed_ms += (delta_ms - avg_elapsed_ms) * (delta_ms - prev_avg_elapsed_ms);
      }
    }

    std::clog << "  num trials:     " << num_trials << std::endl;
    std::clog << "  avg wallclock:  " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  err wallclock:  " << std::sqrt(sum_sq_rms_elapsed_ms / static_cast<double>(num_trials - 1) / static_cast<double>(num_trials)) << " ms" << std::endl;
    std::clog << "  std wallclock:  " << std::sqrt(sum_sq_rms_elapsed_ms / static_cast<double>(num_trials - 1)) << " ms" << std::endl;
  }

  {
    const int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    double sum_sq_rms_elapsed_ms = 0.0;

    std::clog << "running: spatial conv (bwd)" << std::endl;

    for (int t = 0; t < num_trials + 1; ++t) {
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        ctx->device_context(dev)->sync();
      }

      auto start = std::chrono::steady_clock::now();

      float alpha = 1.0f;
      float beta = 0.0f;

      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
            conn.cudnn(),
            &alpha,
            src_desc, x[dev],
            dst_desc, dy[dev],
            conv_desc,
            conv_bwd_w_algo,
            scratch[dev],
            conv_bwd_w_workspace,
            &beta,
            filter_desc, dw[dev]));
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        CUDNN_CHECK(cudnnSetStream(conn.cudnn(), conn.stream()));
        CUDNN_CHECK(cudnnConvolutionBackwardData(
            conn.cudnn(),
            &alpha,
            filter_desc, w[dev],
            dst_desc, dy[dev],
            conv_desc,
            conv_bwd_x_algo,
            scratch[dev],
            conv_bwd_x_workspace,
            &beta,
            src_desc, dx[dev]));
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        auto conn = ctx->device_context(dev)->conn();
        ctx->device_context(dev)->sync();
      }

      if (halo_pad > 0) {
        for (int dev = 0; dev < num_devices; ++dev) {
          auto conn = ctx->device_context(dev)->conn();
          comms[dev].group_start();
          comms[dev].halo_reduce(
              dx[dev],
              halo_pad, width, height, channels, batch_size,
              SPATIAL2D_TENSOR_NCHW,
              SPATIAL2D_DATA_FLOAT);
        }
        for (int dev = 0; dev < num_devices; ++dev) {
          auto conn = ctx->device_context(dev)->conn();
          comms[dev].group_end(conn);
        }
        for (int dev = 0; dev < num_devices; ++dev) {
          auto conn = ctx->device_context(dev)->conn();
          ctx->device_context(dev)->sync();
        }
      }

      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      if (t >= 1) {
        double delta_ms = std::chrono::duration<double, std::milli>(diff).count();
        double prev_avg_elapsed_ms = avg_elapsed_ms;
        avg_elapsed_ms += (delta_ms - avg_elapsed_ms) / (static_cast<double>(t));
        sum_sq_rms_elapsed_ms += (delta_ms - avg_elapsed_ms) * (delta_ms - prev_avg_elapsed_ms);
      }
    }

    std::clog << "  num trials:     " << num_trials << std::endl;
    std::clog << "  avg wallclock:  " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  err wallclock:  " << std::sqrt(sum_sq_rms_elapsed_ms / static_cast<double>(num_trials - 1) / static_cast<double>(num_trials)) << " ms" << std::endl;
    std::clog << "  std wallclock:  " << std::sqrt(sum_sq_rms_elapsed_ms / static_cast<double>(num_trials - 1)) << " ms" << std::endl;
  }

  return 0;
}
