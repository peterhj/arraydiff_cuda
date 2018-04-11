//#include "comm.hh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
//#include <cudnn.h>
#include <nccl.h>

#include <cassert>
#include <chrono>
#include <iostream>

//#define ncclGroupStart()
//#define ncclGroupEnd()

int main(int argc, const char** argv) {
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  std::clog << "num devices: " << num_devices << std::endl;

  //size_t batch_count = 32;
  //size_t buf_width = 1;
  //size_t buf_height = 1;
  //size_t buf_channels = 1 * 128 * 1024;
  size_t batch_count = 256UL;
  size_t buf_width = 28UL;
  size_t buf_height = 28UL;
  size_t buf_channels = 64UL;
  size_t buf_len = batch_count * buf_channels * buf_height * buf_width;
  //size_t buf_len = 128 * 1024;
  //size_t buf_len = 4 * 1024 * 1024;
  cudaStream_t* streams = new cudaStream_t[num_devices];
  float** src_bufs = new float*[num_devices];
  float** dst_bufs = new float*[num_devices];
  float** tmp1_bufs = new float*[num_devices];
  float** tmp2_bufs = new float*[num_devices];
  for (int dev = 0; dev < num_devices; ++dev) {
    cudaSetDevice(dev);
    cudaStreamCreate(&streams[dev]);
    cudaMalloc((void**)(&src_bufs[dev]), buf_len * sizeof(float));
    cudaMalloc((void**)(&dst_bufs[dev]), buf_len * sizeof(float));
    cudaMalloc((void**)(&tmp1_bufs[dev]), buf_len * sizeof(float));
    cudaMalloc((void**)(&tmp2_bufs[dev]), buf_len * sizeof(float));
  }

  cublasHandle_t cublas_h = NULL;
  cublasCreate(&cublas_h);

  /*cudnnHandle_t cudnn_h = NULL;
  cudnnCreate(&cudnn_h);

  cudnnTensorDescriptor_t tensor_desc;
  cudnnCreateTensorDescriptor(&tensor_desc);
  cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_count, buf_channels, buf_height, buf_width);

  cudnnTensorDescriptor_t transpose_desc;
  cudnnCreateTensorDescriptor(&transpose_desc);
  cudnnSetTensor4dDescriptor(transpose_desc, CUDNN_TENSOR_CHWN, CUDNN_DATA_FLOAT, batch_count, buf_channels, buf_height, buf_width);*/

  ncclComm_t* comms = new ncclComm_t[num_devices];
  ncclCommInitAll(comms, num_devices, NULL);

  //nvarraydiff::MultiGPUComm mcomm(num_devices);

  for (int dev = 0; dev < num_devices; ++dev) {
    cudaSetDevice(dev);
    cudaStreamSynchronize(streams[dev]);
  }

  std::clog << "benchmarking memcpy..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
      auto start = std::chrono::steady_clock::now();
      cudaSetDevice(0);
      cudaMemcpyAsync(dst_bufs[0], src_bufs[0], buf_len * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
      cudaStreamSynchronize(streams[0]);
      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    }
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg bandwidth:        " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };

  std::clog << "benchmarking geam transpose..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
      auto start = std::chrono::steady_clock::now();
      cudaSetDevice(0);
      cublasSetStream(cublas_h, streams[0]);
      float alpha = 1.0f;
      float beta = 0.0f;
      int result = cublasSgeam(cublas_h,
          CUBLAS_OP_T, CUBLAS_OP_N,
          batch_count, buf_width * buf_height * buf_channels,
          &alpha,
          (const float*)src_bufs[0], buf_width * buf_height * buf_channels,
          &beta,
          (const float*)dst_bufs[0], batch_count,
          (float*)dst_bufs[0], batch_count
      );
      assert(0 == result);
      cudaStreamSynchronize(streams[0]);
      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    }
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg bandwidth:        " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };

/*
  std::clog << "benchmarking new allgather..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
      auto start = std::chrono::steady_clock::now();
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        mcomm.all_gather(dev, src_bufs[dev], buf_len / num_devices, ncclFloat, (void**)dst_bufs, streams[dev]);
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }
      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    }
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg quasi-bandwidth:  " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };

  std::clog << "benchmarking new batched allgather..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
      auto start = std::chrono::steady_clock::now();
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        mcomm.all_gather_batch(dev, src_bufs[dev], buf_len / num_devices / batch_count, batch_count, ncclFloat, (void**)dst_bufs, streams[dev]);
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }
      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    }
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg quasi-bandwidth:  " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };
*/

  std::clog << "benchmarking allgather..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    auto start = std::chrono::steady_clock::now();
    for (int t = 0; t < num_trials; ++t) {
      ncclGroupStart();
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        //ncclAllGather(src_bufs[dev], buf_len / num_devices, ncclFloat, dst_bufs[dev], comms[dev], streams[dev]);
        ncclAllGather(src_bufs[dev], dst_bufs[dev], buf_len / num_devices, ncclFloat, comms[dev], streams[dev]);
      }
      ncclGroupEnd();
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }
    }
    auto lap = std::chrono::steady_clock::now();
    auto diff = lap - start;
    avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg quasi-bandwidth:  " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };

  /*std::clog << "benchmarking batched allgather..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
      auto start = std::chrono::steady_clock::now();
      size_t batch_stride = buf_len / num_devices / batch_count;
      for (size_t idx = 0; idx < batch_count; ++idx) {
        ncclGroupStart();
        for (int dev = 0; dev < num_devices; ++dev) {
          cudaSetDevice(dev);
          //ncclAllGather(src_bufs[dev] + batch_stride * idx, batch_stride, ncclFloat, dst_bufs[dev] + num_devices * batch_stride * idx, comms[dev], streams[dev]);
          ncclAllGather(src_bufs[dev] + batch_stride * idx, dst_bufs[dev] + num_devices * batch_stride * idx, batch_stride, ncclFloat, comms[dev], streams[dev]);
        }
        ncclGroupEnd();
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }
      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    }
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg quasi-bandwidth:  " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };*/

  std::clog << "benchmarking pack/unpack batched allgather..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
      auto start = std::chrono::steady_clock::now();

      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);

        float alpha = 1.0f;
        float beta = 0.0f;
        int result = -1;

        cublasSetStream(cublas_h, streams[dev]);
        result = cublasSgeam(cublas_h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            batch_count, buf_width * buf_height * buf_channels / num_devices,
            &alpha,
            (const float*)src_bufs[dev], buf_width * buf_height * buf_channels / num_devices,
            &beta,
            (const float*)tmp1_bufs[dev], batch_count,
            (float*)tmp1_bufs[dev], batch_count
        );
        assert(0 == result);
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }

      ncclGroupStart();
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        //ncclAllGather(tmp1_bufs[dev], buf_len / num_devices, ncclFloat, tmp2_bufs[dev], comms[dev], streams[dev]);
        ncclAllGather(tmp1_bufs[dev], tmp2_bufs[dev], buf_len / num_devices, ncclFloat, comms[dev], streams[dev]);
      }
      ncclGroupEnd();
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }

      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);

        float alpha = 1.0f;
        float beta = 0.0f;
        int result = -1;

        cublasSetStream(cublas_h, streams[dev]);
        result = cublasSgeam(cublas_h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            buf_width * buf_height * buf_channels, batch_count,
            &alpha,
            (const float*)tmp2_bufs[dev], batch_count,
            &beta,
            (const float*)dst_bufs[dev], buf_width * buf_height * buf_channels,
            (float*)dst_bufs[dev], buf_width * buf_height * buf_channels
        );
        assert(0 == result);
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }

      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    }
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg quasi-bandwidth:  " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };

  std::clog << "benchmarking reduce-scatter..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
      auto start = std::chrono::steady_clock::now();
      ncclGroupStart();
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        ncclReduceScatter(dst_bufs[dev], src_bufs[dev], buf_len / num_devices, ncclFloat, ncclSum, comms[dev], streams[dev]);
      }
      ncclGroupEnd();
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }
      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    }
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg quasi-bandwidth:  " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };

  std::clog << "benchmarking batched reduce-scatter..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
      auto start = std::chrono::steady_clock::now();
      size_t batch_stride = buf_len / num_devices / batch_count;
      for (size_t idx = 0; idx < batch_count; ++idx) {
        ncclGroupStart();
        for (int dev = 0; dev < num_devices; ++dev) {
          cudaSetDevice(dev);
          ncclReduceScatter(dst_bufs[dev] + batch_stride * idx, src_bufs[dev] + num_devices * batch_stride * idx, batch_stride, ncclFloat, ncclSum, comms[dev], streams[dev]);
        }
        ncclGroupEnd();
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }
      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    }
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg quasi-bandwidth:  " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };

  std::clog << "benchmarking allreduce..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
      auto start = std::chrono::steady_clock::now();
      ncclGroupStart();
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        ncclAllReduce(src_bufs[dev], dst_bufs[dev], buf_len, ncclFloat, ncclSum, comms[dev], streams[dev]);
      }
      ncclGroupEnd();
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }
      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    }
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg quasi-bandwidth:  " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };

  /*std::clog << "benchmarking batched allreduce..." << std::endl;
  {
    int num_trials = 1000;
    double avg_elapsed_ms = 0.0;
    for (int t = 0; t < num_trials; ++t) {
      auto start = std::chrono::steady_clock::now();
      size_t batch_stride = buf_len / batch_count;
      for (size_t idx = 0; idx < batch_count; ++idx) {
        ncclGroupStart();
        for (int dev = 0; dev < num_devices; ++dev) {
          cudaSetDevice(dev);
          ncclAllReduce(src_bufs[dev] + batch_stride * idx, dst_bufs[dev] + batch_stride * idx, batch_stride, ncclFloat, ncclSum, comms[dev], streams[dev]);
        }
        ncclGroupEnd();
      }
      for (int dev = 0; dev < num_devices; ++dev) {
        cudaSetDevice(dev);
        cudaStreamSynchronize(streams[dev]);
      }
      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
    }
    avg_elapsed_ms /= num_trials;

    double buf_size_mb = (double)(buf_len * sizeof(float)) / (double)(1024 * 1024);
    double avg_bandwidth_gbps = (double)(buf_len * sizeof(float)) / avg_elapsed_ms * 1000.0 / (double)(1024 * 1024 * 1024);
    std::clog << "  buffer size:          " << buf_size_mb << " MB" << std::endl;
    std::clog << "  avg wallclock:        " << avg_elapsed_ms << " ms" << std::endl;
    std::clog << "  avg quasi-bandwidth:  " << avg_bandwidth_gbps << " GB/s" << std::endl;
  };*/

  for (int dev = 0; dev < num_devices; ++dev) {
    ncclCommDestroy(comms[dev]);
    cudaFree((void*)dst_bufs[dev]);
    cudaFree((void*)src_bufs[dev]);
    cudaStreamDestroy(streams[dev]);
  }

  delete[] comms;
  delete[] dst_bufs;
  delete[] src_bufs;
  delete[] streams;

  return 0;
}
