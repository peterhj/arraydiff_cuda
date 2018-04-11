#include <cuda_runtime.h>
//#include <cublas_v2.h>
//#include <cublasXt.h>
//#include <cudnn.h>
//#include <nccl.h>

#include <cassert>
#include <chrono>
#include <iostream>

#define CUDA_CHECK(e) (assert(cudaSuccess == (e)))
#define CUBLAS_CHECK(e) (assert(CUBLAS_STATUS_SUCCESS == (e)))
#define CUDNN_CHECK(e) (assert(CUDNN_STATUS_SUCCESS == (e)))

int main(int argc, const char** argv) {
  int num_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  std::clog << "num devices: " << num_devices << std::endl;
  num_devices = 1;

  const size_t buf_size = 32UL;

  cudaStream_t stream = NULL;
  float* x_h = NULL;
  float* x = NULL;
  float* y = NULL;

  x_h = (float*)malloc(buf_size * sizeof(float));

  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaMalloc((void**)&x, buf_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&y, buf_size * sizeof(float)));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaDeviceSynchronize());

  const int num_trials = 100;
  double avg_elapsed_ms = 0.0;
  std::clog << "running: reduce" << std::endl;

  for (int t = 0; t < num_trials; ++t) {
    for (size_t i = 0; i < buf_size; ++i) {
      x_h[i] = 42.0f;
    }
    std::clog << "DEBUG: x_h[0]: before: " << x_h[0] << std::endl;

    auto start = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaSetDevice(0));
    //reduce<float, AtomicReduceMap><<<(buf_size+1024-1)/1024, 1024, 0, stream>>>(
    //    buf_size, x, y);
    CUDA_CHECK(cudaMemcpyAsync(
        x_h,
        x,
        buf_size * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::clog << "DEBUG: x_h[0]: after:  " << x_h[0] << std::endl;

    auto lap = std::chrono::steady_clock::now();
    auto diff = lap - start;
    avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
  }

  avg_elapsed_ms /= num_trials;
  double avg_bandwidth = ((double)(buf_size * sizeof(float)) * 1.0e-9) / (avg_elapsed_ms * 1.0e-3);
  std::clog << "  avg wallclock:  " << avg_elapsed_ms << " ms" << std::endl;
  std::clog << "  avg bandwidth:  " << avg_bandwidth << " GB/s" << std::endl;

  return 0;
}
