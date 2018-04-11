#include "arraydiff/context.hh"
#include "arraydiff/spatial.hh"

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <iostream>

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

  const size_t halo_pad   = 1UL;
  const size_t batch_size = 32UL;
  const size_t channels   = 64UL;
  const size_t height     = 256UL / num_devices;
  const size_t width      = 256UL;
  const size_t buf_size = (width + halo_pad * 2UL) * (height + halo_pad * 2UL) * channels * batch_size;

  std::clog << "shape:"
      << " halo: " << halo_pad
      << " n: " << batch_size
      << " c: " << channels
      << " h: " << height
      << " w: " << width
      << std::endl;

  std::clog << "init ctx..." << std::endl;
  //cudaStream_t* streams = new cudaStream_t[num_devices];
  auto ctx = MultiGPUContext::Make();
  float** x = new float*[num_devices];
  float** y = new float*[num_devices];

  std::clog << "allocate comms..." << std::endl;
  Spatial2DComm* comms = new Spatial2DComm[num_devices];

  std::clog << "allocate bufs..." << std::endl;
  for (int dev = 0; dev < num_devices; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    //CUDA_CHECK(cudaStreamCreate(&streams[dev]));
    CUDA_CHECK(cudaMalloc((void**)&x[dev], buf_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&y[dev], buf_size * sizeof(float)));
    //CUDA_CHECK(cudaStreamSynchronize(streams[dev]));
    ctx->device_context(dev)->sync();
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  std::clog << "init comms..." << std::endl;
  Spatial2DComm::InitAll(comms, 1, num_devices);

  for (int dev = 0; dev < num_devices; ++dev) {
    CUDA_CHECK(cudaSetDevice(dev));
    //CUDA_CHECK(cudaStreamSynchronize(streams[dev]));
    ctx->device_context(dev)->sync();
  }

  const int num_trials = 1000;
  double avg_elapsed_ms = 0.0;
  std::clog << "running: spatial ops" << std::endl;

  for (int t = 0; t < num_trials; ++t) {
    auto start = std::chrono::steady_clock::now();

    for (int dev = 0; dev < num_devices; ++dev) {
      CUDA_CHECK(cudaSetDevice(dev));
      comms[dev].group_start();
      comms[dev].halo_exchange(
          x[dev],
          halo_pad, width, height, channels, batch_size,
          SPATIAL2D_TENSOR_NCHW,
          SPATIAL2D_DATA_FLOAT);
    }
    for (int dev = 0; dev < num_devices; ++dev) {
      CUDA_CHECK(cudaSetDevice(dev));
      auto conn = ctx->device_context(dev)->conn();
      comms[dev].group_end(conn);
    }
    for (int dev = 0; dev < num_devices; ++dev) {
      CUDA_CHECK(cudaSetDevice(dev));
      //CUDA_CHECK(cudaStreamSynchronize(streams[dev]));
      ctx->device_context(dev)->sync();
    }

    auto lap = std::chrono::steady_clock::now();
    auto diff = lap - start;
    avg_elapsed_ms += std::chrono::duration<double, std::milli>(diff).count();
  }

  avg_elapsed_ms /= num_trials;
  size_t work_size = halo_pad * width * channels * batch_size;
  double avg_bandwidth = ((double)(work_size * sizeof(float)) * 1.0e-9) / (avg_elapsed_ms * 1.0e-3);
  std::clog << "  avg wallclock:  " << avg_elapsed_ms << " ms" << std::endl;
  std::clog << "  avg bandwidth:  " << avg_bandwidth << " GB/s" << std::endl;

  return 0;
}
