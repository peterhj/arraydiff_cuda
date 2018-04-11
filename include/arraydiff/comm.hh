#ifndef ARRAYDIFF_COMM_HH
#define ARRAYDIFF_COMM_HH

#include <cuda_runtime.h>
//#include <nccl.h>

#include <cstdint>

#define ARRAYDIFF_MAX_PEERS 8

namespace arraydiff {

class MultiGPUComm2D {
public:
  static void InitAll(MultiGPUComm2D* comms, uint32_t dim_u, uint32_t dim_v /*, const int* devices*/);

  template <typename T>
  void ghost_copy_nchw(
      T* padded_buf,
      uint32_t batch_size, uint32_t channels,
      uint32_t h_base, uint32_t w_base,
      uint32_t h_halo, uint32_t w_halo,
      cudaStream_t stream);
  template <typename T>
  void ghost_reduce_edges_nchw(
      T* padded_buf,
      uint32_t batch_size, uint32_t channels,
      uint32_t h_base, uint32_t w_base,
      uint32_t h_halo, uint32_t w_halo,
      cudaStream_t stream);
  template <typename T>
  void ghost_reduce_corners_nchw(
      T* padded_buf,
      uint32_t batch_size, uint32_t channels,
      uint32_t h_base, uint32_t w_base,
      uint32_t h_halo, uint32_t w_halo,
      cudaStream_t stream);

  uint32_t rank;
  uint32_t num_ranks;

  uint32_t u;
  uint32_t v;
  uint32_t dim_u;
  uint32_t dim_v;

  MultiGPUComm2D* dev_comm;
  MultiGPUComm2D* peer_dev_comms[ARRAYDIFF_MAX_PEERS];
};

/*class MultiGPUComm {
public:
  static void InitAll(MultiGPUComm* comms, size_t num_procs);

  explicit MultiGPUComm(size_t num_procs);
  //~MultiGPUComm();

  void all_gather(size_t send_rank, const void* send_buf, uint32_t send_count, ncclDataType_t datatype, void** recv_bufs, cudaStream_t stream);
  void all_gather_batch(size_t send_rank, const void* send_buf, uint32_t send_count, uint32_t batch_count, ncclDataType_t datatype, void** recv_bufs, cudaStream_t stream);
  void reduce_scatter(const void* send_buf, void* recv_buf, uint32_t recv_count, ncclDataType_t datatype, cudaStream_t stream);
  void reduce_scatter_batch(const void* send_buf, void* recv_buf, uint32_t recv_count, uint32_t batch_count, ncclDataType_t datatype, cudaStream_t stream);

  size_t _nccl_datatype_size(ncclDataType_t datatype);

private:
  size_t rank_;
  size_t num_procs_;
};*/

} // namespace arraydiff

#endif
