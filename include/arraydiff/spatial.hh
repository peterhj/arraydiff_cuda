#ifndef ARRAYDIFF_SPATIAL_HH
#define ARRAYDIFF_SPATIAL_HH

#include "arraydiff/context.hh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#define ARRAYDIFF_MAX_PEERS 8

namespace arraydiff {

enum Spatial2DOperation {
  SPATIAL2D_OP_HXEXCH   = 0,
  SPATIAL2D_OP_HXREDUCE = 1,
};

enum Spatial2DTensorFormat {
  SPATIAL2D_TENSOR_NCHW = 0,
  SPATIAL2D_TENSOR_NHWC = 1,
};

enum Spatial2DDataType {
  SPATIAL2D_DATA_FLOAT  = 0,
  SPATIAL2D_DATA_DOUBLE = 1,
  SPATIAL2D_DATA_HALF   = 2,
};

class Spatial2DArg {
public:
  Spatial2DOperation operation;
  Spatial2DTensorFormat tensor_format;
  Spatial2DDataType data_type;
  uint32_t halo_pad;
  uint32_t base_width;
  uint32_t base_height;
  uint32_t num_channels;
  uint32_t batch_size;
  void* buffer;
  volatile uint64_t epoch;
};

class Spatial2DComm {
public:
  static void InitAll(Spatial2DComm* comms, uint32_t num_w_ranks, uint32_t num_h_ranks);

  Spatial2DComm() {}
  ~Spatial2DComm() {}

  void group_start();
  void group_end(GPUConn& conn);
  void halo_exchange(
      void* buf,
      uint32_t halo_pad,
      uint32_t base_width,
      uint32_t base_height,
      uint32_t num_channels,
      uint32_t batch_size,
      Spatial2DTensorFormat tensor_format,
      Spatial2DDataType data_type);
  void halo_reduce(
      void* buf,
      uint32_t halo_pad,
      uint32_t base_width,
      uint32_t base_height,
      uint32_t num_channels,
      uint32_t batch_size,
      Spatial2DTensorFormat tensor_format,
      Spatial2DDataType data_type);

  uint32_t rank;
  uint32_t rank_w;
  uint32_t rank_h;
  uint32_t num_ranks;
  uint32_t num_w_ranks;
  uint32_t num_h_ranks;

  uint64_t epoch;
  volatile Spatial2DArg arg;

  Spatial2DComm* peer_comms[ARRAYDIFF_MAX_PEERS];

  //Spatial2DComm* dev_comm;
  //Spatial2DComm* peer_dev_comms[ARRAYDIFF_MAX_PEERS];
};

} // namespace arraydiff

#endif
