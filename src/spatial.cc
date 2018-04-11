#include "arraydiff/spatial.hh"
#include "arraydiff/context.hh"
#include "arraydiff/cuda_common.hh"
#include "arraydiff/kernels.hh"

#include <cassert>

namespace arraydiff {

void Spatial2DComm::InitAll(Spatial2DComm* comms, uint32_t num_w_ranks, uint32_t num_h_ranks) {
  int num_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&num_devices));
  for (int dev = 0; dev < num_devices; ++dev) {
    for (int dev2 = 0; dev2 < num_devices; ++dev2) {
      if (dev != dev2) {
        int can_access = -1;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, dev, dev2));
        CUDA_CHECK(cudaSetDevice(dev));
        auto status = cudaDeviceEnablePeerAccess(dev2, 0);
        if (cudaErrorPeerAccessAlreadyEnabled == status) {
          // OK.
        } else {
          CUDA_CHECK(status);
        }
        //std::clog << "DEBUG: p2p enabled? " << dev << " <- " << dev2 << " " << can_access << " " << res1 << " " << res2 << std::endl;
      }
    }
  }
  CUDA_CHECK(cudaSetDevice(0));

  uint32_t num_ranks = num_w_ranks * num_h_ranks;
  for (uint32_t r_w = 0; r_w < num_w_ranks; ++r_w) {
    for (uint32_t r_h = 0; r_h < num_h_ranks; ++r_h) {
      uint32_t r = r_w + num_w_ranks * r_h;
      comms[r].rank = r;
      comms[r].rank_w = r_w;
      comms[r].rank_h = r_h;
      comms[r].num_ranks = num_ranks;
      comms[r].num_w_ranks = num_w_ranks;
      comms[r].num_h_ranks = num_h_ranks;
      comms[r].epoch = 0;
      comms[r].arg.buffer = NULL;
      comms[r].arg.epoch = 0;
    }
  }
  for (uint32_t r = 0; r < num_ranks; ++r) {
    for (uint32_t r2 = 0; r2 < num_ranks; ++r2) {
      comms[r].peer_comms[r2] = &comms[r2];
    }
  }
}

void Spatial2DComm::group_start() {
  ++epoch;
  arg.buffer = NULL;
}

void Spatial2DComm::group_end(GPUConn& conn) {
  for (;;) {
    bool wait_epoch = false;
    for (uint32_t r = 0; r < num_ranks; ++r) {
      if (epoch != peer_comms[r]->arg.epoch) {
        wait_epoch = true;
        break;
      }
    }
    if (!wait_epoch) {
      break;
    }
  }
  for (uint32_t r = 0; r < num_ranks; ++r) {
    assert(peer_comms[r]->arg.buffer != NULL);
  }

  if (arg.operation     == SPATIAL2D_OP_HXEXCH &&
      arg.tensor_format == SPATIAL2D_TENSOR_NCHW &&
      arg.data_type     == SPATIAL2D_DATA_FLOAT)
  {
    if (num_w_ranks == 1 && num_h_ranks == 1) {
      // Do nothing.
    } else if (num_w_ranks == 1 && num_h_ranks == 2) {
      spatial2d_1x2_halo_exch_nchw<float>(
          rank_h,
          arg.halo_pad,
          arg.base_width,
          arg.base_height,
          arg.num_channels,
          arg.batch_size,
          reinterpret_cast<float*>(peer_comms[0]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[1]->arg.buffer),
          conn.stream());
    } else if (num_w_ranks == 1 && num_h_ranks == 4) {
      spatial2d_1x4_halo_exch_nchw<float>(
          rank_h,
          arg.halo_pad,
          arg.base_width,
          arg.base_height,
          arg.num_channels,
          arg.batch_size,
          reinterpret_cast<float*>(peer_comms[0]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[1]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[2]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[3]->arg.buffer),
          conn.stream());
    } else if (num_w_ranks == 2 && num_h_ranks == 1) {
      spatial2d_2x1_halo_exch_nchw<float>(
          rank_w,
          arg.halo_pad,
          arg.base_width,
          arg.base_height,
          arg.num_channels,
          arg.batch_size,
          reinterpret_cast<float*>(peer_comms[0]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[1]->arg.buffer),
          conn.stream());
    } else if (num_w_ranks == 2 && num_h_ranks == 2) {
      // FIXME: linkage.
      assert(0 && "unimplemented");
      /*spatial2d_2x2_halo_exch_nchw<float>(
          rank_w,
          rank_h,
          arg.halo_pad,
          arg.base_width,
          arg.base_height,
          arg.num_channels,
          arg.batch_size,
          reinterpret_cast<float*>(peer_comms[0]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[1]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[2]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[3]->arg.buffer),
          conn.stream());*/
    } else {
      assert(0 && "unimplemented");
    }
  } else if (
      arg.operation     == SPATIAL2D_OP_HXREDUCE &&
      arg.tensor_format == SPATIAL2D_TENSOR_NCHW &&
      arg.data_type     == SPATIAL2D_DATA_FLOAT)
  {
    if (num_w_ranks == 1 && num_h_ranks == 1) {
      // Do nothing.
    } else if (num_w_ranks == 1 && num_h_ranks == 2) {
      spatial2d_1x2_halo_reduce_nchw<float>(
          rank_h,
          arg.halo_pad,
          arg.base_width,
          arg.base_height,
          arg.num_channels,
          arg.batch_size,
          reinterpret_cast<float*>(peer_comms[0]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[1]->arg.buffer),
          conn.stream());
    } else if (num_w_ranks == 1 && num_h_ranks == 4) {
      spatial2d_1x4_halo_reduce_nchw<float>(
          rank_h,
          arg.halo_pad,
          arg.base_width,
          arg.base_height,
          arg.num_channels,
          arg.batch_size,
          reinterpret_cast<float*>(peer_comms[0]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[1]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[2]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[3]->arg.buffer),
          conn.stream());
    } else if (num_w_ranks == 2 && num_h_ranks == 1) {
      // FIXME: linkage.
      assert(0 && "unimplemented");
      /*spatial2d_2x1_halo_reduce_nchw<float>(
          rank_w,
          arg.halo_pad,
          arg.base_width,
          arg.base_height,
          arg.num_channels,
          arg.batch_size,
          reinterpret_cast<float*>(peer_comms[0]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[1]->arg.buffer),
          conn.stream());*/
    } else if (num_w_ranks == 2 && num_h_ranks == 2) {
      // FIXME: linkage.
      assert(0 && "unimplemented");
      /*spatial2d_2x2_halo_reduce_nchw<float>(
          rank_w,
          rank_h,
          arg.halo_pad,
          arg.base_width,
          arg.base_height,
          arg.num_channels,
          arg.batch_size,
          reinterpret_cast<float*>(peer_comms[0]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[1]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[2]->arg.buffer),
          reinterpret_cast<float*>(peer_comms[3]->arg.buffer),
          conn.stream());*/
    } else {
      assert(0 && "unimplemented");
    }
  } else {
    assert(0 && "unimplemented");
  }
}

void Spatial2DComm::halo_exchange(
    void* buf,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    Spatial2DTensorFormat tensor_format,
    Spatial2DDataType data_type)
{
  arg.operation = SPATIAL2D_OP_HXEXCH;
  arg.tensor_format = tensor_format;
  arg.data_type = data_type;
  arg.halo_pad = halo_pad;
  arg.base_width = base_width;
  arg.base_height = base_height;
  arg.num_channels = num_channels;
  arg.batch_size = batch_size;
  arg.buffer = buf;
  arg.epoch = epoch;
}

void Spatial2DComm::halo_reduce(
    void* buf,
    uint32_t halo_pad,
    uint32_t base_width,
    uint32_t base_height,
    uint32_t num_channels,
    uint32_t batch_size,
    Spatial2DTensorFormat tensor_format,
    Spatial2DDataType data_type)
{
  arg.operation = SPATIAL2D_OP_HXREDUCE;
  arg.tensor_format = tensor_format;
  arg.data_type = data_type;
  arg.halo_pad = halo_pad;
  arg.base_width = base_width;
  arg.base_height = base_height;
  arg.num_channels = num_channels;
  arg.batch_size = batch_size;
  arg.buffer = buf;
  arg.epoch = epoch;
}

} // namespace arraydiff
