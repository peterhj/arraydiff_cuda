# arraydiff_cuda

<code>arraydiff_cuda</code> is a prototype autodiff framework. Its main feature
is a preliminary implementation of <em>spatially parallel convolutions</em>,
described in this report: (https://openreview.net/forum?id=S1Yt0d1vG).

Note that more up-to-date halo/ring kernels have since been implemented; c.f.
(https://github.com/peterhj/gpudevicemem/blob/master/routines_gpu/halo_ring.cu).

## Requirements

* Requires CUDA 8.0, cuDNN 6.0, NCCL 2.0.
* Known to compile with g++ 5.4 on Ubuntu 16.04.

## Building

Build with <code>make</code>.
