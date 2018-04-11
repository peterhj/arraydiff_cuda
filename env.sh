#!/bin/sh
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64"
export LD_LIBRARY_PATH="/opt/cudnn-7.0.5-cuda-9.0/lib64:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/opt/nccl-2.1.2-cuda-9.0/lib:${LD_LIBRARY_PATH}"
