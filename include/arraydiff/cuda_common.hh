#ifndef ARRAYDIFF_CUDA_COMMON_HH
#define ARRAYDIFF_CUDA_COMMON_HH

#include <iostream>

#define CUDA_CHECK(e) ({ cudaError_t status = (e); if (cudaSuccess != status) { std::clog << "FATAL CUDA ERROR: " << cudaGetErrorName(status) << " : " << cudaGetErrorString(status) << std::endl; assert(0 && "fatal cuda error"); } })
#define CUBLAS_CHECK(e) ({ cublasStatus_t status = (e); assert(CUBLAS_STATUS_SUCCESS == status); })
#define CUDNN_CHECK(e) ({ cudnnStatus_t status = (e); if (CUDNN_STATUS_SUCCESS != status) { std::clog << "FATAL CUDNN ERROR: " << cudnnGetErrorString(status) << std::endl; assert(0 && "fatal cudnn error"); } })
#define NCCL_CHECK(e) ({ ncclResult_t status = (e); assert(ncclSuccess == status); })

#endif
