CONFIG_FILE := Makefile.config
include $(CONFIG_FILE)

BUILD_PREFIX ?= $(shell pwd)

LIBRARY_INCLUDE_DIR := include
LIBRARY_SRC_DIR := src

CUDA_BIN_DIR := $(CUDA_ROOT)/bin
CUDA_INCLUDE_DIR := $(CUDA_ROOT)/include
CUDA_LIB_DIR := $(CUDA_ROOT)/lib64

CUDNN_INCLUDE_DIR := $(CUDNN_ROOT)/include
CUDNN_LIB_DIR := $(CUDNN_ROOT)/lib64

NCCL_INCLUDE_DIR := $(NCCL_ROOT)/include
NCCL_LIB_DIR := $(NCCL_ROOT)/lib

#BOOST_INCLUDE_DIR := $(BOOST_ROOT)/include
#BOOST_LIB_DIR := $(BOOST_ROOT)/lib

#OPENCV_INCLUDE_DIR := $(OPENCV_ROOT)/include
#OPENCV_LIB_DIR := $(OPENCV_ROOT)/lib

AR ?= ar
CXX ?= g++
NVCC := $(CUDA_BIN_DIR)/nvcc

INCLUDE_DIRS := $(LIBRARY_INCLUDE_DIR) $(NCCL_INCLUDE_DIR) $(CUDNN_INCLUDE_DIR) $(CUDA_INCLUDE_DIR) #$(BOOST_INCLUDE_DIR) $(OPENCV_INCLUDE_DIR)
INCLUDE_FLAGS := $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))

TOOL_LINK_DIRS := $(shell pwd) $(NCCL_LIB_DIR) $(CUDNN_LIB_DIR) $(CUDA_LIB_DIR) #$(BOOST_LIB_DIR) $(OPENCV_LIB_DIR)
TOOL_LINK_LIBS := arraydiff_cuda_static cudart cublas cudnn nccl opencv_core opencv_highgui opencv_imgproc
TOOL_LINK_FLAGS := $(foreach linkdir,$(TOOL_LINK_DIRS),-L$(linkdir)) $(foreach linklib,$(TOOL_LINK_LIBS),-l$(linklib))

CXXFLAGS := -O2 -std=c++14 -fPIC -fno-strict-aliasing -pthread -g -Wall -Wextra $(INCLUDE_FLAGS)
NVCCFLAGS := -ccbin=$(CXX) -O2 -std=c++11 -Xcompiler -fPIC -Xcompiler -pthread -G $(NVCC_CUDA_ARCHS) $(INCLUDE_FLAGS)

LIBRARY_CXX_HEADERS := $(shell find $(LIBRARY_INCLUDE_DIR) -name "*.hh")
LIBRARY_CXX_SRCS := $(shell find $(LIBRARY_SRC_DIR) -name "*.cc")
LIBRARY_CU_HEADERS := $(shell find $(LIBRARY_INCLUDE_DIR) -name "*.cuh")
LIBRARY_CU_SRCS := $(shell find $(LIBRARY_SRC_DIR) -name "*.cu")

LIBRARY_CXX_OBJS := ${LIBRARY_CXX_SRCS:.cc=.o}
LIBRARY_CU_OBJS := ${LIBRARY_CU_SRCS:.cu=.o}

LIBRARY_NAME := arraydiff_cuda
SHARED_LIBRARY_TARGET := lib$(LIBRARY_NAME).so
STATIC_LIBRARY_TARGET := lib$(LIBRARY_NAME)_static.a

TOOL_TARGETS := check_mem.tool \
                check_nccl.tool \
                microbench.tool \
                microbench_atomic.tool \
                microbench_conv.tool \
                microbench_spatial.tool \
                microbench_spatial_conv.tool \
                test_cityscapes.tool \
                test_imagenet.tool \
                train_imagenet.tool \
                train_imagenet_resnet50.tool
                #check_mpi.tool \
                #train_imagenet_mpi.tool

TARGETS := $(STATIC_LIBRARY_TARGET) $(TOOL_TARGETS)

.PHONY: all clean build prepare_build compile_targets

all: build

%.o: %.cc $(LIBRARY_CXX_HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

%.o: %.cu $(LIBRARY_CXX_HEADERS) $(LIBRARY_CU_HEADERS)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ -c $<

$(STATIC_LIBRARY_TARGET): $(LIBRARY_CU_OBJS) $(LIBRARY_CXX_OBJS)
	$(AR) cr $@ $^

%.tool: tools/%.cc $(STATIC_LIBRARY_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $< $(TOOL_LINK_FLAGS)

%.tool: tools/%.cu $(STATIC_LIBRARY_TARGET)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH) -o $@ $< $(TOOL_LINK_FLAGS)

clean:
	rm -f $(TARGETS) $(LIBRARY_CXX_OBJS) $(LIBRARY_CU_OBJS)

build: prepare_build compile_targets

prepare_build:

compile_targets: $(TARGETS)
