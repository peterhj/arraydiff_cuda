#include <nccl.h>

#include <iostream>

int main(int argc, const char** argv) {
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  std::clog << "DEBUG: num devices: " << num_devices << std::endl;

  std::clog << "DEBUG: init nccl..." << std::endl;
  //ncclComm_t* comms = new ncclComm_t[num_devices];
  ncclComm_t* comms = (ncclComm_t*)malloc(sizeof(ncclComm_t) * num_devices);
  auto e = ncclCommInitAll(comms, num_devices, NULL);
  std::clog << "DEBUG: init nccl done: " << e << std::endl;
}
