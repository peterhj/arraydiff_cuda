#include "arraydiff/io/mmap.hh"

#include <string>

extern "C" {
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
}

namespace arraydiff {
namespace io {

MmapROFile::MmapROFile(const string& source) {
  mem_fd_ = -1;
  mem_sz_ = 0;
  mem_addr_ = NULL;

  mem_fd_ = open(source.c_str(), O_RDONLY);
  if (-1 == mem_fd_) {
    //LOG(FATAL) << "MmapROFile failed to open source file";
    return;
  }

  struct stat stat_buf;
  if (-1 == fstat(mem_fd_, &stat_buf)) {
    //LOG(FATAL) << "MmapROFile failed to query source file metadata";
    return;
  }
  mem_sz_ = stat_buf.st_size;

  mem_addr_ = mmap(NULL, mem_sz_, PROT_READ, MAP_SHARED, mem_fd_, 0);
  if (NULL == mem_addr_) {
    //LOG(FATAL) << "MmapROFile failed to memory map source file";
    return;
  }

  /*if (-1 == madvise(mem_addr_, mem_sz_, MADV_RANDOM)) {
    //LOG(FATAL) << "MmapROFile failed to call madvise";
    return;
  }*/
}

MmapROFile::~MmapROFile() {
  // TODO: check non nulls etc.
  munmap(mem_addr_, mem_sz_);
  close(mem_fd_);
}

} // namespace io
} // namespace arraydiff
