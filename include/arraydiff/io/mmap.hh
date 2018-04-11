#ifndef ARRAYDIFF_IO_MMAP_HH
#define ARRAYDIFF_IO_MMAP_HH

#include <string>

namespace arraydiff {
namespace io {

using std::string;

class MmapROFile {
public:
  explicit MmapROFile(const string& source);
  ~MmapROFile();

  const uint8_t* const_ptr() const {
    return reinterpret_cast<const uint8_t*>(mem_addr_);
  }
  size_t size() const {
    return mem_sz_;
  }

private:
  int mem_fd_;
  size_t mem_sz_;
  void* mem_addr_;
};

} // namespace io
} // namespace arraydiff

#endif
