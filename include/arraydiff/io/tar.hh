#ifndef ARRAYDIFF_IO_TAR_HH
#define ARRAYDIFF_IO_TAR_HH

#include <cstdint>
#include <string>
#include <vector>

namespace arraydiff {
namespace io {

using std::string;
using std::vector;

class TarEasyEntry {
public:
  size_t header_pos;
  size_t data_pos;
  const uint8_t* data_ptr;
  size_t data_size;
  bool noncontiguous;
  string filename;
};

class TarMemoryFile {
public:
  TarMemoryFile(const uint8_t* addr, size_t sz);
  ~TarMemoryFile();

  const vector<TarEasyEntry>& const_entries() const {
    return entries_;
  }

private:
  const uint8_t* mem_addr_;
  size_t mem_sz_;
  vector<TarEasyEntry> entries_;
};

} // namespace io
} // namespace arraydiff

#endif
