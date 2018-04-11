#include "arraydiff/io/tar.hh"

#include <cstdint>
//#include <iostream>
#include <string>
#include <vector>

namespace arraydiff {
namespace io {

using std::string;
using std::vector;

struct PartialTarHeader {
  char name[100];
  char _unused_1[24];
  char size[12];
  char _unused_2[20];
  char type;
  char _unused_3[355];
} __attribute__ ((packed));

TarMemoryFile::TarMemoryFile(const uint8_t* mem_addr, size_t mem_sz)
  : mem_addr_(mem_addr), mem_sz_(mem_sz)
{
  const size_t TAR_BLOCK_SIZE = 512;

  size_t cursor = 0;
  for (;;) {
    // If there are two all-zero blocks in a row, we are at EOF.
    bool not_empty_header = false;
    for (size_t header_offset = 0; header_offset < TAR_BLOCK_SIZE; header_offset++) {
      if (0 != mem_addr_[cursor + header_offset]) {
        not_empty_header = true;
        break;
      }
    }
    if (!not_empty_header) {
      // FIXME: check for the second all-zero block.
      break;
    }

    // Read header at the current cursor.
    // FIXME: need to check contiguous or sparse header.
    const PartialTarHeader* header = reinterpret_cast<const PartialTarHeader*>(mem_addr_ + cursor);
    string entry_name(header->name, 100);
    string entry_data_size_octal(header->size, 12);
    size_t entry_data_sz = std::stoull(entry_data_size_octal, NULL, 8);

    // Append a new archive entry.
    if ('\0' == header->type || '0' == header->type) {
      TarEasyEntry entry = {
        .header_pos = cursor,
        .data_pos = cursor + TAR_BLOCK_SIZE,
        .data_ptr = mem_addr_ + cursor + TAR_BLOCK_SIZE,
        .data_size = entry_data_sz,
        .noncontiguous = false,
        .filename = entry_name,
      };
      entries_.push_back(entry);
    } else {
      //std::clog << "DEBUG: TarMemoryFile: non-file type: " << header->type << std::endl;
    }

    // Skip ahead to the next header.
    size_t entry_num_blocks = (entry_data_sz + TAR_BLOCK_SIZE - 1) / TAR_BLOCK_SIZE + 1;
    size_t entry_total_footprint = entry_num_blocks * TAR_BLOCK_SIZE;
    cursor += entry_total_footprint;
  }
  //assert(cursor == mem_sz); // FIXME
}

TarMemoryFile::~TarMemoryFile() {
}

} // namespace io
} // namespace arraydiff
