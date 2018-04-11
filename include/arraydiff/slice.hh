#ifndef ARRAYDIFF_SLICE_HH
#define ARRAYDIFF_SLICE_HH

#include <cstddef>

namespace arraydiff {

template <typename T>
class Slice {
public:
  Slice(T* ptr, size_t size)
    : ptr(ptr), size(size) {}
  ~Slice() {}

  T* ptr;
  size_t size;
};

} // namespace arraydiff

#endif
