#ifndef ARRAYDIFF_IO_DUMMY_HH
#define ARRAYDIFF_IO_DUMMY_HH

#include "arraydiff/io/pipeline.hh"

#include <utility>

namespace arraydiff {
namespace io {

using std::make_pair;
using std::pair;

template <typename T>
class DummyData : public virtual RandomAccessData<pair<MutDatum<T>, uint32_t> > {
public:
  DummyData(size_t width, size_t height, size_t channels, uint32_t num_labels, size_t size)
    : width_(width), height_(height), channels_(channels), num_labels_(num_labels), size_(size) {}
  ~DummyData() {}

  virtual size_t size() const {
    return size_;
  }

  virtual pair<MutDatum<T>, uint32_t> at(size_t index) const {
    //assert(0 && "unimplemented");
    auto datum = MutDatum<T>::Alloc(width_ * height_ * channels_);
    auto label = index % num_labels_;
    return make_pair(datum, label);
  }

private:
  size_t width_;
  size_t height_;
  size_t channels_;
  uint32_t num_labels_;
  size_t size_;
};

} // namespace io
} // namespace arraydiff

#endif
