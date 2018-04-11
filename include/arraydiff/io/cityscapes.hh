#ifndef ARRAYDIFF_IO_CITYSCAPES_HH
#define ARRAYDIFF_IO_CITYSCAPES_HH

#include "arraydiff/io/mmap.hh"
#include "arraydiff/io/pipeline.hh"

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arraydiff {
namespace io {

using std::map;
using std::pair;
using std::string;
using std::unordered_map;
using std::vector;

uint32_t cityscapes_convert_raw_label(uint32_t raw_id);
uint32_t cityscapes_convert_preproc_label(uint32_t preproc_id);

class CityscapesTarIndex {
public:
  unordered_map<string, size_t> prefix_to_idx;
};

class CityscapesKey {
public:
  bool operator<(const CityscapesKey& rhs) const {
    const CityscapesKey& lhs = *this;
    return std::tie(lhs.split, lhs.city, lhs.seq_num, lhs.frame_num)
         < std::tie(rhs.split, rhs.city, rhs.seq_num, rhs.frame_num);
  }

  string type;
  string split;
  string city;
  size_t seq_num;
  size_t frame_num;
};

class CityscapesEntry {
public:
  size_t pos;
  size_t size;
  CityscapesKey key;
};

class CityscapesLabelsEntry {
public:
  CityscapesEntry class_label;
  CityscapesEntry instance_label;
};

class CityscapesLabelsTar {
public:
  explicit CityscapesLabelsTar(const string& path);
  ~CityscapesLabelsTar() {}

  size_t size() const {
    return entries_.size();
  }

  const vector<CityscapesEntry>& const_entries() const {
    return entries_;
  }

  const map<CityscapesKey, size_t>& const_keys() const {
    return keys_;
  }

  const uint8_t* const_ptr() const {
    return mapped_file_.const_ptr();
  }

private:
  MmapROFile mapped_file_;
  //vector<CityscapesLabelsEntry> entries_;
  vector<CityscapesEntry> entries_;
  map<CityscapesKey, size_t> keys_;
};

class CityscapesImagesTar {
public:
  explicit CityscapesImagesTar(const string& path);
  ~CityscapesImagesTar() {}

  size_t size() const {
    return entries_.size();
  }

  const vector<CityscapesEntry>& const_entries() const {
    return entries_;
  }

  const map<CityscapesKey, size_t>& const_keys() const {
    return keys_;
  }

  const uint8_t* const_ptr() const {
    return mapped_file_.const_ptr();
  }

private:
  MmapROFile mapped_file_;
  vector<CityscapesEntry> entries_;
  map<CityscapesKey, size_t> keys_;
};

class CityscapesData : public virtual RandomAccessData<pair<Datum, Datum> > {
public:
  /**
   * Open a pair of Cityscapes image and annotation tar files.
   *
   * @param images_path: Path to the images tar file.
   * @param labels_path: Path to the fine annotations tar file.
   */
  CityscapesData(const string& images_path, const string& labels_path);
  virtual ~CityscapesData() {}

  virtual size_t size() const;
  virtual pair<Datum, Datum> at(size_t index) const;

private:
  CityscapesImagesTar images_;
  CityscapesLabelsTar labels_;
};

} // namespace io
} // namespace arraydiff

#endif
