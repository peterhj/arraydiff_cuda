#ifndef ARRAYDIFF_IO_IMAGENET_HH
#define ARRAYDIFF_IO_IMAGENET_HH

#include "arraydiff/io/mmap.hh"
#include "arraydiff/io/pipeline.hh"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arraydiff {
namespace io {

using std::pair;
using std::string;
using std::unordered_map;
using std::vector;

class ImagenetMetadata {
public:
  /**
    * Open ILSVRC2012 devkit metadata. A prerequisite is running a preprocessing
    * script to generate two ordered text files, one containing a list of
    * WordNet IDs, the other containing a companion list of ILSVRC2012 IDs.
    *
    * @param ilsvrc2012_id_path: Path to a text file where the i-th line
    *     contains the ILSVRC2012 ID (from 1 to 1000, inclusive) corresponding
    *     to the i-th wordnet ID in the previous `wordnet_id_path` text file.
    * @param wordnet_id_path: Path to a text file where each line contains a
    *     WordNet ID from the "meta.mat" file from the ILSVRC2012 devkit.
    * @param val_ground_truth_path: Path to the raw validation ground truth file
    *     from the devkit, usually "ILSVRC2012_validation_ground_truth.txt".
    */
  ImagenetMetadata(const string& ilsvrc2012_id_path, const string& wordnet_id_path, const string& val_ground_truth_path);

  unordered_map<string, int> wordnet_id_to_rank_;
  unordered_map<int, string> rank_to_wordnet_id_;
  unordered_map<string, int> ilsvrc2012_id_to_rank_;
  unordered_map<int, string> rank_to_ilsvrc2012_id_;
  vector<string> val_ground_truth_;
};

class ImagenetEntry {
public:
  size_t pos;
  size_t size;
  uint32_t label;
};

class ImagenetTrainTar : public virtual RandomAccessData<pair<Datum, uint32_t> > {
public:
  /**
    * Open ILSVRC2012 training set tar file.
    *
    * @param metadata: Previously opened metadata file (see above).
    * @param source: Path to the tar file.
    */
  ImagenetTrainTar(ImagenetMetadata metadata, const string& source);

  virtual size_t size() const;
  virtual pair<Datum, uint32_t> at(size_t index) const;

private:
  ImagenetMetadata metadata_;
  MmapROFile mapped_file_;
  vector<ImagenetEntry> entries_;
};

class ImagenetValTar : public virtual RandomAccessData<pair<Datum, uint32_t> > {
public:
  /**
    * Open ILSVRC2012 validation set tar file.
    *
    * @param metadata: Previously opened metadata file (see above).
    * @param source: Path to the tar file.
    */
  ImagenetValTar(ImagenetMetadata metadata, const string& source);

  virtual size_t size() const;
  virtual pair<Datum, uint32_t> at(size_t index) const;

private:
  ImagenetMetadata metadata_;
  MmapROFile mapped_file_;
  vector<ImagenetEntry> entries_;
};

} // namespace io
} // namespace arraydiff

#endif
