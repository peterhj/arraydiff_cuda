#include "arraydiff/io/imagenet.hh"
#include "arraydiff/io/mmap.hh"
#include "arraydiff/io/tar.hh"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace arraydiff {
namespace io {

using std::string;

ImagenetMetadata::ImagenetMetadata(const string& ilsvrc2012_id_path, const string& wordnet_id_path, const string& val_ground_truth_path) {
  string buf;
  std::ifstream ilsvrc2012_id_file(ilsvrc2012_id_path);
  int ilsvrc2012_id_rank = 0;
  while (std::getline(ilsvrc2012_id_file, buf)) {
    ilsvrc2012_id_to_rank_.emplace(buf, ilsvrc2012_id_rank);
    rank_to_ilsvrc2012_id_.emplace(ilsvrc2012_id_rank, buf);
    ++ilsvrc2012_id_rank;
  }
  std::ifstream wordnet_id_file(wordnet_id_path);
  int wordnet_id_rank = 0;
  while (std::getline(wordnet_id_file, buf)) {
    wordnet_id_to_rank_.emplace(buf, wordnet_id_rank);
    rank_to_wordnet_id_.emplace(wordnet_id_rank, buf);
    ++wordnet_id_rank;
  }
  std::ifstream ground_truth_file(val_ground_truth_path);
  while (std::getline(ground_truth_file, buf)) {
    val_ground_truth_.push_back(buf);
  }
}

ImagenetTrainTar::ImagenetTrainTar(ImagenetMetadata metadata, const string& source)
  : metadata_(metadata), mapped_file_(source)
{
  TarMemoryFile mapped_tar(mapped_file_.const_ptr(), mapped_file_.size());
  auto top_entries = mapped_tar.const_entries();
  for (auto top_iter = top_entries.begin(); top_iter != top_entries.end(); ++top_iter) {
    auto top_entry = *top_iter;
    if (top_entry.filename.size() < 4) {
      std::clog << "WARNING: imagenet inner tar file has bad filename, file is probably corrupt" << std::endl;
      continue;
    }
    const string wordnet_id = top_entry.filename.substr(0, std::strlen(top_entry.filename.c_str()) - 4);
    int category_rank = -1;
    auto rank_search = metadata_.wordnet_id_to_rank_.find(wordnet_id);
    if (rank_search != metadata_.wordnet_id_to_rank_.end()) {
      category_rank = rank_search->second;
    } else {
      std::clog << "WARNING: imagenet inner tar file has no rank; is your metadata correct?" << std::endl;
      continue;
    }
    const uint32_t label = static_cast<uint32_t>(std::stoi(metadata_.rank_to_ilsvrc2012_id_.at(category_rank)) - 1);
    assert(label < 1000U);
    //std::clog << "DEBUG: found imagenet inner tar: wordnet id: " << wordnet_id << " label: " << label << std::endl;
    TarMemoryFile category_tar(top_entry.data_ptr, top_entry.data_size);
    auto category_entries = category_tar.const_entries();
    for (auto category_iter = category_entries.begin(); category_iter != category_entries.end(); ++category_iter) {
      const auto category_entry = *category_iter;
      const ImagenetEntry image_entry = {
        .pos    = top_entry.data_pos + category_entry.data_pos,
        .size   = category_entry.data_size,
        .label  = label
      };
      entries_.push_back(image_entry);
    }
  }
  std::clog << "DEBUG: imagenet train entries: " << entries_.size() << std::endl;
}

size_t ImagenetTrainTar::size() const {
  return entries_.size();
}

pair<Datum, uint32_t> ImagenetTrainTar::at(size_t index) const {
  const ImagenetEntry& entry = entries_.at(index);
  return pair<Datum, uint32_t>(Datum(mapped_file_.const_ptr() + entry.pos, entry.size), entry.label);
}

ImagenetValTar::ImagenetValTar(ImagenetMetadata metadata, const string& source)
  : metadata_(metadata), mapped_file_(source)
{
  TarMemoryFile mapped_tar(mapped_file_.const_ptr(), mapped_file_.size());
  auto top_entries = mapped_tar.const_entries();
  for (auto top_iter = top_entries.begin(); top_iter != top_entries.end(); ++top_iter) {
    const auto top_entry = *top_iter;
    const int entry_index = std::stoi(top_entry.filename.substr(15, 23)) - 1;
    const uint32_t label = static_cast<uint32_t>(std::stoi(metadata_.val_ground_truth_.at(entry_index)) - 1);
    //std::clog << "DEBUG: imagenet val entry: filename: " << top_entry.filename.c_str() << " label: " << label << std::endl;
    assert(label < 1000U);
    const ImagenetEntry image_entry = {
      .pos    = top_entry.data_pos,
      .size   = top_entry.data_size,
      .label  = label
    };
    entries_.push_back(image_entry);
  }
  std::clog << "DEBUG: imagenet val entries: " << entries_.size() << std::endl;
}

size_t ImagenetValTar::size() const {
  return entries_.size();
}

pair<Datum, uint32_t> ImagenetValTar::at(size_t index) const {
  const ImagenetEntry& entry = entries_.at(index);
  return pair<Datum, uint32_t>(Datum(mapped_file_.const_ptr() + entry.pos, entry.size), entry.label);
}

} // namespace io
} // namespace arraydiff
