#include "arraydiff/io/cityscapes.hh"
#include "arraydiff/io/mmap.hh"
#include "arraydiff/io/tar.hh"
#include "arraydiff/str_util.hh"

#include <cassert>
#include <iostream>
#include <map>
#include <string>
#include <utility>

namespace arraydiff {
namespace io {

using std::string;

uint32_t cityscapes_convert_raw_label(uint32_t raw_id) {
  // TODO: See: <https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py>.
  switch (raw_id) {
    // These raw labels exactly correspond to "train labels".
    case 7:     return 0;
    case 8:     return 1;
    case 11:    return 2;
    case 12:    return 3;
    case 13:    return 4;
    case 17:    return 5;
    case 19:    return 6;
    case 20:    return 7;
    case 21:    return 8;
    case 22:    return 9;
    case 23:    return 10;
    case 24:    return 11;
    case 25:    return 12;
    case 26:    return 13;
    case 27:    return 14;
    case 28:    return 15;
    case 31:    return 16;
    case 32:    return 17;
    case 33:    return 18;
    // TODO: Custom "extra labels".
    // These are approximations.
    case 18:    return 5;   // "polegroup" (18) => "pole" (5).
    // These are "void" labels.
    default:    return -1;
  }
}

uint32_t cityscapes_convert_preproc_label(uint32_t preproc_id) {
  // TODO
  assert(0 && "unimplemented");
}

CityscapesLabelsTar::CityscapesLabelsTar(const string& path)
  : mapped_file_(path), entries_(), keys_()
{
  TarMemoryFile mapped_tar(mapped_file_.const_ptr(), mapped_file_.size());
  auto top_entries = mapped_tar.const_entries();
  for (auto top_iter = top_entries.begin(); top_iter != top_entries.end(); ++top_iter) {
    auto top_entry = *top_iter;
    auto toks = split(top_entry.filename, '/');
    assert(4 == toks.size());
    string cs_type = toks.at(0);
    string cs_split = toks.at(1);
    string cs_city = toks.at(2);
    auto name_toks = split(toks.at(3), '_');
    assert(5 == name_toks.size());
    size_t cs_seqnum = std::stoull(name_toks.at(1));
    size_t cs_framenum = std::stoull(name_toks.at(2));
    string cs_ext = name_toks.at(4);
    if (0 == cs_ext.compare(0, 12, "labelIds.png", 0, 12)) {
      const CityscapesEntry label_entry = {
        .pos  = top_entry.data_pos,
        .size = top_entry.data_size,
        .key  = {
          .type       = cs_type,
          .split      = cs_split,
          .city       = cs_city,
          .seq_num    = cs_seqnum,
          .frame_num  = cs_framenum,
        },
      };
      size_t idx = entries_.size();
      entries_.push_back(label_entry);
      auto res = keys_.emplace(std::make_pair(label_entry.key, idx));
      assert(res.second);
    }
  }
  std::clog << "DEBUG: cityscapes labels: size: " << entries_.size() << std::endl;
}

CityscapesImagesTar::CityscapesImagesTar(const string& path)
  : mapped_file_(path), entries_()
{
  TarMemoryFile mapped_tar(mapped_file_.const_ptr(), mapped_file_.size());
  auto top_entries = mapped_tar.const_entries();
  for (auto top_iter = top_entries.begin(); top_iter != top_entries.end(); ++top_iter) {
    auto top_entry = *top_iter;
    auto toks = split(top_entry.filename, '/');
    assert(4 == toks.size());
    string cs_type = toks.at(0);
    string cs_split = toks.at(1);
    string cs_city = toks.at(2);
    auto name_toks = split(toks.at(3), '_');
    assert(4 == name_toks.size());
    size_t cs_seq_num = std::stoull(name_toks.at(1));
    size_t cs_frame_num = std::stoull(name_toks.at(2));
    const CityscapesEntry image_entry = {
      .pos  = top_entry.data_pos,
      .size = top_entry.data_size,
      .key  = {
        .type       = cs_type,
        .split      = cs_split,
        .city       = cs_city,
        .seq_num    = cs_seq_num,
        .frame_num  = cs_frame_num,
      },
    };
    size_t idx = entries_.size();
    entries_.push_back(image_entry);
    auto res = keys_.emplace(std::make_pair(image_entry.key, idx));
    assert(res.second);
  }
  std::clog << "DEBUG: cityscapes images: size: " << entries_.size() << std::endl;
}

CityscapesData::CityscapesData(const string& images_path, const string& labels_path)
  : images_(images_path), labels_(labels_path)
{
}

size_t CityscapesData::size() const {
  return images_.size();
}

pair<Datum, Datum> CityscapesData::at(size_t index) const {
  //std::clog << "DEBUG: cityscapes data: query image entry" << std::endl;
  const CityscapesEntry& image_entry = images_.const_entries().at(index);
  //std::clog << "DEBUG: cityscapes data: query image key" << std::endl;
  size_t image_index = images_.const_keys().at(image_entry.key);
  assert(image_index == index);
  //std::clog << "DEBUG: cityscapes data: label keys: " << labels_.const_keys().size() << std::endl;
  //std::clog << "DEBUG: cityscapes data: query label key" << std::endl;
  size_t label_index = labels_.const_keys().at(image_entry.key);
  //std::clog << "DEBUG: cityscapes data: query label entry" << std::endl;
  const CityscapesEntry& label_entry = labels_.const_entries().at(label_index);
  //std::clog << "DEBUG: cityscapes data: make pair" << std::endl;
  return std::make_pair(
      Datum(images_.const_ptr() + image_entry.pos, image_entry.size),
      Datum(labels_.const_ptr() + label_entry.pos, label_entry.size));
}

} // namespace io
} // namespace arraydiff
