#include "arraydiff/io/cityscapes.hh"
#include "arraydiff/io/image.hh"
#include "arraydiff/io/pipeline.hh"

#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#include <map>
#include <memory>
#include <utility>

using namespace arraydiff;
using namespace arraydiff::io;
using std::make_shared;
using std::pair;
using std::shared_ptr;

int main(int argc, const char** argv) {
  shared_ptr<RandomAccessData<pair<Datum, Datum>>> val_dataset =
      make_shared<CityscapesData>(
          "/scratch/phj/data/cityscapes/cityscapes_leftimg_val.tar",
          "/scratch/phj/data/cityscapes/cityscapes_fine_val.tar");

  shared_ptr<RandomAccessData<pair<Datum, Datum>>> train_dataset =
      make_shared<CityscapesData>(
          "/scratch/phj/data/cityscapes/cityscapes_leftimg_train.tar",
          "/scratch/phj/data/cityscapes/cityscapes_fine_train.tar");

  /*shared_ptr<RandomAccessData<pair<Datum, Datum>>> test_dataset =
      make_shared<CityscapesData>(
          "/scratch/phj/data/cityscapes/cityscapes_leftimg_test.tar",
          "/scratch/phj/data/cityscapes/cityscapes_fine_test.tar");*/

  //auto datum = MutDatum<uint32_t>::Alloc(2048UL * 1024UL);
  uint32_t label_min = -1;
  uint32_t label_max = 0;
  std::map<uint32_t, size_t> label_counts;

  for (size_t idx = 0; idx < val_dataset->size(); ++idx) {
    auto item = val_dataset->at(idx);
    cv::Mat img;
    DecodeImage(item.second, img);
    //LabelImageToDatum(img, datum);
    AccumulateLabelImageStats(img, label_min, label_max, label_counts);
  }

  for (size_t idx = 0; idx < train_dataset->size(); ++idx) {
    auto item = train_dataset->at(idx);
    cv::Mat img;
    DecodeImage(item.second, img);
    //LabelImageToDatum(img, datum);
    AccumulateLabelImageStats(img, label_min, label_max, label_counts);
  }

  return 0;
}
