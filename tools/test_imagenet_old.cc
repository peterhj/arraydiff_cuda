#include "nvarraydiff/io/imagenet.hh"
#include "nvarraydiff/io/pipeline.hh"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>

using namespace nvarraydiff;
using namespace nvarraydiff::io;
using std::make_shared;
using std::shared_ptr;
using std::vector;

int main(int argc, const char** argv) {
  ImagenetMetadata metadata("ilsvrc2012_ids.txt", "wordnet_ids.txt", "/raid/ilsvrc2012_devkit/data/ILSVRC2012_validation_ground_truth.txt");
  ImagenetValTar eval_dataset(metadata, "/raid/imagenet/ILSVRC2012_tars/ILSVRC2012_img_val.tar");
  shared_ptr<RandomAccessData<LabeledDatum>> train_dataset =
      make_shared<ImagenetTrainTar>(metadata, "/raid/imagenet/ILSVRC2012_tars/ILSVRC2012_img_train.tar");
  shared_ptr<DataSource<LabeledDatum>> train_reader =
      make_shared<RandomSamplingDataSource<LabeledDatum>>(train_dataset);
  assert(train_reader.get() != NULL);
  //auto item0 = train_reader->next();
  //std::clog << "DEBUG: item size: " << item0.first.size << std::endl;
  int counter = 0;
  DataPipelineDataSource<LabeledDatum> train_pipeline(
      train_reader,
      static_cast<std::function<LabeledDatum(LabeledDatum, std::mt19937_64&)>>([&](auto src_item, auto rng) mutable {
        uint32_t label = src_item.second;
        cv::_InputArray buf(src_item.first.buffer, src_item.first.size);
        cv::Mat img = cv::imdecode(buf, 1);
        std::clog << "DEBUG: image shape: " << counter << " " << label << " " << img.channels() << " " << img.cols << " " << img.rows << std::endl;
        ++counter;
        return src_item;
      }));
  for (int t = 0; t < 1350; ++t) {
    auto item = train_pipeline.next();
    assert((bool)(item));
    //std::clog << "DEBUG: item size: " << item.value().first.size << std::endl;
  }
  return 0;
}
