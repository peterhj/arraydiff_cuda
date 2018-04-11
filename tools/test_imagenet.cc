#include "arraydiff/context.hh"
#include "arraydiff/io/image.hh"
#include "arraydiff/io/imagenet.hh"
#include "arraydiff/io/pipeline.hh"
#include "arraydiff/optimize.hh"
#include "arraydiff/slice.hh"
#include "arraydiff/zoo.hh"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <experimental/any>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>

using namespace arraydiff;
using namespace arraydiff::io;
using std::experimental::any;
using std::make_pair;
using std::make_shared;
using std::pair;
using std::shared_ptr;
using std::vector;

int main(int argc, const char** argv) {
  //ImagenetMetadata metadata("ilsvrc2012_ids.txt", "wordnet_ids.txt", "/raid/ilsvrc2012_devkit/data/ILSVRC2012_validation_ground_truth.txt");
  ImagenetMetadata metadata("ilsvrc2012_ids.txt", "wordnet_ids.txt", "ground_truth.txt");

  shared_ptr<RandomAccessData<pair<Datum, uint32_t>>> val_dataset =
      make_shared<ImagenetValTar>(metadata, "/scratch/phj/data/ilsvrc2012/ILSVRC2012_img_val.tar");
  const size_t val_size = val_dataset->size();

  shared_ptr<RandomAccessData<pair<Datum, uint32_t>>> train_dataset =
      make_shared<ImagenetTrainTar>(metadata, "/scratch/phj/data/ilsvrc2012/ILSVRC2012_img_train.tar");
  const size_t train_size = train_dataset->size();

  DataPipelineDataSource<pair<MutDatum<>, uint32_t>> train_pipeline(
  //DataPipelineDataSource<pair<MutDatum<float>, uint32_t>> train_pipeline(
      8,
      static_cast<std::function<shared_ptr<DataSource<pair<Datum, uint32_t>>>(void)>>([train_dataset](void) {
        shared_ptr<DataSource<pair<Datum, uint32_t>>> train_reader =
            //make_shared<RandomPermuteDataSource<pair<Datum, uint32_t>>>(train_dataset);
            make_shared<RandomSamplingDataSource<pair<Datum, uint32_t>>>(train_dataset);
        return train_reader;
      }),
      static_cast<std::function<pair<MutDatum<>, uint32_t>(size_t, size_t, pair<Datum, uint32_t>, std::mt19937_64*)>>([](size_t tid, size_t num_tids, auto src_item, auto rng) {
      //static_cast<std::function<pair<MutDatum<float>, uint32_t>(size_t, size_t, pair<Datum, uint32_t>, std::mt19937_64*)>>([](size_t tid, size_t num_tids, auto src_item, auto rng) {
        uint32_t label = src_item.second;
        cv::Mat img;
        DecodeImage(src_item.first, img);
        // TODO: Inception-style data augmentation.
        /*InceptionPatchSample inception_patch_cfg = {
          .area_frac_lower = 0.08,
          .area_frac_upper = 1.0,
          .aspect_ratio_lower = 3.0 / 4.0,
          .aspect_ratio_upper = 4.0 / 3.0,
          .target_resize = 224,
          //.target_resize = 256,
          //.target_resize = 320,
          .reject_larger_images = false,
        };
        TransformImage(inception_patch_cfg, img, rng);
        RandomXFlip xflip_cfg;
        XFlipImage(xflip_cfg, img, rng);*/
        //ConvertByteToFloatImage(img);
        /*RandomColorNoise color_noise_cfg = {
          .randomize_order = true,
          .brightness = 0.4,
          .contrast = 0.4,
          .saturation = 0.4,
        };
        ColorNoiseImage(color_noise_cfg, img, rng);
        PCALightingNoise lighting_noise_cfg = {
          .scale = 255.0,
          .stddev = 0.1,
          //.stddev = 1.0,
          .eval = {  0.2175,  0.0188,  0.0045, },
          .evec = { -0.5675,  0.7192,  0.4009,
                    -0.5808, -0.0045, -0.8140,
                    -0.5836, -0.6948,  0.4203, },
        };
        LightingNoiseImage(lighting_noise_cfg, img, rng);*/
        MutDatum<> datum = MutDatum<>::Alloc(3UL * img.cols * img.rows);
        //MutDatum<float> datum = MutDatum<float>::Alloc(3UL * img.cols * img.rows);
        //MutDatum<float> datum = MutDatum<float>::Alloc(3UL * 224UL * 224UL);
        //MutDatum<float> datum = MutDatum<float>::Alloc(3UL * 256UL * 256UL);
        ImageToDatum(img, datum);
        //ImageToFloatDatum(img, datum);
        return make_pair(datum, label);
      }));

  double num_trials = 0.0;
  double avg_elapsed_ms = 0.0;
  auto start = std::chrono::steady_clock::now();
  for (size_t idx = 0; ; ++idx) {
    auto item = train_pipeline.next();
    if ((idx + 1) % 1000 == 0) {
      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      double elapsed_ms = std::chrono::duration<double, std::milli>(diff).count();
      start = lap;
      num_trials += 1.0;
      avg_elapsed_ms += (1.0 / num_trials) * (elapsed_ms - avg_elapsed_ms);
      std::clog << "DEBUG:   "
          << " samples: " << idx + 1
          << " avg elapsed: " << avg_elapsed_ms << " ms" << std::endl;
    }
  }

  return 0;
}
