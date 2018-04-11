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

  DataPipelineDataSource<pair<MutDatum<float>, uint32_t>> train_pipeline(
      8,
      static_cast<std::function<shared_ptr<DataSource<pair<Datum, uint32_t>>>(void)>>([train_dataset](void) {
        shared_ptr<DataSource<pair<Datum, uint32_t>>> train_reader =
            //make_shared<RandomPermuteDataSource<pair<Datum, uint32_t>>>(train_dataset);
            make_shared<RandomSamplingDataSource<pair<Datum, uint32_t>>>(train_dataset);
        return train_reader;
      }),
      static_cast<std::function<pair<MutDatum<float>, uint32_t>(size_t, size_t, pair<Datum, uint32_t>, std::mt19937_64*)>>([](size_t tid, size_t num_tids, auto src_item, auto rng) {
        uint32_t label = src_item.second;
        cv::Mat img;
        DecodeImage(src_item.first, img);
        // TODO: Inception-style data augmentation.
        InceptionPatchSample inception_patch_cfg = {
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
        XFlipImage(xflip_cfg, img, rng);
        ConvertByteToFloatImage(img);
        RandomColorNoise color_noise_cfg = {
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
        LightingNoiseImage(lighting_noise_cfg, img, rng);
        MutDatum<float> datum = MutDatum<float>::Alloc(3UL * 224UL * 224UL);
        //MutDatum<float> datum = MutDatum<float>::Alloc(3UL * 256UL * 256UL);
        ImageToFloatDatum(img, datum);
        return make_pair(datum, label);
      }));

  DataPipelineDataSource<pair<MutDatum<float>, uint32_t>> val_pipeline(
      1,
      static_cast<std::function<shared_ptr<DataSource<pair<Datum, uint32_t>>>(void)>>([val_dataset](void) {
        shared_ptr<DataSource<pair<Datum, uint32_t>>> val_reader =
            make_shared<PermuteDataSource<pair<Datum, uint32_t>>>(val_dataset);
        return val_reader;
      }),
      static_cast<std::function<pair<MutDatum<float>, uint32_t>(size_t, size_t, pair<Datum, uint32_t>, std::mt19937_64*)>>([](size_t tid, size_t num_tids, auto src_item, auto rng) {
        uint32_t label = src_item.second;
        cv::Mat img;
        DecodeImage(src_item.first, img);
        UniformRandomIsotropicResize resize_cfg = { .lower = 256, .upper = 256 };
        //UniformRandomIsotropicResize resize_cfg = { .lower = 288, .upper = 288 };
        //UniformRandomIsotropicResize resize_cfg = { .lower = 320, .upper = 320 };
        //UniformRandomIsotropicResize resize_cfg = { .lower = 384, .upper = 384 };
        TransformImage(resize_cfg, img, rng);
        CenterCrop crop_cfg = { .crop_width = 224, .crop_height = 224 };
        //CenterCrop crop_cfg = { .crop_width = 256, .crop_height = 256 };
        TransformImage(crop_cfg, img, rng);
        ConvertByteToFloatImage(img);
        MutDatum<float> datum = MutDatum<float>::Alloc(3UL * 224UL * 224UL);
        //MutDatum<float> datum = MutDatum<float>::Alloc(3UL * 256UL * 256UL);
        ImageToFloatDatum(img, datum);
        return make_pair(datum, label);
      }));

  std::clog << "DEBUG: creating multi gpu context..." << std::endl;
  //auto ctx = GPUContext::Make(0);
  auto parallel_ctx = MultiGPUContext::Make();
  std::clog << "DEBUG: getting single gpu context..." << std::endl;
  shared_ptr<Context> ctx = parallel_ctx->device_context(0);

  std::clog << "DEBUG: creating net..." << std::endl;
  //auto loss = ImagenetPreproc<float>::Make(224, 224, batch_size, ctx);
  const size_t batch_size = 16UL;
  //const size_t batch_size = 32UL;
  const size_t minibatch_size = 256UL;
  //const size_t minibatch_size = 8192UL;
  ConvNetConfig net_cfg = {
    //.input_c = 3,
    .input_w = 224,
    .input_h = 224,
    //.input_w = 256,
    //.input_h = 256,
    .batch_size = batch_size,
    .minibatch_size = minibatch_size,
  };
  const size_t image_size = 3UL * net_cfg.input_w * net_cfg.input_h;

  //auto net = ResNet<float>::MakeTestNet(net_cfg, ctx);
  //auto net = ResNet<float>::MakeResNet18(net_cfg, ctx);
  //auto net = ResNet<float>::MakeParallelResNet18(net_cfg, parallel_ctx);
  auto net = ResNet<float>::MakeResNet50(net_cfg, ctx);
  parallel_ctx->sync();
  //std::clog << "DEBUG: op label: " << net.op->_debug_op_label() << std::endl;

  std::random_device init_rd;
  std::mt19937_64 init_rng(init_rd());
  auto init_txn = TxnId();
  net.op->init(init_txn, &init_rng);

  //const size_t train_iters = 19200UL;
  const size_t train_iters = 600000UL;
  //const size_t validate_interval = 160UL;
  const size_t validate_interval = 1250UL;
  //const size_t validate_interval = 5000UL;
  vector<pair<size_t, double>> schedule;
  /*schedule.push_back(make_pair(160, 0.1 * 8));
  schedule.push_back(make_pair(320, 0.1 * 14));
  schedule.push_back(make_pair(480, 0.1 * 20));
  schedule.push_back(make_pair(640, 0.1 * 26));
  schedule.push_back(make_pair(800, 0.1 * 32));
  schedule.push_back(make_pair(4800, 0.01 * 32));
  schedule.push_back(make_pair(9600, 0.001 * 32));*/
  schedule.push_back(make_pair(150000, 0.01));
  schedule.push_back(make_pair(300000, 0.001));
  const SGDConfig sgd_cfg = {
    .minibatch_size = minibatch_size,
    .compute_batch_size = batch_size,
    //.step_size = make_shared<PiecewiseStepSchedule<double>>(0.1, 150000, 0.01, 300000, 0.001),
    .step_size = make_shared<PiecewiseStepSchedule<double>>(0.1, schedule),
    .momentum = make_shared<ConstantSchedule<double>>(0.9),
    .nesterov = true,
    .l2_reg = 1.0e-4,
    //.batch_norm_rate = 0.003,
    .batch_norm_rate = 0.1,
  };

  //SGDOptimizer<float> sgd(sgd_cfg, net.sum_loss, net.params, net.grads, net.batch_stats, net.fixed_stats);

  /*vector<VarIdSet> params;
  params.push_back(net.params);
  vector<VarIdSet> grads;
  grads.push_back(net.grads);
  vector<VarIdSet> batch_stats;
  batch_stats.push_back(net.batch_stats);
  vector<VarIdSet> fixed_stats;
  fixed_stats.push_back(net.fixed_stats);
  ParallelSGDOptimizer<float> sgd(sgd_cfg, net.sum_loss, params, grads, batch_stats, fixed_stats);*/

  SGDOptimizer<float> sgd(sgd_cfg, net.sum_loss, net.params, net.grads, net.batch_stats, net.fixed_stats);
  //ParallelSGDOptimizer<float> sgd(sgd_cfg, net.sum_loss, net.dev_params, net.dev_grads, net.dev_batch_stats, net.dev_fixed_stats);
  //DataParallelSGDOptimizer<float> sgd(sgd_cfg, net.sum_loss, net.params, net.grads, net.batch_stats, net.fixed_stats);

  sgd.reset(ctx);
  //sgd.reset(parallel_ctx);

  //uint8_t* batch_input = (uint8_t*)std::malloc(sizeof(uint8_t) * image_size * batch_size);
  float* batch_input = (float*)std::malloc(sizeof(float) * image_size * batch_size);
  uint32_t* batch_label = (uint32_t*)std::malloc(sizeof(uint32_t) * batch_size);
  float* batch_logit = (float*)std::malloc(sizeof(float) * 1000UL * batch_size);
  float* batch_loss = (float*)std::malloc(sizeof(float) * batch_size);

  size_t sample_count = 0;

  for (int t = 0; t < train_iters; ++t) {
    auto start = std::chrono::steady_clock::now();
    std::clog << "DEBUG: minibatch: " << t << std::endl;

    float avg_minibatch_accuracy = 0.0f;
    float avg_minibatch_loss = 0.0f;
    auto batch_fn = static_cast<function<void(TxnId, size_t, shared_ptr<Op>)>>(
        [&train_pipeline, &net, image_size, batch_input, batch_label, batch_logit, batch_loss, &avg_minibatch_accuracy, &avg_minibatch_loss, parallel_ctx]
        (auto batch_txn, auto batch_size, auto loss) mutable {
          *net.batch_norm_use_fixed = false;

          loss->persist(batch_txn, net.params);
          loss->persist(batch_txn, net.fixed_stats);
          /*const size_t num_ranks = parallel_ctx->num_ranks();
          for (size_t rank = 0; rank < num_ranks; ++rank) {
            loss->persist(batch_txn, net.dev_params.at(rank));
            loss->persist(batch_txn, net.dev_fixed_stats.at(rank));
          }*/

          for (size_t idx = 0; idx < batch_size; ++idx) {
            auto item = train_pipeline.next();
            assert(static_cast<bool>(item));
            assert(item.value().first.size == image_size);
            //std::memcpy(batch_input + image_size * idx, item.value().first.buffer, sizeof(uint8_t) * image_size);
            std::memcpy(batch_input + image_size * idx, item.value().first.buffer, sizeof(float) * image_size);
            item.value().first.dealloc();
            batch_label[idx] = item.value().second;
          }

          //any input_reader(Slice<uint8_t>(batch_input, image_size * batch_size));
          any input_reader(Slice<float>(batch_input, image_size * batch_size));
          size_t input_offset = loss->load_val(batch_txn, net.inputs, 0, &input_reader);
          assert(image_size * batch_size == input_offset);

          any label_reader(Slice<uint32_t>(batch_label, batch_size));
          size_t label_offset = loss->load_val(batch_txn, net.labels, 0, &label_reader);
          assert(batch_size == label_offset);

          loss->gradient(batch_txn);
          parallel_ctx->sync();

          any logit_writer(Slice<float>(batch_logit, 1000UL * batch_size));
          size_t logit_offset = loss->store_val(batch_txn, net.logits, 0, &logit_writer);
          assert(1000UL * batch_size == logit_offset);
          for (size_t idx = 0; idx < batch_size; ++idx) {
            float max_logit = -INFINITY;
            size_t argmax_logit = -1;
            for (size_t k = 0; k < 1000UL; ++k) {
              if (batch_logit[k + 1000UL * idx] > max_logit) {
                max_logit = batch_logit[k + 1000UL * idx];
                argmax_logit = k;
              }
            }
            if (argmax_logit == batch_label[idx]) {
              avg_minibatch_accuracy += 1.0f;
            }
          }

          any loss_writer(Slice<float>(batch_loss, batch_size));
          size_t loss_offset = loss->store_val(batch_txn, net.losses, 0, &loss_writer);
          assert(batch_size == loss_offset);
          for (size_t idx = 0; idx < batch_size; ++idx) {
            //std::clog << "DEBUG:   batch loss nll [" << idx << "]: " << batch_loss[idx] << std::endl;
            avg_minibatch_loss += batch_loss[idx];
          }
        }
    );
    sgd.step(batch_fn, ctx);
    //sgd.step(batch_fn, parallel_ctx);
    avg_minibatch_accuracy /= (float)sgd_cfg.minibatch_size;
    avg_minibatch_loss /= (float)sgd_cfg.minibatch_size;
    sample_count += sgd_cfg.minibatch_size;

    std::clog << "DEBUG:   avg minibatch accuracy: " << avg_minibatch_accuracy << std::endl;
    std::clog << "DEBUG:   avg minibatch loss:     " << avg_minibatch_loss << std::endl;

    auto lap = std::chrono::steady_clock::now();
    auto diff = lap - start;
    double elapsed_ms = std::chrono::duration<double, std::milli>(diff).count();
    std::clog << "DEBUG:   "
        << "epochs: " << sample_count / train_size
        << " samples: " << sample_count << "/" << train_size
        << " lr: " << sgd_cfg.step_size->at(t)
        << " elapsed: " << elapsed_ms << " ms" << std::endl;

    if ((t + 1) % validate_interval == 0) {
      auto start = std::chrono::steady_clock::now();
      std::clog << "DEBUG: evaluating..." << std::endl;

      size_t total_val_idx = 0;
      size_t num_val_batches = (val_size + batch_size - 1) / batch_size;

      float val_accuracy = 0.0f;
      //float val_loss = 0.0f;

      *net.batch_norm_use_fixed = true;
      //sgd.reload_lagged_param(parallel_ctx);

      for (size_t batch_nr = 0; batch_nr < num_val_batches; ++batch_nr) {
        TxnId batch_txn;

        net.loss->persist(batch_txn, net.params);
        net.loss->persist(batch_txn, net.fixed_stats);
        /*const size_t num_ranks = parallel_ctx->num_ranks();
        for (size_t rank = 0; rank < num_ranks; ++rank) {
          net.loss->persist(batch_txn, net.dev_params.at(rank));
          net.loss->persist(batch_txn, net.dev_fixed_stats.at(rank));
        }*/

        size_t actual_batch_size = std::min(batch_size, val_size - batch_nr * batch_size);
        // FIXME: this is a hack, to avoid allocating extra workspace
        // when memory usage is tight.
        /*if (actual_batch_size < batch_size) {
          for (size_t idx = 0; idx < actual_batch_size; ++idx) {
            auto item = val_pipeline.next();
          }
          break;
        }*/
        for (size_t idx = 0; idx < actual_batch_size; ++idx) {
          auto item = val_pipeline.next();
          assert(static_cast<bool>(item));
          assert(item.value().first.size == image_size);
          //std::memcpy(batch_input + image_size * idx, item.value().first.buffer, sizeof(uint8_t) * image_size);
          std::memcpy(batch_input + image_size * idx, item.value().first.buffer, sizeof(float) * image_size);
          item.value().first.dealloc();
          batch_label[idx] = item.value().second;
          ++total_val_idx;
        }

        //any input_reader(Slice<uint8_t>(batch_input, image_size * actual_batch_size));
        any input_reader(Slice<float>(batch_input, image_size * actual_batch_size));
        size_t input_offset = net.op->load_val(batch_txn, net.inputs, 0, &input_reader);
        assert(image_size * actual_batch_size == input_offset);

        /*any label_reader(Slice<uint32_t>(batch_label, actual_batch_size));
        size_t label_offset = net.loss->load_val(batch_txn, net.labels, 0, &label_reader);
        assert(actual_batch_size == label_offset);*/

        net.op->eval(batch_txn);
        parallel_ctx->sync();

        any logit_writer(Slice<float>(batch_logit, 1000UL * actual_batch_size));
        size_t logit_offset = net.op->store_val(batch_txn, net.logits, 0, &logit_writer);
        assert(1000UL * actual_batch_size == logit_offset);
        for (size_t idx = 0; idx < actual_batch_size; ++idx) {
          float max_logit = -INFINITY;
          size_t argmax_logit = -1;
          for (size_t k = 0; k < 1000UL; ++k) {
            if (batch_logit[k + 1000UL * idx] > max_logit) {
              max_logit = batch_logit[k + 1000UL * idx];
              argmax_logit = k;
            }
          }
          if (argmax_logit == batch_label[idx]) {
            val_accuracy += 1.0f;
          }
        }

        /*any loss_writer(Slice<float>(batch_loss, actual_batch_size));
        size_t loss_offset = net.loss->store_val(batch_txn, net.losses, 0, &loss_writer);
        assert(actual_batch_size == loss_offset);
        for (size_t idx = 0; idx < actual_batch_size; ++idx) {
          //std::clog << "DEBUG:   batch loss nll [" << idx << "]: " << batch_loss[idx] << std::endl;
          val_loss += batch_loss[idx];
        }*/
      }
      //assert(val_size == total_val_idx);

      *net.batch_norm_use_fixed = false;
      //sgd.reload_param(parallel_ctx);

      size_t val_accuracy_count = val_accuracy;
      val_accuracy /= static_cast<float>(total_val_idx);
      //val_loss /= static_cast<float>(val_size);

      std::clog << "DEBUG:   validation accuracy: " << val_accuracy
          << " (" << val_accuracy_count << "/" << total_val_idx << ")" << std::endl;
      //std::clog << "DEBUG:   validation loss:     " << val_loss << std::endl;

      auto lap = std::chrono::steady_clock::now();
      auto diff = lap - start;
      double elapsed_ms = std::chrono::duration<double, std::milli>(diff).count();
      std::clog << "DEBUG:   samples: " << total_val_idx << " elapsed: " << elapsed_ms * 0.001 << " s" << std::endl;
    }
  }

  return 0;
}
