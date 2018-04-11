#include "arraydiff/io/image.hh"
#include "arraydiff/io/pipeline.hh"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace arraydiff {
namespace io {

using std::array;
using std::vector;

void DecodeImage(Datum datum, cv::Mat& img) {
  cv::_InputArray buf(datum.buffer, datum.size);
  img = cv::imdecode(buf, cv::IMREAD_COLOR);
}

void ConvertByteToFloatImage(cv::Mat& img) {
  const size_t img_channels = img.channels();
  assert(img_channels == 3);
  cv::Mat src_img = img;
  img.release();
  src_img.convertTo(img, CV_32FC3);
  src_img.release();
}

void ImageToDatum(const cv::Mat& img, MutDatum<> datum) {
  const size_t channels = img.channels();
  const size_t width = img.cols;
  const size_t height = img.rows;
  assert(channels == 3);
  assert(width * height * channels == datum.size);
  size_t idx = 0;
  for (size_t h = 0; h < height; ++h) {
    const uint8_t* row_ptr = img.ptr<uint8_t>(h);
    for (size_t w = 0; w < width; ++w) {
      for (size_t c = 0; c < channels; ++c) {
        datum.buffer[idx] = row_ptr[c + channels * w];
        ++idx;
      }
    }
  }
  assert(idx == datum.size);
}

void ImageToFloatDatum(const cv::Mat& img, MutDatum<float> datum) {
  const size_t channels = img.channels();
  const size_t width = img.cols;
  const size_t height = img.rows;
  assert(channels == 3);
  assert(width * height * channels == datum.size);
  size_t idx = 0;
  for (size_t h = 0; h < height; ++h) {
    const float* row_ptr = img.ptr<float>(h);
    for (size_t w = 0; w < width; ++w) {
      for (size_t c = 0; c < channels; ++c) {
        datum.buffer[idx] = row_ptr[c + channels * w];
        ++idx;
      }
    }
  }
  assert(idx == datum.size);
}

void ImageToVector(const cv::Mat& img, vector<uint8_t>& buffer) {
  const size_t channels = img.channels();
  const size_t width = img.cols;
  const size_t height = img.rows;
  assert(channels == 3);
  buffer.clear();
  buffer.resize(width * height * channels);
  size_t idx = 0;
  for (size_t h = 0; h < height; ++h) {
    const uint8_t* row_ptr = img.ptr<uint8_t>(h);
    for (size_t w = 0; w < width; ++w) {
      for (size_t c = 0; c < channels; ++c) {
        buffer.at(idx) = row_ptr[c + channels * w];
        ++idx;
      }
    }
  }
}

void AccumulateLabelImageStats(const cv::Mat& img, uint32_t& label_min, uint32_t& label_max, std::map<uint32_t, size_t>& label_counts) {
  const size_t channels = img.channels();
  const size_t width = img.cols;
  const size_t height = img.rows;
  assert(channels == 3);
  //assert(width * height == datum.size);
  //size_t idx = 0;
  size_t prev_card = label_counts.size();
  for (size_t h = 0; h < height; ++h) {
    const uint8_t* row_ptr = img.ptr<uint8_t>(h);
    for (size_t w = 0; w < width; ++w) {
      uint32_t label = static_cast<uint32_t>(row_ptr[channels * w]);
      if (label_min > label) {
        label_min = label;
      }
      if (label_max < label) {
        label_max = label;
      }
      label_counts.emplace(std::make_pair(label, 1));
      //datum.buffer[idx] = static_cast<uint32_t>(row_ptr[channels * w]);
      //++idx;
    }
  }
  size_t next_card = label_counts.size();
  if (prev_card != next_card) {
    std::clog << "DEBUG: LabelImageToDatum:"
        " cardinality: " << label_counts.size() <<
        " min: " << (int)label_min <<
        " max: " << (int)label_max << std::endl;
  }
  //assert(idx == datum.size);
}

void LabelImageToDatum(const cv::Mat& img, MutDatum<uint32_t> datum) {
  const size_t channels = img.channels();
  const size_t width = img.cols;
  const size_t height = img.rows;
  assert(channels == 3);
  assert(width * height == datum.size);
  size_t idx = 0;
  for (size_t h = 0; h < height; ++h) {
    const uint8_t* row_ptr = img.ptr<uint8_t>(h);
    for (size_t w = 0; w < width; ++w) {
      datum.buffer[idx] = static_cast<uint32_t>(row_ptr[channels * w]);
      ++idx;
    }
  }
  //std::clog << "DEBUG: LabelImageToDatum" << std::endl;
  assert(idx == datum.size);
}

template <>
void TransformImage(const InceptionPatchSample& cfg, cv::Mat& img, std::mt19937_64* rng) {
  assert(cfg.area_frac_lower > 0.0);
  assert(cfg.area_frac_lower <= cfg.area_frac_upper);
  assert(cfg.area_frac_upper <= 1.0);
  assert(cfg.aspect_ratio_lower > 0.0);
  assert(cfg.aspect_ratio_lower <= 1.0);
  assert(cfg.aspect_ratio_lower <= cfg.aspect_ratio_upper);
  assert(cfg.aspect_ratio_upper >= 1.0);

  const size_t img_width = img.cols;
  const size_t img_height = img.rows;

  std::uniform_real_distribution<double> dist_area_frac(cfg.area_frac_lower, cfg.area_frac_upper);
  std::uniform_real_distribution<double> dist_aspect_ratio(cfg.aspect_ratio_lower, cfg.aspect_ratio_upper);
  std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);
  /*std::uniform_int_distribution<size_t> dist_aspect_ratio_lo_or_hi(0, 1);
  std::uniform_real_distribution<double> dist_aspect_ratio_lo(cfg.aspect_ratio_lower, 1.0);
  std::uniform_real_distribution<double> dist_aspect_ratio_hi(1.0, cfg.aspect_ratio_upper);*/

  size_t attempt = 0;
  for (;;) {
    double area_frac = dist_area_frac(*rng);
    double aspect_ratio = dist_aspect_ratio(*rng);
    /*size_t target_aspect_ratio_lo_or_hi = dist_aspect_ratio_lo_or_hi(*rng);
    double aspect_ratio_mix = 0.0;
    if (0 == target_aspect_ratio_lo_or_hi) {
      aspect_ratio_mix = dist_aspect_ratio_lo(*rng);
    } else if (1 == target_aspect_ratio_lo_or_hi) {
      aspect_ratio_mix = dist_aspect_ratio_hi(*rng);
    } else {
      assert(0 && "unreachable");
    }*/

    double img_patch_area = area_frac * static_cast<double>(img_width * img_height);
    double img_patch_width = std::sqrt(img_patch_area * aspect_ratio);
    double img_patch_height = std::sqrt(img_patch_area / aspect_ratio);
    if (dist_uniform(*rng) < 0.5) {
      img_patch_width = std::sqrt(img_patch_area / aspect_ratio);
      img_patch_height = std::sqrt(img_patch_area * aspect_ratio);
    }
    /*double old_aspect_ratio = static_cast<double>(img_height) / static_cast<double>(img_width);
    double new_aspect_ratio = aspect_ratio_mix * old_aspect_ratio;
    double img_patch_width = std::sqrt(img_patch_area / new_aspect_ratio);
    double img_patch_height = new_aspect_ratio * img_patch_width;*/

    size_t img_patch_width_r = static_cast<size_t>(std::round(img_patch_width));
    size_t img_patch_height_r = static_cast<size_t>(std::round(img_patch_height));
    size_t img_patch_smaller_side_r = std::min(img_patch_width_r, img_patch_height_r);

    /*std::clog << "DEBUG: InceptionPatchSample:"
        << " attempt: " << attempt
        << " af: " << area_frac
        << " ow: " << img_width
        << " oh: " << img_height
        << " pw: " << img_patch_width_r
        << " ph: " << img_patch_height_r
        << std::endl;*/

    if (cfg.reject_larger_images) {
      if (img_patch_width_r > img_width || img_patch_height_r > img_height) {
        ++attempt;
        if (10 == attempt) {
          double resize_scale = static_cast<double>(cfg.target_resize) / static_cast<double>(img_patch_smaller_side_r);
          double resize_width = resize_scale * static_cast<double>(img_patch_width_r);
          double resize_height = resize_scale * static_cast<double>(img_patch_height_r);
          size_t resize_width_r = static_cast<size_t>(std::round(resize_width));
          size_t resize_height_r = static_cast<size_t>(std::round(resize_height));
          assert(resize_width_r >= cfg.target_resize);
          assert(resize_height_r >= cfg.target_resize);

          cv::Mat src_img = img;
          img.release();
          cv::resize(src_img, img, cv::Size(resize_width_r, resize_height_r), 0, 0, cv::INTER_CUBIC);
          src_img.release();

          size_t offset_w = (resize_width_r - cfg.target_resize) / 2;
          size_t offset_h = (resize_height_r - cfg.target_resize) / 2;

          src_img = img;
          img.release();
          cv::Rect crop_roi = cv::Rect(offset_w, offset_h, cfg.target_resize, cfg.target_resize);
          img = src_img(crop_roi);
          src_img.release();

          assert(img.cols == cfg.target_resize);
          assert(img.rows == cfg.target_resize);
          return;
        }
        continue;
      }
    } else {
      img_patch_width_r = std::min(img_width, img_patch_width_r);
      img_patch_height_r = std::min(img_height, img_patch_height_r);
    }

    std::uniform_int_distribution<size_t> dist_offset_w(0, img_width - img_patch_width_r);
    std::uniform_int_distribution<size_t> dist_offset_h(0, img_height - img_patch_height_r);
    size_t offset_w = dist_offset_w(*rng);
    size_t offset_h = dist_offset_h(*rng);

    size_t resize_width_r = cfg.target_resize;
    size_t resize_height_r = cfg.target_resize;

    /*double resize_scale = static_cast<double>(cfg.target_resize) / static_cast<double>(img_patch_smaller_side_r);
    double resize_width = resize_scale * static_cast<double>(img_patch_width_r);
    double resize_height = resize_scale * static_cast<double>(img_patch_height_r);

    size_t resize_width_r = static_cast<size_t>(std::round(resize_width));
    size_t resize_height_r = static_cast<size_t>(std::round(resize_height));*/

    cv::Mat src_img = img;
    img.release();
    cv::Rect patch_crop_roi = cv::Rect(offset_w, offset_h, img_patch_width_r, img_patch_height_r);
    img = src_img(patch_crop_roi);
    src_img.release();

    assert(img.cols == img_patch_width_r);
    assert(img.rows == img_patch_height_r);

    if (resize_width_r == img_patch_width_r && resize_height_r == img_patch_height_r) {
    } else if (resize_width_r <= img_patch_width_r && resize_height_r <= img_patch_height_r) {
      src_img = img;
      img.release();
      cv::resize(src_img, img, cv::Size(resize_width_r, resize_height_r), 0, 0, cv::INTER_AREA);
      src_img.release();

    } else if (resize_width_r >= img_patch_width_r && resize_height_r >= img_patch_height_r) {
      src_img = img;
      img.release();
      cv::resize(src_img, img, cv::Size(resize_width_r, resize_height_r), 0, 0, cv::INTER_CUBIC);
      src_img.release();

    } else if (resize_width_r < img_patch_width_r && resize_height_r >= img_patch_height_r) {
      src_img = img;
      img.release();
      cv::resize(src_img, img, cv::Size(resize_width_r, resize_height_r), 0, 0, cv::INTER_CUBIC);
      src_img.release();

      /*src_img = img;
      img.release();
      cv::resize(src_img, img, cv::Size(resize_width_r, img_patch_height_r), 0, 0, cv::INTER_AREA);
      src_img.release();

      src_img = img;
      img.release();
      cv::resize(src_img, img, cv::Size(resize_width_r, resize_height_r), 0, 0, cv::INTER_CUBIC);
      src_img.release();*/

    } else if (resize_width_r >= img_patch_width_r && resize_height_r < img_patch_height_r) {
      src_img = img;
      img.release();
      cv::resize(src_img, img, cv::Size(resize_width_r, resize_height_r), 0, 0, cv::INTER_CUBIC);
      src_img.release();

      /*src_img = img;
      img.release();
      cv::resize(src_img, img, cv::Size(img_patch_width_r, resize_height_r), 0, 0, cv::INTER_AREA);
      src_img.release();

      src_img = img;
      img.release();
      cv::resize(src_img, img, cv::Size(resize_width_r, resize_height_r), 0, 0, cv::INTER_CUBIC);
      src_img.release();*/

    } else {
      assert(0 && "unreachable");
    }

    /*assert(img.cols == resize_width_r);
    assert(img.rows == resize_height_r);*/
    assert(img.cols == cfg.target_resize);
    assert(img.rows == cfg.target_resize);
    return;
  }
}

template <>
void TransformImage(const UniformRandomIsotropicResize& cfg, cv::Mat& img, std::mt19937_64* rng) {
  const size_t width = img.cols;
  const size_t height = img.rows;
  std::uniform_int_distribution<size_t> dist_side_(cfg.lower, cfg.upper);
  size_t resize_side = dist_side_(*rng);
  size_t smaller_side = std::min(width, height);
  double scale = static_cast<double>(resize_side) / static_cast<double>(smaller_side);
  size_t resize_w = static_cast<size_t>(std::round(scale * static_cast<double>(width)));
  size_t resize_h = static_cast<size_t>(std::round(scale * static_cast<double>(height)));
  cv::Mat src_img = img;
  img.release();
  if (resize_w == width && resize_h == height) {
    img = src_img;
  } else if (resize_w <= width && resize_h <= height) {
    cv::resize(src_img, img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_AREA);
  } else if (resize_w >= width && resize_h >= height) {
    cv::resize(src_img, img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_CUBIC);
  } else {
    assert(0 && "unreachable");
  }
  src_img.release();
}

template <>
void TransformImage(const UniformRandomCrop& cfg, cv::Mat& img, std::mt19937_64* rng) {
  const size_t width = img.cols;
  const size_t height = img.rows;
  assert(cfg.crop_width <= width);
  assert(cfg.crop_height <= height);
  std::uniform_int_distribution<size_t> dist_w_(0UL, width - cfg.crop_width);
  std::uniform_int_distribution<size_t> dist_h_(0UL, height - cfg.crop_height);
  size_t offset_w = dist_w_(*rng);
  size_t offset_h = dist_h_(*rng);
  cv::Mat src_img = img;
  cv::Rect crop_roi = cv::Rect(offset_w, offset_h, cfg.crop_width, cfg.crop_height);
  img.release();
  img = src_img(crop_roi);
  src_img.release();
}

template <>
void TransformImage(const CenterCrop& cfg, cv::Mat& img, std::mt19937_64* rng) {
  const size_t width = img.cols;
  const size_t height = img.rows;
  assert(cfg.crop_width <= width);
  assert(cfg.crop_height <= height);
  size_t offset_w = (width - cfg.crop_width) / 2;
  size_t offset_h = (height - cfg.crop_height) / 2;
  cv::Mat src_img = img;
  cv::Rect crop_roi = cv::Rect(offset_w, offset_h, cfg.crop_width, cfg.crop_height);
  img.release();
  img = src_img(crop_roi);
  src_img.release();
}

template <>
void XFlipImage(const RandomXFlip& cfg, cv::Mat& img, std::mt19937_64* rng) {
  const size_t width = img.cols;
  const size_t height = img.rows;
  std::uniform_int_distribution<int> dist(0, 1);
  int flip = dist(*rng);
  if (1 == flip) {
    cv::Mat src_img = img;
    img.release();
    cv::flip(src_img, img, 1);
    src_img.release();
  }
}

template <>
void ColorNoiseImage(const RandomColorNoise& cfg, cv::Mat& img, std::mt19937_64* rng) {
  assert(cfg.randomize_order);
  assert(cfg.brightness >= 0.0);
  assert(cfg.contrast >= 0.0);
  assert(cfg.saturation >= 0.0);
  const size_t img_channels = img.channels();
  const size_t img_width = img.cols;
  const size_t img_height = img.rows;
  assert(img_channels == 3);

  vector<size_t> order = {0, 1, 2};
  std::shuffle(order.begin(), order.end(), *rng);
  for (size_t rank = 0; rank < 3; ++rank) {
    if (0 == order.at(rank)) {
      std::uniform_real_distribution<double> dist(-cfg.brightness, cfg.brightness);
      float alpha = 1.0 + dist(*rng);
      cv::Mat src_img = img;
      img.release();
      img.create(img_width, img_height, CV_32FC3);
      // TODO: blend: (1-alpha) * 0 + alpha * pixel.
      size_t idx = 0;
      for (size_t h = 0; h < img_height; ++h) {
        const float* src_row_ptr = src_img.ptr<float>(h);
        float* dst_row_ptr = img.ptr<float>(h);
        for (size_t w = 0; w < img_width; ++w) {
          for (size_t c = 0; c < img_channels; ++c) {
            dst_row_ptr[c + img_channels * w] = alpha * src_row_ptr[c + img_channels * w];
            ++idx;
          }
        }
      }
      assert(idx == img_channels * img_width * img_height);
      src_img.release();
    } else if (1 == order.at(rank)) {
      std::uniform_real_distribution<double> dist(-cfg.contrast, cfg.contrast);
      float alpha = 1.0 + dist(*rng);
      cv::Mat src_img = img;
      img.release();
      img.create(img_width, img_height, CV_32FC3);
      // TODO: blend: (1-alpha) * mean(gray(pixel)) + alpha * pixel.
      float mean_gray_src_pixel = 0.0f;
      size_t idx = 0;
      for (size_t h = 0; h < img_height; ++h) {
        const float* src_row_ptr = src_img.ptr<float>(h);
        for (size_t w = 0; w < img_width; ++w) {
          // NOTE: OpenCV default channel order is BGR.
          float gray_src_pixel =
              0.299f * src_row_ptr[2 + img_channels * w] +
              0.587f * src_row_ptr[1 + img_channels * w] +
              0.114f * src_row_ptr[0 + img_channels * w];
          mean_gray_src_pixel += (1.0f / static_cast<float>(idx + 1)) * (gray_src_pixel - mean_gray_src_pixel);
          ++idx;
        }
      }
      assert(idx == img_width * img_height);
      idx = 0;
      for (size_t h = 0; h < img_height; ++h) {
        const float* src_row_ptr = src_img.ptr<float>(h);
        float* dst_row_ptr = img.ptr<float>(h);
        for (size_t w = 0; w < img_width; ++w) {
          for (size_t c = 0; c < img_channels; ++c) {
            dst_row_ptr[c + img_channels * w] = mean_gray_src_pixel + alpha * (src_row_ptr[c + img_channels * w] - mean_gray_src_pixel);
            ++idx;
          }
        }
      }
      assert(idx == img_channels * img_width * img_height);
      src_img.release();
    } else if (2 == order.at(rank)) {
      std::uniform_real_distribution<double> dist(-cfg.saturation, cfg.saturation);
      float alpha = 1.0 + dist(*rng);
      cv::Mat src_img = img;
      img.release();
      img.create(img_width, img_height, CV_32FC3);
      // TODO: blend: (1-alpha) * gray(pixel) + alpha * pixel.
      size_t idx = 0;
      for (size_t h = 0; h < img_height; ++h) {
        const float* src_row_ptr = src_img.ptr<float>(h);
        float* dst_row_ptr = img.ptr<float>(h);
        for (size_t w = 0; w < img_width; ++w) {
          // NOTE: OpenCV default channel order is BGR.
          float gray_src_pixel =
              0.299f * src_row_ptr[2 + img_channels * w] +
              0.587f * src_row_ptr[1 + img_channels * w] +
              0.114f * src_row_ptr[0 + img_channels * w];
          for (size_t c = 0; c < img_channels; ++c) {
            dst_row_ptr[c + img_channels * w] = gray_src_pixel + alpha * (src_row_ptr[c + img_channels * w] - gray_src_pixel);
            ++idx;
          }
        }
      }
      assert(idx == img_channels * img_width * img_height);
      src_img.release();
    } else {
      assert(0 && "unreachable");
    }
  }
}

template <>
void LightingNoiseImage(const PCALightingNoise& cfg, cv::Mat& img, std::mt19937_64* rng) {
  assert(cfg.stddev >= 0.0);
  const size_t img_channels = img.channels();
  const size_t img_width = img.cols;
  const size_t img_height = img.rows;
  assert(img_channels == 3);

  std::normal_distribution<double> dist(0.0, cfg.stddev);
  array<double, 3> alpha = { dist(*rng), dist(*rng), dist(*rng) };
  array<double, 3> rgb_noise = { 0.0, 0.0, 0.0 };
  for (size_t c = 0; c < 3; ++c) {
    for (size_t v = 0; v < 3; ++v) {
      rgb_noise.at(c) += cfg.scale * alpha.at(v) * cfg.eval.at(v) * cfg.evec.at(v + 3 * c);
    }
  }

  cv::Mat src_img = img;
  img.release();
  img.create(img_width, img_height, CV_32FC3);
  size_t idx = 0;
  for (size_t h = 0; h < img_height; ++h) {
    const float* src_row_ptr = src_img.ptr<float>(h);
    float* dst_row_ptr = img.ptr<float>(h);
    for (size_t w = 0; w < img_width; ++w) {
      for (size_t c = 0; c < img_channels; ++c) {
        /*if (idx < 20) {
          std::clog << "DEBUG: lighting noise:"
              << " idx: " << idx
              << " src: " << src_row_ptr[c + img_channels * w]
              << " noise: " << rgb_noise.at(c)
              << std::endl;
        }*/
        dst_row_ptr[c + img_channels * w] = src_row_ptr[c + img_channels * w] + rgb_noise.at(c);
        ++idx;
      }
    }
  }
  assert(idx == img_channels * img_width * img_height);
  src_img.release();
}

} // namespace io
} // namespace arraydiff
