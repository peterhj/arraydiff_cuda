#ifndef ARRAYDIFF_IO_IMAGE_HH
#define ARRAYDIFF_IO_IMAGE_HH

#include "arraydiff/io/pipeline.hh"

#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace arraydiff {
namespace io {

using std::array;
using std::vector;

class ImageDim {
public:
  size_t width;
  size_t height;
  size_t channels;
};

void DecodeImage(Datum datum, cv::Mat& img);
void ConvertByteToFloatImage(cv::Mat& img);
void ImageToDatum(const cv::Mat& img, MutDatum<> datum);
void ImageToFloatDatum(const cv::Mat& img, MutDatum<float> datum);
void ImageToVector(const cv::Mat& img, vector<uint8_t>& buffer);
void AccumulateLabelImageStats(const cv::Mat& img, uint32_t& label_min, uint32_t& label_max, std::map<uint32_t, size_t>& label_counts);
void LabelImageToDatum(const cv::Mat& mask_img, MutDatum<uint32_t> datum);

class InceptionPatchSample {
public:
  double area_frac_lower;
  double area_frac_upper;
  double aspect_ratio_lower;
  double aspect_ratio_upper;
  size_t target_resize;
  bool reject_larger_images;
};

class UniformRandomIsotropicResize {
public:
  size_t lower;
  size_t upper;
};

class UniformRandomCrop {
public:
  size_t crop_width;
  size_t crop_height;
};

class CenterCrop {
public:
  size_t crop_width;
  size_t crop_height;
};

template <typename TransformConfig>
void TransformImage(const TransformConfig& cfg, cv::Mat& img, std::mt19937_64* rng);

class RandomXFlip {};

template <typename XFlipConfig>
void XFlipImage(const XFlipConfig& cfg, cv::Mat& img, std::mt19937_64* rng);

class RandomColorNoise {
public:
  bool randomize_order;
  double brightness;
  double contrast;
  double saturation;
};

template <typename ColorNoiseConfig>
void ColorNoiseImage(const ColorNoiseConfig& cfg, cv::Mat& img, std::mt19937_64* rng);

class PCALightingNoise {
public:
  double scale;
  double stddev;
  array<double, 3> eval;
  array<double, 9> evec;
};

template <typename LightingNoiseConfig>
void LightingNoiseImage(const LightingNoiseConfig& cfg, cv::Mat& img, std::mt19937_64* rng);

} // namespace io
} // namespace arraydiff

#endif
