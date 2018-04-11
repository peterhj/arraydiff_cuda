#ifndef ARRAYDIFF_ARRAY_ABSTRACT_HH
#define ARRAYDIFF_ARRAY_ABSTRACT_HH

#include <array>
#include <cstddef>
#include <iostream>

namespace arraydiff {

using std::array;

class Shape0D;
class Shape1D;
class Shape2D;
class Shape3D;
class Shape4D;

class Shape4D {
public:
  static Shape4D Zero() {
    return Shape4D(0, 0, 0, 0);
  }

  Shape4D() {}
  Shape4D(size_t d0, size_t d1, size_t d2, size_t d3)
    : shape({ d0, d1, d2, d3 }) {}

  size_t major() const {
    return shape.at(0);
  }

  size_t minor() const {
    return shape.at(3);
  }

  size_t flat_size() const {
    return shape.at(0) * shape.at(1) * shape.at(2) * shape.at(3);
  }

  Shape4D packed_stride() const {
    Shape4D packed(0, 0, 0, 0);
    packed.shape.at(0) = 1UL;
    packed.shape.at(1) = packed.shape.at(0) * shape.at(0);
    packed.shape.at(2) = packed.shape.at(1) * shape.at(1);
    packed.shape.at(3) = packed.shape.at(2) * shape.at(2);
    return packed;
  }

  Shape4D packed_dim(size_t minor) const {
    Shape4D dim(0, 0, 0, 0);
    dim.shape.at(0) = shape.at(1) / shape.at(0);
    dim.shape.at(1) = shape.at(2) / shape.at(1);
    dim.shape.at(2) = shape.at(3) / shape.at(2);
    dim.shape.at(3) = minor;
    return dim;
  }

  bool is_packed(const Shape4D& stride) const {
    auto packed = packed_stride();
    return packed.shape.at(0) == stride.shape.at(0)
        && packed.shape.at(1) == stride.shape.at(1)
        && packed.shape.at(2) == stride.shape.at(2)
        && packed.shape.at(3) == stride.shape.at(3);
  }

  bool is_zero() const {
    return shape.at(0) == 0
        && shape.at(1) == 0
        && shape.at(2) == 0
        && shape.at(3) == 0;
  }

  size_t to_index(const Shape4D& stride) const {
    //std::clog << "DEBUG: Shape4D: to_index(): (" << shape.at(0) << " " << shape.at(1) << " " << shape.at(2) << " " << shape.at(3) << ") (" << stride.shape.at(0) << " " << stride.shape.at(1) << " " << stride.shape.at(2) << " " << stride.shape.at(3) << ")" << std::endl;
    return
        stride.shape.at(0) * shape.at(0) +
        stride.shape.at(1) * shape.at(1) +
        stride.shape.at(2) * shape.at(2) +
        stride.shape.at(3) * shape.at(3);
  }

  void _debug_print() const {
    std::clog << "Shape4D(" <<
        shape.at(0) << ", " <<
        shape.at(1) << ", " <<
        shape.at(2) << ", " <<
        shape.at(3) << ")";
  }

  array<size_t, 4> shape;
};

class Shape3D {
public:
  typedef Shape4D Upper;

  static Shape3D Zero() {
    return Shape3D(0, 0, 0);
  }

  Shape3D() {}
  Shape3D(size_t d0, size_t d1, size_t d2)
    : shape({ d0, d1, d2 }) {}

  size_t major() const {
    return shape.at(0);
  }

  size_t minor() const {
    return shape.at(2);
  }

  size_t flat_size() const {
    return shape.at(0) * shape.at(1) * shape.at(2);
  }

  Shape4D append(size_t minor_dim) const {
    return Shape4D(shape.at(0), shape.at(1), shape.at(2), minor_dim);
  }

  Shape3D packed_stride() const {
    Shape3D packed(0, 0, 0);
    packed.shape.at(0) = 1UL;
    packed.shape.at(1) = packed.shape.at(0) * shape.at(0);
    packed.shape.at(2) = packed.shape.at(1) * shape.at(1);
    return packed;
  }

  Shape3D packed_dim(size_t minor) const {
    Shape3D dim(0, 0, 0);
    dim.shape.at(0) = shape.at(1) / shape.at(0);
    dim.shape.at(1) = shape.at(2) / shape.at(1);
    dim.shape.at(2) = minor;
    return dim;
  }

  bool is_packed(const Shape3D& stride) const {
    auto packed = packed_stride();
    return packed.shape.at(0) == stride.shape.at(0)
        && packed.shape.at(1) == stride.shape.at(1)
        && packed.shape.at(2) == stride.shape.at(2);
  }

  bool is_zero() const {
    return shape.at(0) == 0
        && shape.at(1) == 0
        && shape.at(2) == 0;
  }

  size_t to_index(const Shape3D& stride) const {
    //std::clog << "DEBUG: Shape4D: to_index(): (" << shape.at(0) << " " << shape.at(1) << " " << shape.at(2) << " " << ") (" << stride.shape.at(0) << " " << stride.shape.at(1) << " " << stride.shape.at(2) << " " << ")" << std::endl;
    return
        stride.shape.at(0) * shape.at(0) +
        stride.shape.at(1) * shape.at(1) +
        stride.shape.at(2) * shape.at(2);
  }

  void _debug_print() const {
    std::clog << "Shape3D(" <<
        shape.at(0) << ", " <<
        shape.at(1) << ", " <<
        shape.at(2) << ")";
  }

  array<size_t, 3> shape;
};

class Shape2D {
public:
  typedef Shape3D Upper;

  static Shape2D Zero() {
    return Shape2D(0, 0);
  }

  Shape2D() {}
  Shape2D(size_t d0, size_t d1)
    : shape({ d0, d1 }) {}

  size_t major() const {
    return shape.at(0);
  }

  size_t minor() const {
    return shape.at(1);
  }

  size_t flat_size() const {
    return shape.at(0) * shape.at(1);
  }

  Shape3D append(size_t minor_dim) const {
    return Shape3D(shape.at(0), shape.at(1), minor_dim);
  }

  Shape2D packed_stride() const {
    Shape2D packed(0, 0);
    packed.shape.at(0) = 1UL;
    packed.shape.at(1) = packed.shape.at(0) * shape.at(0);
    return packed;
  }

  Shape2D packed_dim(size_t minor) const {
    Shape2D dim(0, 0);
    dim.shape.at(0) = shape.at(1) / shape.at(0);
    dim.shape.at(1) = minor;
    return dim;
  }

  bool is_packed(const Shape2D& stride) const {
    auto packed = packed_stride();
    return packed.shape.at(0) == stride.shape.at(0)
        && packed.shape.at(1) == stride.shape.at(1);
  }

  bool is_zero() const {
    return shape.at(0) == 0
        && shape.at(1) == 0;
  }

  size_t to_index(const Shape2D& stride) const {
    //std::clog << "DEBUG: Shape4D: to_index(): (" << shape.at(0) << " " << shape.at(1) << " " << ") (" << stride.shape.at(0) << " " << stride.shape.at(1) << ")" << std::endl;
    return
        stride.shape.at(0) * shape.at(0) +
        stride.shape.at(1) * shape.at(1);
  }

  void _debug_print() const {
    std::clog << "Shape2D(" <<
        shape.at(0) << ", " <<
        shape.at(1) << ")";
  }

  array<size_t, 2> shape;
};

class Shape1D {
public:
  typedef Shape2D Upper;

  static Shape1D Zero() {
    return Shape1D(0);
  }

  Shape1D() {}
  explicit Shape1D(size_t d0)
    : shape(d0) {}

  size_t major() const {
    return shape;
  }

  size_t minor() const {
    return shape;
  }

  size_t flat_size() const {
    return shape;
  }

  Shape2D append(size_t minor_dim) const {
    return Shape2D(shape, minor_dim);
  }

  Shape1D packed_stride() const {
    return Shape1D(1);
  }

  Shape1D packed_dim(size_t minor) const {
    return Shape1D(minor);
  }

  bool is_packed(const Shape1D& stride) const {
    auto packed = packed_stride();
    return packed.shape == stride.shape;
  }

  bool is_zero() const {
    return shape == 0;
  }

  size_t to_index(const Shape1D& stride) const {
    //std::clog << "DEBUG: Shape4D: to_index(): (" << shape << ") (" << stride.shape << ")" << std::endl;
    return stride.shape * shape;
  }

  void _debug_print() const {
    std::clog << "Shape1D(" << shape << ")";
  }

  size_t shape;
};

class Shape0D {
public:
  typedef Shape1D Upper;

  static Shape0D Zero() {
    return Shape0D();
  }

  Shape0D() {}

  size_t major() const {
    return 1;
  }

  size_t minor() const {
    return 1;
  }

  size_t flat_size() const {
    return 1;
  }

  Shape1D append(size_t minor_dim) const {
    return Shape1D(minor_dim);
  }

  Shape0D packed_stride() const {
    return Shape0D();
  };

  Shape0D packed_dim(size_t minor) const {
    (void)minor;
    return Shape0D();
  }

  bool is_zero() const {
    return true;
  }

  bool is_packed(const Shape0D& stride) const {
    (void)stride;
    return true;
  }

  void _debug_print() const {
    std::clog << "Shape0D()" << std::endl;
  }
};

class ScalarDesc {
public:
  size_t flat_size() const {
    return 1;
  }
};

class TensorDesc {
public:
  size_t flat_size() const {
    return channels;
  }

  size_t channels;
};

class FilterDesc {
public:
  size_t flat_size() const {
    return channels_out * channels_in;
  }

  size_t channels_out;
  size_t channels_in;
};

class TensorDesc2D {
public:
  size_t flat_size() const {
    return channels * height * width;
  }

  size_t channels;
  size_t height;
  size_t width;
};

class FilterDesc2D {
public:
  size_t flat_size() const {
    return channels_out * channels_in * kernel_height * kernel_width;
  }

  size_t channels_out;
  size_t channels_in;
  size_t kernel_height;
  size_t kernel_width;
};

template <typename ShapeDesc>
class ArrayDesc {
public:
  ShapeDesc shape() const;
};

template <typename ShapeDesc>
class BatchArrayDesc {
public:
  ShapeDesc shape() const;
  size_t batch_capacity() const;
};

} // arraydiff

#endif
