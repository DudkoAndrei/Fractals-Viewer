#pragma once

#include <cassert>
#include <string>

#include "color.h"

class Image {
 public:
  template <typename ValueT>
  class ImageRow {
   public:
    ValueT& operator[](int index) const;

    friend class Image;

   private:
    ImageRow(ValueT* image, int index, int len);
    ValueT* row_{nullptr};
    int len_{0};
  };
  using Row = ImageRow<Color>;
  using ConstRow = ImageRow<const Color>;

 public:
  Image() = default;
  Image(int n, int m, Color color = {0, 0, 0});
  Image(const Image& other);
  Image(Image&& other) noexcept;
  Image& operator=(const Image& other);
  Image& operator=(Image&& other) noexcept;
  int GetWidth() const;
  int GetHeight() const;

  Row operator[](int index);
  ConstRow operator[](int index) const;

  bool operator==(const Image& rhs) const;
  bool operator!=(const Image& rhs) const;

  ~Image();

  void LoadFromFile(const std::string& filename);
  void WriteToFile(const std::string& filename) const;

 private:
  void Delete();

  Color* data_{nullptr};
  int width_{0};
  int height_{0};
};

template<typename ValueT>
Image::ImageRow<ValueT>::ImageRow(ValueT* image, int index, int len) :
    row_(image + index * len), len_(len) {}

template<typename ValueT>
ValueT& Image::ImageRow<ValueT>::operator[](int index) const {
  assert(0 <= index && index < len_);
  return row_[index];
}
