#include <algorithm>
#include <cassert>
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

Image::Image(int n, int m, Color color) {
  width_ = n;
  height_ = m;
  data_ = new Color[n * m];
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      data_[i * width_ + j] = color;
    }
  }
}

Image::Image(const Image& other) {
  *this = other;
}

Image::Image(Image&& other) noexcept {
  *this = std::move(other);
}

Image& Image::operator=(const Image& other) {
  if (this == &other) {
    return *this;
  }
  Delete();
  height_ = other.height_;
  width_ = other.width_;
  data_ = new Color[height_ * width_];
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      int index = i * width_ + j;
      data_[index] = other.data_[index];
    }
  }
  return *this;
}

Image& Image::operator=(Image&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  std::swap(data_, other.data_);
  std::swap(width_, other.width_);
  std::swap(height_, other.height_);
  return *this;
}

int Image::GetWidth() const {
  return width_;
}

int Image::GetHeight() const {
  return height_;
}

Color& Image::Get(int i, int j) {
  assert(0 <= i && i < height_ && 0 <= j && j < width_);
  return data_[i * width_ + j];
}

const Color& Image::Get(int i, int j) const {
  assert(0 <= i && i < height_ && 0 <= j && j < width_);
  return data_[i * width_ + j];
}

bool Image::operator==(const Image& rhs) const {
  if (width_ != rhs.width_) {
    return false;
  }
  if (height_ != rhs.height_) {
    return false;
  }
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      int index = i * width_ + j;
      if (data_[index] != rhs.data_[index]) {
        return false;
      }
    }
  }
  return true;
}

Image::~Image() {
  Delete();
}

void Image::Delete() {
  width_ = 0;
  height_ = 0;
  delete[] data_;
}

void Image::LoadFromFile(const std::string& filename) {
  int width, height, bpp;
  uint8_t* image = stbi_load(filename.c_str(), &width, &height, &bpp, 3);
  assert(image);
  assert(width > 0);
  assert(height > 0);
  assert(bpp == 3);

  *this = Image(width, height);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int pixel_pos = width * 3 * i + 3 * j;
      Get(i, j).r() = image[pixel_pos];
      Get(i, j).g() = image[pixel_pos + 1];
      Get(i, j).b() = image[pixel_pos + 2];
    }
  }

  stbi_image_free(image);
}

void Image::WriteToFile(const std::string& filename) const {
  assert(height_ > 0);
  assert(width_ > 0);

  int stb_image_size = width_ * height_ * 3;
  auto* image = new uint8_t[stb_image_size];
  for (int i = 0; i < stb_image_size; i += 3) {
    int pixel_pos = i / 3;
    image[i] = data_[pixel_pos].r();
    image[i + 1] = data_[pixel_pos].g();
    image[i + 2] = data_[pixel_pos].b();
  }
  stbi_write_bmp(filename.c_str(), width_, height_, 3, image);
}
