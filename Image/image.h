#pragma once

#include <string>

#include "color.h"

class Image {
 public:
  Image() = default;
  Image(int n, int m, Color color = {0, 0, 0});
  Image(const Image& other);
  Image(Image&& other) noexcept;
  Image& operator=(const Image& other);
  Image& operator=(Image&& other) noexcept;

  int GetWidth() const;
  int GetHeight() const;

  Color& Get(int i, int j);
  const Color& Get(int i, int j) const;

  bool operator==(const Image& rhs) const;
  ~Image();

  void LoadFromFile(const std::string& filename);
  void WriteToFile(const std::string& filename) const;

 private:
  void Delete();

  Color* data_{nullptr};
  int width_{0};
  int height_{0};
};


