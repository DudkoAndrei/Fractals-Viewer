#pragma once

#include <optional>
#include <cassert>

template<typename T>
class MinMaxContainer {
 public:
  MinMaxContainer() = default;
  MinMaxContainer(std::initializer_list<T> list);

  template<typename U>
  void Push(U&& value);

  const T& GetMin() const;
  const T& GetMax() const;

 private:
  std::optional<T> min_;
  std::optional<T> max_;
};

template<typename T>
MinMaxContainer<T>::MinMaxContainer(std::initializer_list<T> list) {
  for (const auto& item : list) {
    Push(item);
  }
}

template<typename T>
template<typename U>
void MinMaxContainer<T>::Push(U&& value) {
  if (min_ == std::nullopt || min_.value() > value) {
    min_ = value;
  }
  if (max_ == std::nullopt || max_.value() < value) {
    max_ = value;
  }
}

template<typename T>
const T& MinMaxContainer<T>::GetMin() const {
  assert(min_.has_value());
  return min_.value();
}

template<typename T>
const T& MinMaxContainer<T>::GetMax() const {
  assert(max_.has_value());
  return max_.value();
}
