#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#include <cuda/std/cassert>
#else
#define CUDA_CALLABLE_MEMBER
#include <cassert>
#endif

template<typename T>
class Stack {
 public:
  Stack() = default;

  CUDA_CALLABLE_MEMBER   Stack(const Stack& other) = delete;
  CUDA_CALLABLE_MEMBER   Stack& operator=(const Stack& other) = delete;

  CUDA_CALLABLE_MEMBER   size_t Size() const;
  CUDA_CALLABLE_MEMBER   bool IsEmpty() const;

  CUDA_CALLABLE_MEMBER   T& Top();

  CUDA_CALLABLE_MEMBER   void Push(const T& value);
  CUDA_CALLABLE_MEMBER  void Pop();

  CUDA_CALLABLE_MEMBER  void Clear();

  CUDA_CALLABLE_MEMBER  ~Stack();

 private:
  struct Node {
    T value_;
    Node* previous_;
  };

  Node* top_{nullptr};
  size_t size_{0};
};

template<typename T>
CUDA_CALLABLE_MEMBER  size_t Stack<T>::Size() const {
  return size_;
}

template<typename T>
CUDA_CALLABLE_MEMBER  bool Stack<T>::IsEmpty() const {
  return size_ == 0;
}

template<typename T>
CUDA_CALLABLE_MEMBER  T& Stack<T>::Top() {
  assert(!IsEmpty());

  return top_->value_;
}

template<typename T>
CUDA_CALLABLE_MEMBER  void Stack<T>::Push(const T& value) {
  Node* new_top = new Node;

  new_top->previous_ = top_;
  new_top->value_ = value;
  top_ = new_top;
  ++size_;
}

template<typename T>
CUDA_CALLABLE_MEMBER  void Stack<T>::Pop() {
  assert(!IsEmpty());
  Node* to_delete = top_;
  top_ = top_->previous_;
  --size_;

  delete to_delete;
}

template<typename T>
CUDA_CALLABLE_MEMBER  void Stack<T>::Clear() {
  while (!IsEmpty()) {
    Pop();
  }
}

template<typename T>
CUDA_CALLABLE_MEMBER  Stack<T>::~Stack() {
  Clear();
}
