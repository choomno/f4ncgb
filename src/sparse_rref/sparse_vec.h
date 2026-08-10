#ifndef SPARSE_VEC_H
#define SPARSE_VEC_H

#include "flint/fmpz.h"
#include <algorithm>
#include <iostream>

// Memory management

template<typename T>
inline T*
s_malloc(const size_t size) {
  return (T*)std::malloc(size * sizeof(T));
}

template<typename T>
inline void
s_free(T* s) {
  std::free(s);
}

template<typename T>
inline T*
s_realloc(T* s, const size_t size) {
  assert(size > 0);
  return (T*)std::realloc(s, size * sizeof(T));
}

// Scalar basics

template<typename T>
struct sparse_vec_struct {
  uint32_t nnz = 0;
  uint32_t alloc = 0;
  uint32_t* indices = NULL;
  T* entries = NULL;
};

template<typename T>
using sparse_vec_t = struct sparse_vec_struct<T>[1];

// sparse_vec

// memory management
template<typename T>
void
sparse_vec_realloc(sparse_vec_t<T> vec, uint32_t alloc) {

  if(alloc == vec->alloc)
    return;

  vec->alloc = alloc;
  vec->indices = s_realloc(vec->indices, vec->alloc);
  vec->entries = s_realloc(vec->entries, vec->alloc);
}

#define sparse_vec_entry_pointer(vec, index) ((vec)->entries + (index))

template<typename T>
inline void
sparse_vec_init(sparse_vec_t<T> vec, uint32_t alloc = 0) {
  vec->nnz = 0;
  vec->alloc = alloc;
  vec->indices = s_malloc<uint32_t>(alloc);
  vec->entries = s_malloc<T>(alloc);
}

// set zero and clear memory
template<typename T>
inline void
sparse_vec_clear(sparse_vec_t<T> vec) {
  s_free(vec->indices);
  s_free(vec->entries);
  vec->indices = NULL;
  vec->entries = NULL;
  vec->nnz = 0;
  vec->alloc = 0;
}

template<typename T>
inline T
sparse_vec_entry(sparse_vec_t<T> vec, uint32_t index) {

  if(vec->nnz == 0)
    return 0;

  auto it = std::lower_bound(vec->indices, vec->indices + vec->nnz, index);
  if(it != vec->indices + vec->nnz && *it == index) {
    int pos = it - vec->indices;
    return vec->entries[pos];
  }
  return 0;
}

// debug only, not used to the large vector
template<typename T>
void
sparse_vec_print(const sparse_vec_t<T> vec) {
  std::cout << "-------------------" << std::endl;
  std::cout << "nnz: " << vec->nnz << std::endl;
  std::cout << "alloc: " << vec->alloc << std::endl;
  std::cout << "indices: ";
  for(size_t i = 0; i < vec->nnz; i++)
    std::cout << vec->indices[i] << " ";
  std::cout << "\nentries: ";
  for(size_t i = 0; i < vec->nnz; i++)
    std::cout << *(vec->entries + i) << " ";
  std::cout << std::endl;
}

#endif
