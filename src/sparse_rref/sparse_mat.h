#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include "sparse_vec.h"
#include <cstdint>
#include <utility>

#ifdef F4NCGB_WIDE_MATRIX_OFFSETS
using mat_offset_t = uint64_t;
#else
using mat_offset_t = uint32_t;
#endif

template<typename T>
struct sparse_mat_ro_struct {
  ulong nrow;
  ulong ncol;
  ulong trace_len;
  ulong unique_nnz;

  mat_offset_t* row_start;
  mat_offset_t* index_start;
  uint32_t* row_nnz;
  uint32_t* indices;
  unsigned char* trace;
  bool owns_indices;
};

template<typename T>
struct sparse_mat_struct {
  ulong nrow;
  ulong ncol;
  sparse_vec_struct<T>* rows;
};

template<typename T>
using sparse_mat_ro_t = sparse_mat_ro_struct<T>[1];
template<typename T>
using sparse_mat_t = struct sparse_mat_struct<T>[1];

#define sparse_mat_row(mat, ind) ((mat)->rows + (ind))

#define red_mat_entries(ro, values, ind) ((values) + (ro)->row_start[(ind)])

#define red_mat_indices(ro, ind) ((ro)->indices + (ro)->index_start[(ind)])

#define red_mat_nnz(ro, ind) ((ro)->row_nnz[(ind)])

template<typename T>
void
sparse_mat_ro_init(sparse_mat_ro_t<T> ro,
                   ulong nrow,
                   ulong ncol,
                   ulong nnz,
                   ulong unique_nnz,
                   uint32_t* borrowed = NULL) {
  ro->nrow = nrow;
  ro->ncol = ncol;
  ro->trace_len = 0;
  ro->unique_nnz = unique_nnz;
  ro->trace = NULL;
  ro->owns_indices = (borrowed == NULL);

  if(nrow > 0) {
    ro->row_start = s_malloc<mat_offset_t>(nrow);
    ro->index_start = s_malloc<mat_offset_t>(nrow);
    ro->row_nnz = s_malloc<uint32_t>(nrow);
    ro->indices = borrowed ? borrowed : s_malloc<uint32_t>(nnz);
  } else {
    ro->row_start = NULL;
    ro->index_start = NULL;
    ro->row_nnz = NULL;
    ro->indices = NULL;
    ro->owns_indices = true;
  }
}

template<typename T>
inline void
sparse_mat_init(sparse_mat_t<T> mat, ulong nrow, ulong ncol) {
  mat->nrow = nrow;
  mat->ncol = ncol;
  mat->rows = s_malloc<sparse_vec_struct<T>>(nrow);
}

template<typename T>
inline void
sparse_mat_ro_clear(sparse_mat_ro_t<T> ro) {
  s_free(ro->row_start);
  s_free(ro->index_start);
  s_free(ro->row_nnz);
  if(ro->owns_indices)
    s_free(ro->indices);
  s_free(ro->trace);
  ro->nrow = 0;
  ro->ncol = 0;
  ro->trace_len = 0;
  ro->unique_nnz = 0;

  ro->trace = NULL;
  ro->row_start = NULL;
  ro->index_start = NULL;
  ro->row_nnz = NULL;
  ro->indices = NULL;
}

template<typename T>
inline void
sparse_mat_clear(sparse_mat_t<T> mat) {
  for(size_t i = 0; i < mat->nrow; i++)
    sparse_vec_clear(sparse_mat_row(mat, i));
  s_free(mat->rows);
  mat->nrow = 0;
  mat->ncol = 0;
  mat->rows = NULL;
}

template<typename T>
inline sparse_vec_struct<T>*
sparse_mat_row_init(sparse_mat_t<T> mat, ulong i, ulong alloc) {
  sparse_vec_init(sparse_mat_row(mat, i), alloc);
  return sparse_mat_row(mat, i);
}

template<typename T>
inline void
sparse_mat_init_trace(sparse_mat_ro_t<T> mat, ulong len) {
  mat->trace = s_malloc<unsigned char>(len);
  mat->trace_len = len;
  std::fill(mat->trace, mat->trace + len, (unsigned char)0);
}

template<typename T>
inline ulong
sparse_mat_nnz(const sparse_mat_t<T> mat) {
  ulong nnz = 0;
  for(size_t i = 0; i < mat->nrow; i++)
    nnz += sparse_mat_row(mat, i)->nnz;
  return nnz;
}

template<typename T>
inline T
sparse_mat_entry(sparse_mat_t<T> mat, ulong row, ulong col) {
  return sparse_vec_entry(sparse_mat_row(mat, row), col);
}

template<typename T>
void
sparse_mat_print(const sparse_mat_ro_t<T> mat, const T* values) {
  for(unsigned long i = 0; i < mat->nrow; i++) {
    unsigned long idx_off = mat->index_start[i];
    unsigned long ent_off = mat->row_start[i];
    unsigned long nnz = mat->row_nnz[i];

    unsigned long k = 0;
    for(unsigned long j = 0; j < mat->ncol; j++) {
      if(k < nnz && mat->indices[idx_off + k] == j) {
        std::cout << values[ent_off + k] << " ";
        k++;
      } else {
        std::cout << 0 << " ";
      }
    }
    std::cout << "\n";
  }
}

template<typename T>
void
sparse_mat_print(sparse_mat_t<T> mat) {
  std::cout << mat->nrow << ' ' << mat->ncol << ' ' << sparse_mat_nnz(mat)
            << '\n';
  for(size_t i = 0; i < mat->nrow; i++) {
    for(size_t j = 0; j < mat->ncol; j++) {
      auto c = sparse_mat_entry(mat, i, j);
      std::cout << c << ' ';
    }
    std::cout << "\n";
  }
}

#endif
