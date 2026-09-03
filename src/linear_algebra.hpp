#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

#include <flint/fmpz.h>
#include <flint/nmod.h>

#include "coeff.hpp"
#include "diagnostics.hpp"
#include "solve_state.hpp"
#include "sparse_rref/sparse_mat.h"
#include "sparse_rref/sparse_vec.h"
#include "sparse_rref/thread_pool.hpp"

namespace f4ncgb {

typedef sparse_vec_t<uint32_t> uint32_vec_t;
typedef sparse_mat_t<uint32_t> uint32_mat_t;
typedef sparse_mat_ro_t<uint32_t> uint32_mat_ro_t;
using pivots = std::vector<size_t>;

inline void
check_offset(size_t value, const char* what) {
  if constexpr(sizeof(mat_offset_t) < sizeof(size_t)) {
    if(value > std::numeric_limits<mat_offset_t>::max())
      die(21,
          "%s reached %zu, past what a %zu-bit matrix offset can address. "
          "Rebuild with -DWIDE_MATRIX_OFFSETS=ON.",
          what,
          value,
          8 * sizeof(mat_offset_t));
  } else {
    (void)value;
    (void)what;
  }
}

inline void
check_fits(size_t value, size_t limit, const char* what) {
  if(value > limit)
    die(
      21, "%s reached %zu, past the %zu this can address.", what, value, limit);
}

// Column indices of a sparse matrix: 
// row i is [row_start[i], row_start[i + 1]) of `data`.
struct csr_indices {
  std::vector<uint32_t> data;
  std::vector<mat_offset_t> row_start{ 0 };

  void clear() {
    data.clear();
    row_start.assign(1, 0);
  }
  void push(uint32_t column) { data.push_back(column); }
  void end_row() {
    check_offset(data.size(), "number of nonzeros in a matrix block");
    row_start.push_back(static_cast<mat_offset_t>(data.size()));
  }

  size_t rows() const { return row_start.size() - 1; }
  size_t nnz() const { return data.size(); }
  size_t bytes() const {
    return data.capacity() * sizeof(uint32_t)
           + row_start.capacity() * sizeof(mat_offset_t);
  }
  std::span<const uint32_t> operator[](size_t i) const {
    return std::span<const uint32_t>(data.data() + row_start[i],
                                     row_start[i + 1] - row_start[i]);
  }
};

std::pair<std::vector<std::pair<size_t, size_t>>, fmpz*>
linear_algebra(const solve_state& state,
               const csr_indices& idxs_spol,
               const csr_indices& idxs_red,
               const std::vector<size_t>& red_order,
               std::vector<std::span<coeff>>& entries_spol,
               std::vector<std::span<coeff>>& entries_red,
               size_t characteristic,
               std::unique_ptr<BS::thread_pool<BS::none>>& pool,
               bool tracer = true);
}
