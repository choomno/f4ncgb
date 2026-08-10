#pragma once

#include <cstddef>
#include <vector>

#include <flint/fmpz.h>
#include <flint/nmod.h>

#include "coeff.hpp"
#include "solve_state.hpp"
#include "sparse_rref/sparse_mat.h"
#include "sparse_rref/sparse_vec.h"
#include "sparse_rref/thread_pool.hpp"

namespace f4ncgb {

typedef sparse_vec_t<uint32_t> uint32_vec_t;
typedef sparse_mat_t<uint32_t> uint32_mat_t;
typedef sparse_mat_ro_t<uint32_t> uint32_mat_ro_t;
using pivots = std::vector<size_t>;

/// Column indices of a sparse matrix, one flat array instead of a vector per
/// row: row i is [row_start[i], row_start[i + 1]) of `data`.
struct csr_indices {
  std::vector<uint32_t> data;
  std::vector<uint32_t> row_start{ 0 };

  /// Keeps the capacity, which is the point of reusing this across iterations.
  void clear() {
    data.clear();
    row_start.assign(1, 0);
  }
  void push(uint32_t column) { data.push_back(column); }
  void end_row() { row_start.push_back(static_cast<uint32_t>(data.size())); }

  size_t rows() const { return row_start.size() - 1; }
  std::span<const uint32_t> operator[](size_t i) const {
    return std::span<const uint32_t>(data.data() + row_start[i],
                                     row_start[i + 1] - row_start[i]);
  }
};

std::pair<std::vector<std::pair<size_t, size_t>>, fmpz*>
linear_algebra(const solve_state& state,
               const csr_indices& idxs_spol,
               const csr_indices& idxs_red,
               std::vector<std::span<coeff>>& entries_spol,
               std::vector<std::span<coeff>>& entries_red,
               size_t characteristic,
               std::unique_ptr<BS::thread_pool<BS::none>>& pool,
               bool tracer = true);
}
