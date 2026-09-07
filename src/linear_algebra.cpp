#include <algorithm>
#include <atomic>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "coeff.hpp"
#include "diagnostics.hpp"
#include "linear_algebra.hpp"
#include "linear_algebra_internal.hpp"
#include "multimodular.hpp"
#include "primes.hpp"
#include "sparse_rref/sparse_mat.h"

extern int verbose;

namespace f4ncgb {

void inline set_up_reducer_mat(la_context& la,
                               const solve_state& state,
                               const csr_indices& idxs,
                               const std::vector<size_t>& order,
                               const std::vector<std::span<coeff>>& entries) {
  auto& red_mat = la.red_mat;
  const size_t nrows = idxs.rows();
  size_t ncols = state.nr_cols;
  assert(order.size() == nrows);

  boost::unordered_flat_map<const coeff*, uint32_t> first_row;
  first_row.reserve(nrows);
  // Dropped again at the end of this function, once the bits are set.
  std::vector<uint32_t> ref_row(nrows);

  size_t nnz = 0;
  size_t unique_nnz = 0;
  for(size_t i = 0; i < nrows; i++) {
    const size_t len = idxs[order[i]].size();
    nnz += len;
    auto [it, inserted]
      = first_row.try_emplace(entries[i].data(), static_cast<uint32_t>(i));
    ref_row[i] = it->second;
    if(inserted)
      unique_nnz += len;
  }

  if(state.proof_level > 0) {
    nnz += nrows;
    unique_nnz += first_row.size();
    ncols += la.nrows_total;
  }

  // check that we don't have overflows
  check_offset(nnz, "number of nonzeros in the reducer matrix");
  check_offset(unique_nnz, "number of stored reducer coefficients");
  check_fits(ncols, UINT32_MAX, "width of the reducer matrix");

  const bool borrow_indices = state.proof_level == 0;

  sparse_mat_ro_init(red_mat,
                     nrows,
                     ncols,
                     nnz,
                     unique_nnz,
                     borrow_indices ? const_cast<uint32_t*>(idxs.data.data())
                                    : nullptr);
  double ratio = nnz > 0 ? (100. * unique_nnz) / nnz : 0;
  if(verbose > 2)
    msg("Reducer matrix has size (%d, %d) (%.1f%% entries actually stored)",
        red_mat->nrow,
        red_mat->ncol,
        ratio);

  size_t idx_k = 0;
  size_t entry_k = 0;
  uint32_t* indices = red_mat->indices;
  mat_offset_t* index_start = red_mat->index_start;
  mat_offset_t* row_start = red_mat->row_start;
  uint32_t* row_nnz = red_mat->row_nnz;

  for(size_t i = 0; i < nrows; i++) {
    const auto row = idxs[order[i]];
    row_nnz[i] = row.size();

    if(borrow_indices) {
      index_start[i] = idxs.row_start[order[i]];
      idx_k += row.size();
    } else {
      index_start[i] = static_cast<mat_offset_t>(idx_k);
      for(const auto j : row)
        indices[idx_k++] = j;
    }

    // store coeff offset
    if(i == ref_row[i]) {
      row_start[i] = static_cast<mat_offset_t>(entry_k);
      entry_k += row.size();
    } else {
      row_start[i] = row_start[ref_row[i]];
    }

    // store transformation matrix
    if(state.proof_level > 0) {
      row_nnz[i] += 1;
      indices[idx_k++] = state.nr_cols + i;
      if(i == ref_row[i])
        entry_k++;
    }
  }

  la.rep_mask.assign(nrows / 64 + 1, 0);
  for(size_t i = 0; i < nrows; i++)
    if(i == ref_row[i])
      la.set_represents(i);
}
//------------------------------------------------------------------------------
bool inline fill_reducer_mat(la_context& la,
                             const solve_state& state,
                             const std::vector<std::span<coeff>>& entries,
                             nmod_t mod,
                             uint32_t* mat_entries) {
  auto& red_mat = la.red_mat;

  // only fill entries for first occurrence of each span
  for(size_t i = 0; i < red_mat->nrow; i++) {
    std::span<coeff> coeffs = entries[i];

    // just a duplicate -> nothing to do
    if(!la.represents(i))
      continue;

    // check leading coeff for zero
    uint32_t cc = fmpz_get_nmod(coeffs[0].value, mod);
    if(!cc)
      return false;

    // only write entries for first occurrence
    size_t k = red_mat->row_start[i];
    mat_entries[k++] = 1;
    uint32_t inv = nmod_inv(cc, mod);
    for(size_t j = 1; j < coeffs.size(); j++) {
      cc = fmpz_get_nmod(coeffs[j].value, mod);
      mat_entries[k++] = nmod_mul(inv, cc, mod);
    }
    if(state.proof_level > 0)
      mat_entries[k++] = 1;
  }
  return true;
}

//------------------------------------------------------------------------------
void inline set_reducer_pivots(la_context& la) {
  auto& red_mat = la.red_mat;
  auto& red_pivots = la.red_pivots;
  // clear
  red_pivots.resize(red_mat->ncol);
  std::fill(red_pivots.begin(), red_pivots.end(), -1);

  for(uint32_t i = 0; i < red_mat->nrow; i++) {
    uint32_t j = red_mat->indices[red_mat->index_start[i]];
    red_pivots[j] = static_cast<int32_t>(i);
  }
}

//------------------------------------------------------------------------------

bool inline spol_pivots_survive(const la_context& la,
                                const std::vector<std::span<coeff>>& entries,
                                nmod_t mod,
                                const unsigned char* trace) {
  (void)la;
  for(size_t i = 0; i < entries.size(); i++) {
    if(trace[i] or entries[i].empty())
      continue;
    if(!fmpz_get_nmod(entries[i][0].value, mod))
      return false;
  }
  return true;
}

//------------------------------------------------------------------------------

void
set_up_spol_matrix(const la_context& la,
                   const solve_state& state,
                   uint32_mat_t mat,
                   nmod_t mod,
                   const csr_indices& idxs,
                   const std::vector<std::span<coeff>>& entries,
                   const unsigned char* trace) {
  const size_t nrows = idxs.rows();

  // initialize matrix
  if(state.proof_level > 0)
    sparse_mat_init(mat, nrows, state.nr_cols + la.nrows_total);
  else
    sparse_mat_init(mat, nrows, state.nr_cols);

  if(verbose > 2)
    msg("S-Pol matrix has size (%d, %d) with %d nonzeros",
        mat->nrow,
        mat->ncol,
        idxs.data.size());

  // fill rows
  uint32_t offset = state.nr_cols + la.nrows_total - nrows;
  for(size_t i = 0; i < nrows; i++) {

    if(trace[i]) {
      // dummy init to be safe
      sparse_mat_row_init(mat, i, 0);
      continue;
    }

    std::span<coeff> coeffs = entries[i];
    const auto cols = idxs[i];

    const size_t max_nnz = coeffs.size() + (state.proof_level > 0 ? 1 : 0);

    auto row = sparse_mat_row_init(mat, i, max_nnz);

    size_t nnz = 0;
    // insert reduced entries
    for(size_t k = 0; k < coeffs.size(); k++) {
      uint32_t c = fmpz_get_nmod(coeffs[k].value, mod);
      if(c != 0) {
        row->indices[nnz] = cols[k];
        row->entries[nnz] = c;
        nnz++;
      }
    }
    assert(coeffs.empty() or nnz > 0);

    // transformation matrix
    if(state.proof_level > 0) {
      row->indices[nnz] = offset + i;
      // insert denom from input polynomial
      if(state.interreduce) {
        row->entries[nnz] = fmpz_get_nmod(state.input_denoms[i].value, mod);
      } else if(state.reduce_mode) {
        row->entries[nnz] = fmpz_get_nmod(state.reduce_denom_c.value, mod);
      } else
        row->entries[nnz] = fmpz_get_nmod(coeffs[0].value, mod);
      nnz++;
    }
    row->nnz = nnz;
  }
}
//------------------------------------------------------------------------------

template<bool mersenne = false>
static pivots
reverse_solve(uint32_mat_t mat, nmod_t mod) {
  const modulus<mersenne> m(mod);

  std::vector<int32_t> piv_array(mat->ncol, -1);
  std::vector<int64_t> buffer(mat->ncol, 0);
  std::vector<uint64_t> buffer_mask(mat->ncol / 64 + 1, 0);
  std::vector<uint32_t> buffer_ids;
  int64_t* buff = buffer.data();
  uint64_t* mask = buffer_mask.data();
  buffer_ids.reserve(32);

  // sort rows by first index and nnz
  // assume: mat is in ref
  std::sort(mat->rows, mat->rows + mat->nrow, [](auto& r1, auto& r2) {
    // move zero rows down
    if(r1.nnz == 0 or r2.nnz == 0)
      return r1.nnz > r2.nnz;
    // sort by pivots
    return r1.indices[0] > r2.indices[0];
  });

  // collect pivots -- assume: mat is in ref
  pivots piv;
  piv.reserve(64);
  for(size_t r = 0; r < mat->nrow; r++) {
    auto row = sparse_mat_row(mat, r);
    if(row->nnz == 0)
      break;
    piv_array[row->indices[0]] = static_cast<int32_t>(r);
    piv.push_back(row->indices[0]);
  }

  // reduce the new pivot rows fully
  for(size_t r = 0; r < piv.size(); r++) {
    auto row = sparse_mat_row(mat, r);
    // skip rows with only one entry
    if(row->nnz < 2)
      continue;

    copy_to_buffer(buff, mask, row);
    buffer_ids.clear();
    buffer_ids.push_back(row->indices[0]);
    int64_t cc;
    int32_t rr;
    uint32_t i = row->indices[1];
    uint32_t max_col = row->indices[row->nnz - 1];
    while(true) {
      i = next_marked(mask, i, max_col);
      if(i > max_col)
        break;

      if(buff[i] == 0) {
        unmark(mask, i);
        i++;
        continue;
      }

      cc = m.reduce(buff[i]);
      buff[i] = cc;
      if(cc == 0) {
        unmark(mask, i);
        i++;
        continue;
      }

      rr = piv_array[i];
      if(rr < 0) {
        buffer_ids.push_back(i);
      } else {
        assert(sparse_mat_row(mat, rr)->indices[0] == i);
        assert(sparse_mat_row(mat, rr)->entries[0] == 1);
        buff[i] = 0;
        unmark(mask, i);
        auto row_rr = sparse_mat_row(mat, rr);
        max_col = std::max(max_col, row_rr->indices[row_rr->nnz - 1]);
        xmay(
          buff, mask, cc, row_rr->indices, row_rr->entries, row_rr->nnz, m.p2);
      }
      i++;
    }
    copy_from_buffer_and_clear(buff, mask, buffer_ids, row);
  }

  return piv;
}

//------------------------------------------------------------------------------

// renumber columns after reduction by reducers to keep
// s-polynomial pass and the back substitution dense
static bool
compress_columns(uint32_mat_t mat, std::vector<uint32_t>& cols) {
  size_t nnz = 0;
  for(size_t r = 0; r < mat->nrow; r++)
    nnz += sparse_mat_row(mat, r)->nnz;
  if(nnz * 2 >= mat->ncol)
    return false;

  cols.clear();
  cols.reserve(nnz);
  for(size_t r = 0; r < mat->nrow; r++) {
    const auto row = sparse_mat_row(mat, r);
    cols.insert(cols.end(), row->indices, row->indices + row->nnz);
  }
  std::sort(cols.begin(), cols.end());
  cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

  for(size_t r = 0; r < mat->nrow; r++) {
    const auto row = sparse_mat_row(mat, r);
    for(uint32_t k = 0; k < row->nnz; k++)
      row->indices[k] = static_cast<uint32_t>(
        std::lower_bound(cols.begin(), cols.end(), row->indices[k])
        - cols.begin());
  }
  return true;
}

//------------------------------------------------------------------------------

static void
expand_columns(uint32_mat_t mat,
               const std::vector<uint32_t>& cols,
               pivots& piv) {
  for(size_t r = 0; r < mat->nrow; r++) {
    const auto row = sparse_mat_row(mat, r);
    for(uint32_t k = 0; k < row->nnz; k++)
      row->indices[k] = cols[row->indices[k]];
  }
  for(auto& c : piv)
    c = cols[c];
}

//------------------------------------------------------------------------------

// Reduces the rows of mat against the reducers, a block of columns and a
// batch of rows at a time.
//
// Block reduction does not always help. So a sample of the batches is run
// both ways and timed, and the rest of the matrix goes the way that won.
template<bool mersenne, uint32_t B, typename RowReduce>
static bool
blocked_reducer_pass(la_context& la,
                     const solve_state& state,
                     uint32_mat_t mat,
                     nmod_t mod,
                     BS::thread_pool<BS::none>* pool,
                     const uint32_t* red_values,
                     const unsigned char* trace,
                     RowReduce&& row_reduce) {

  // Rows that still have work. Batching them contiguously keeps the leading
  // columns within a batch close together, which is what makes its lanes hit
  // the same reducers.
  std::vector<size_t> live;
  live.reserve(mat->nrow);
  for(size_t r = 0; r < mat->nrow; r++)
    if(!trace[r] and sparse_mat_row(mat, r)->nnz > 0)
      live.push_back(r);
  if(live.empty())
    return true;

  const size_t nbatch = (live.size() + B - 1) / B;
  auto batch_size
    = [&](size_t b) { return std::min<size_t>(B, live.size() - b * B); };

  // The two ways of getting through the batches
  auto run_blocked
    = [&](const std::vector<size_t>& batches, size_t begin, size_t end) {
        F4NCGB_TIME(elim_task_cpu);
        thread_local blocked_reducer<mersenne, B> worker;
        worker.configure(la, state, mat, mod, red_values);
        for(size_t i = begin; i < end; i++)
          worker.run_batch(live.data() + batches[i] * B,
                           static_cast<uint32_t>(batch_size(batches[i])));
      };

  auto run_by_row
    = [&](const std::vector<size_t>& batches, size_t begin, size_t end) {
        for(size_t i = begin; i < end; i++) {
          const size_t first = batches[i] * B;
          const size_t count = batch_size(batches[i]);
          for(size_t k = 0; k < count; k++)
            row_reduce(live[first + k]);
        }
      };

  // Spreads `batches[from, to)` over the pool and waits for them.
  auto run_all =
    [&](auto& run, const std::vector<size_t>& batches, size_t from, size_t to) {
      if(from >= to)
        return;
      if(pool and to - from > 1) {
        const size_t block
          = std::max<size_t>(1, (to - from) / (pool->get_thread_count() * 32));
        for(size_t begin = from; begin < to; begin += block) {
          const size_t end = std::min(begin + block, to);
          pool->detach_task(
            [&batches, begin, end, &run]() { run(batches, begin, end); });
        }
        pool->wait();
      } else {
        run(batches, from, to);
      }
    };

  std::vector<size_t> rest;

  const size_t want = std::clamp<size_t>(nbatch / 8, 16, 512);
  const size_t stride = nbatch / want;

  if(state.la_pass == reducer_pass::calibrate and stride >= 2) {
    // The sampled batches alternate between the two halves, so that both see
    // the same stretch of the matrix; everything else waits in `rest`.
    std::vector<size_t> half[2];
    for(size_t b = 0, seen = 0; b < nbatch; b++) {
      if(b % stride)
        rest.push_back(b);
      else
        half[seen++ & 1].push_back(b);
    }

    double spent[2] = { 0, 0 };
    size_t rows[2] = { 0, 0 };
    size_t cursor[2] = { 0, 0 };

    auto spend = [&](size_t w, size_t count, bool clocked) {
      const size_t from = cursor[w];
      const size_t to = std::min(from + count, half[w].size());
      if(from >= to)
        return;
      const auto start = std::chrono::steady_clock::now();
      if(w == 0)
        run_all(run_blocked, half[0], from, to);
      else
        run_all(run_by_row, half[1], from, to);
      if(clocked) {
        spent[w] += std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - start)
                      .count();
        for(size_t i = from; i < to; i++)
          rows[w] += batch_size(half[w][i]);
      }
      cursor[w] = to;
    };

    auto rate = [&](size_t w) {
      return spent[w] / static_cast<double>(std::max<size_t>(1, rows[w]));
    };

    spend(0, 1, false);
    spend(1, 1, false);

    constexpr size_t rounds = 4;
    const size_t per_round
      = std::max<size_t>(1, (half[0].size() + rounds - 1) / rounds);
    for(size_t k = 0; k < rounds; k++) {
      spend(0, per_round, true);
      spend(1, per_round, true);
      if(rows[0] == 0 or rows[1] == 0)
        break;
      if(rate(0) * 2 <= rate(1) or rate(1) * 2 <= rate(0))
        break;
    }

    constexpr double margin = 1.5;
    const bool worth_it
      = rows[0] == 0 or rows[1] == 0 or rate(0) * margin <= rate(1);
    if(verbose > 2)
      msg("Sample: %.2f us/row blocked over %zu, %.2f us/row row at a time "
          "over %zu -> %s",
          1e6 * rate(0),
          rows[0],
          1e6 * rate(1),
          rows[1],
          worth_it ? "blocked" : "row at a time");
    if(!worth_it)
      return false;

    // process the rest
    for(size_t w = 0; w < 2; w++)
      rest.insert(rest.end(),
                  half[w].begin() + static_cast<long>(cursor[w]),
                  half[w].end());
    std::sort(rest.begin(), rest.end());
  } else {
    rest.resize(nbatch);
    std::iota(rest.begin(), rest.end(), size_t(0));
  }

  run_all(run_blocked, rest, 0, rest.size());
  return true;
}

//------------------------------------------------------------------------------

template<bool mersenne = false>
pivots
gauss_elim(la_context& la,
           const solve_state& state,
           uint32_mat_t mat,
           nmod_t mod,
           BS::thread_pool<BS::none>* pool,
           const uint32_t* red_values,
           const unsigned char* trace) {

  const modulus<mersenne> m(mod);

  thread_local std::vector<int64_t> buffer_local;
  thread_local std::vector<uint64_t> buffer_mask_local;
  thread_local std::vector<uint32_t> buffer_ids_local;

  std::unique_ptr<std::atomic_int32_t[]> pivot_owner;
  std::atomic_int32_t* atomic_pivots = nullptr;

  uint32_t zero_threshold
    = state.proof_level > 0 ? static_cast<uint32_t>(state.nr_cols) : UINT32_MAX;

  // One left-to-right pass over row r, eliminating against the pivot sets the
  // two flags select: the static reducer rows, the s-polynomial pivots found so
  // far, or both. Returns false once the row is zero.
  auto sweep = [&]<bool use_reducers, bool use_spolies>(size_t r) -> bool {
    auto row = sparse_mat_row(mat, r);
    int64_t* buff = buffer_local.data();
    uint64_t* mask = buffer_mask_local.data();

    copy_to_buffer(buff, mask, row);
    buffer_ids_local.clear();
    uint32_t i = row->indices[0];
    uint32_t max_col = row->indices[row->nnz - 1];

    while(true) {
      // find next nonzero entry
      i = next_marked(mask, i, max_col);
      if(i > max_col)
        break;

      // the mark may be stale
      if(buff[i] == 0) {
        unmark(mask, i);
        i++;
        continue;
      }

      assert(buff[i] > 0);
      // v must be smaller than 2^2b, i.e. 2^62
      assert(buff[i] < 4611686018427387904);
      const int64_t cc = m.reduce(buff[i]);
      buff[i] = cc;
      if(cc == 0) {
        unmark(mask, i);
        i++;
        continue;
      }

      const uint32_t* indices;
      const uint32_t* entries;
      uint32_t nnz;
      int32_t rr = -1;
      if constexpr(use_reducers)
        rr = la.red_pivots[i];

      if(rr >= 0) {
        indices = red_mat_indices(la.red_mat, rr);
        entries = red_mat_entries(la.red_mat, red_values, rr);
        nnz = red_mat_nnz(la.red_mat, rr);
      } else {
        if constexpr(use_spolies)
          rr = atomic_pivots[i].load(std::memory_order_acquire);
        if(rr < 0) {
          buffer_ids_local.push_back(i);
          i++;
          continue;
        }
        const auto pivot_row = sparse_mat_row(mat, rr);
        indices = pivot_row->indices;
        entries = pivot_row->entries;
        nnz = pivot_row->nnz;
      }

      buff[i] = 0;
      unmark(mask, i);
      max_col = std::max(max_col, indices[nnz - 1]);
      xmay(buff, mask, cc, indices, entries, nnz, m.p2);
      i++;
    }

    // we have a zero row
    if(buffer_ids_local.empty() or buffer_ids_local[0] >= zero_threshold) {
      sparse_vec_clear(row);
      for(const auto id : buffer_ids_local) {
        buff[id] = 0;
        unmark(mask, id);
      }
      if(la.use_trace)
        __atomic_store_n(&la.red_mat->trace[r], 1, __ATOMIC_RELAXED);
      return false;
    }
    copy_from_buffer_and_clear(buff, mask, buffer_ids_local, row);
    return true;
  };

  // The buffer is all zeros between rows, so growing it only has to zero the
  // part that was not there before. It never shrinks: the second pass runs on
  // renumbered columns and would otherwise make the next matrix pay for zeroing
  // the full width again.
  auto claim_buffers = [&mat]() {
    if(buffer_local.size() < mat->ncol)
      buffer_local.resize(mat->ncol, 0);
    if(buffer_mask_local.size() < mat->ncol / 64 + 1)
      buffer_mask_local.resize(mat->ncol / 64 + 1, 0);
  };

  // Reduces the survivors of the reducer pass against each other.
  auto eliminate = [&]<bool use_reducers>(size_t r) {
    if(trace[r])
      return;
    F4NCGB_TIME(elim_task_cpu);
    auto row = sparse_mat_row(mat, r);
    if(row->nnz == 0)
      return;
    claim_buffers();
    int32_t expected;
    do {
      if(!sweep.template operator()<use_reducers, true>(r))
        return;
      normalize_row(row, mod);
      expected = -1;
    } while(!atomic_pivots[row->indices[0]].compare_exchange_strong(
      expected,
      static_cast<int32_t>(r),
      std::memory_order_release,
      std::memory_order_relaxed));
  };

  auto reduce_by_spolies
    = [&](size_t r) { eliminate.template operator()<false>(r); };
  auto reduce_fused = [&](size_t r) { eliminate.template operator()<true>(r); };

  // Reduces row r against the reducers only, one row at a time
  auto reduce_by_reducers = [&](size_t r) {
    if(trace[r])
      return;
    F4NCGB_TIME(elim_task_cpu);
    if(sparse_mat_row(mat, r)->nnz == 0)
      return;
    claim_buffers();
    sweep.template operator()<true, false>(r);
  };

  auto run_pass = [&](auto& reduce) {
    if(pool) {
      // rows go out in blocks
      const size_t rows = mat->nrow;
      const size_t block
        = std::max<size_t>(1, rows / (pool->get_thread_count() * 32));
      for(size_t begin = 0; begin < rows; begin += block) {
        const size_t end = std::min(begin + block, rows);
        pool->detach_task([begin, end, &reduce]() {
          for(size_t r = begin; r < end; r++)
            reduce(r);
        });
      }
      pool->wait();
    } else {
      for(size_t r = 0; r < mat->nrow; r++)
        reduce(r);
    }
  };

  const bool blocked
    = state.la_pass != reducer_pass::row_at_a_time
      and blocked_reducer_pass<mersenne, blocked_lanes>(
        la, state, mat, mod, pool, red_values, trace, reduce_by_reducers);

  const bool fused = !blocked;

  std::vector<uint32_t> live_cols;
  const ulong full_ncol = mat->ncol;
  bool compressed = false;
  if(!fused) {
    compressed = compress_columns(mat, live_cols);
    if(compressed) {
      mat->ncol = live_cols.size();
      if(state.proof_level > 0)
        zero_threshold = static_cast<uint32_t>(
          std::lower_bound(live_cols.begin(),
                           live_cols.end(),
                           static_cast<uint32_t>(state.nr_cols))
          - live_cols.begin());
    }
  }

  pivot_owner = std::make_unique<std::atomic_int32_t[]>(mat->ncol);
  atomic_pivots = pivot_owner.get();
  for(size_t i = 0; i < mat->ncol; i++)
    atomic_pivots[i].store(-1, std::memory_order_relaxed);

  if(fused)
    run_pass(reduce_fused);
  else
    run_pass(reduce_by_spolies);

  pivots piv = mod.n == PRIMES[0] ? reverse_solve<true>(mat, mod)
                                  : reverse_solve<false>(mat, mod);
  if(compressed) {
    expand_columns(mat, live_cols, piv);
    mat->ncol = full_ncol;
  }
  return piv;
}

//------------------------------------------------------------------------------

static pivots
gauss_elim_dispatch(la_context& la,
                    const solve_state& state,
                    uint32_mat_t mat,
                    nmod_t mod,
                    BS::thread_pool<BS::none>* pool,
                    const uint32_t* red_values,
                    const unsigned char* trace) {
  if(mod.n == PRIMES[0])
    return gauss_elim<true>(la, state, mat, mod, pool, red_values, trace);
  return gauss_elim<false>(la, state, mat, mod, pool, red_values, trace);
}
//------------------------------------------------------------------------------

struct prime_result {
  uint32_t p = 0;
  bool usable = false;
  pivots piv;
  std::unique_ptr<sparse_mat_struct<uint32_t>> rref;
};

std::pair<std::vector<std::pair<size_t, size_t>>, fmpz*>
multimodular_gauss_elim(la_context& la,
                        const solve_state& state,
                        const csr_indices& idxs_spol,
                        std::vector<std::span<coeff>>& entries_spol,
                        std::vector<std::span<coeff>>& entries_red,
                        std::unique_ptr<BS::thread_pool<BS::none>>& pool) {

  std::vector<std::unique_ptr<sparse_mat_struct<uint32_t>>> rrefs;
  std::vector<uint32_t> used_primes;
  std::vector<sparse_mat_struct<uint32_t>*> rref_ptrs;
  pivots best_piv;

  std::vector<std::pair<size_t, size_t>> idxs;
  fmpz* nums = nullptr;
  fmpz* denoms = nullptr;

  size_t i = 0;

  coeff h = height(la, entries_spol, entries_red);
  coeff prod(1L);
  coeff M = 10000000UL * state.nr_cols * (h + 100) * h + 1;

  // Primes to gather before trying to reconstruct again. Zero while the
  // initial bound M is in charge; from the first failure on, this drives the
  // loop and grows geometrically.
  size_t target = 0;

  const size_t threads = pool ? pool->get_thread_count() : 1;
  std::vector<prime_result> batch;
  std::vector<unsigned char> trace_snapshot;

  while(true) {
    while(prod < M or used_primes.size() < target) {
      // how many primes are still wanted?
      size_t wanted
        = used_primes.size() < target ? target - used_primes.size() : 0;
      if(prod < M) {
        const flint_bitcnt_t missing
          = fmpz_bits(M.value)
            - std::min(fmpz_bits(M.value), fmpz_bits(prod.value));
        wanted = std::max<size_t>(wanted, (missing + 30) / 31);
      }
      wanted = std::max<size_t>(wanted, 1);
      wanted = std::min(wanted, MAX_PRIMES - std::min(i, MAX_PRIMES));
      if(wanted == 0)
        die(14, "Multimodular Gaussian elimination is not converging");

      const bool many = pool and not used_primes.empty()
                        and used_primes.size() + wanted >= threads;
      const size_t batch_size = many ? std::min(wanted, threads) : 1;
      const bool parallel = batch_size > 1;

      batch.clear();
      batch.resize(batch_size);
      for(auto& r : batch) {
        r.p = PRIMES[i++];
        if(verbose > 2)
          msg("Computing mod %lu", r.p);
      }

      trace_snapshot.assign(la.red_mat->trace,
                            la.red_mat->trace + la.red_mat->trace_len);
      la.reserve_values(batch_size);

      auto run_prime = [&](size_t slot, BS::thread_pool<BS::none>* row_pool) {
        prime_result& r = batch[slot];
        nmod_t m;
        nmod_init(&m, r.p);
        uint32_t* values = la.red_values[slot].data();

        // a pivot would be set to zero -- move on before allocating anything
        if(!fill_reducer_mat(la, state, entries_red, m, values)
           or !spol_pivots_survive(la, entries_spol, m, trace_snapshot.data()))
          return;

        r.rref = std::make_unique<sparse_mat_struct<uint32_t>>();
        set_up_spol_matrix(la,
                           state,
                           r.rref.get(),
                           m,
                           idxs_spol,
                           entries_spol,
                           trace_snapshot.data());
        r.piv = gauss_elim_dispatch(
          la, state, r.rref.get(), m, row_pool, values, trace_snapshot.data());
        r.usable = true;
      };

      {
        F4NCGB_TIME(rref);
        if(parallel) {
          for(size_t slot = 0; slot < batch_size; slot++)
            pool->detach_task(
              [&run_prime, slot]() { run_prime(slot, nullptr); });
          pool->wait();
        } else {
          // a batch of one gets the whole pool for its rows.
          run_prime(0, pool.get());
        }
      }

      for(auto& r : batch) {
        if(!r.usable) {
          if(verbose > 2)
            msg("Excluding prime %lu (bad pivots)", r.p);
          continue;
        }

        const int rank = cmp_pivots(best_piv, r.piv);

        if(rank > 0) {
          if(verbose > 2)
            msg("Excluding prime %lu (bad pivots)", r.p);
          sparse_mat_clear(r.rref.get());
          continue;
        }

        if(rank < 0) {
          for(const auto& stale : rrefs)
            sparse_mat_clear(stale.get());
          rrefs.clear();
          used_primes.clear();
          prod = 1;
        }

        best_piv = r.piv;
        rrefs.push_back(std::move(r.rref));
        used_primes.push_back(r.p);
        prod = prod * r.p;
      }
    }

    rref_ptrs.clear();
    for(const auto& rref : rrefs)
      rref_ptrs.push_back(rref.get());
    // Initialize crt_entries in reconstruction
    // and clear here
    fmpz* crt_entries = nullptr;
    idxs.clear();
    {
      F4NCGB_TIME(crt);
      crt_reconstruction(
        crt_entries, idxs, rref_ptrs, used_primes, best_piv.size());
    }

    bool success;
    {
      F4NCGB_PROFILE(auto timer = gstats.time(gstats.ratrec));
      success
        = rational_reconstruction(nums, denoms, crt_entries, prod, idxs.size());
    }

    fmpz_cleanup(crt_entries, idxs.size());

    if(success) {
      if(verify_result(nums, denoms, idxs.size(), h, state.nr_cols, prod))
        break;
      else if(verbose > 2)
        msg("Verification has failed. Increasing bound.");
    }

    target = used_primes.size() + std::max<size_t>(2, used_primes.size() / 2);
    target = std::min(target, MAX_PRIMES);
    if(target <= used_primes.size())
      die(14, "Multimodular Gaussian elimination is not converging");
    fmpz_cleanup(nums, idxs.size());
    fmpz_cleanup(denoms, idxs.size());
  }

  for(const auto& rref : rrefs)
    sparse_mat_clear(rref.get());
  clear_denoms(idxs, nums, denoms);
  fmpz_cleanup(denoms, idxs.size());

  return std::make_pair(idxs, nums);
}
//------------------------------------------------------------------------------

std::pair<std::vector<std::pair<size_t, size_t>>, fmpz*>
nmod_gauss_elim(la_context& la,
                const solve_state& state,
                const csr_indices& idxs_spol,
                std::vector<std::span<coeff>>& entries_spol,
                std::vector<std::span<coeff>>& entries_red,
                size_t p,
                std::unique_ptr<BS::thread_pool<BS::none>>& pool) {
  nmod_t mod;
  nmod_init(&mod, p);

  la.reserve_values(1);
  uint32_t* values = la.red_values[0].data();
  const unsigned char* trace = la.red_mat->trace;

  if(!fill_reducer_mat(la, state, entries_red, mod, values)
     or !spol_pivots_survive(la, entries_spol, mod, trace))
    die(13,
        "Leading coefficient of an input polynomial is divisible by the "
        "characteristic %lu.",
        p);

  uint32_mat_t nmod_mat;
  set_up_spol_matrix(la, state, nmod_mat, mod, idxs_spol, entries_spol, trace);

  F4NCGB_TIME(rref);
  pivots piv(
    gauss_elim_dispatch(la, state, nmod_mat, mod, pool.get(), values, trace));

  // compute nonzero indices & entries
  size_t nnz = sparse_mat_nnz(nmod_mat);
  std::vector<std::pair<size_t, size_t>> idxs;
  idxs.reserve(nnz);
  fmpz* entries = new fmpz[nnz];

  size_t k = 0;
  for(size_t i = 0; i < nmod_mat->nrow; i++) {
    auto row = sparse_mat_row(nmod_mat, i);
    for(size_t j = 0; j < row->nnz; j++) {
      idxs.emplace_back(i, row->indices[j]);
      fmpz_init_set_ui(entries + k++, row->entries[j]);
    }
  }

  sparse_mat_clear(nmod_mat);

  return std::make_pair(idxs, entries);
}
//------------------------------------------------------------------------------

std::pair<std::vector<std::pair<size_t, size_t>>, fmpz*>
linear_algebra(const solve_state& state,
               const csr_indices& idxs_spol,
               const csr_indices& idxs_red,
               const std::vector<size_t>& red_order,
               std::vector<std::span<coeff>>& entries_spol,
               std::vector<std::span<coeff>>& entries_red,
               size_t characteristic,
               std::unique_ptr<BS::thread_pool<BS::none>>& pool,
               bool tracer) {

  la_context la;
  la.use_trace = tracer;
  la.nrows_total = idxs_spol.rows() + idxs_red.rows();

  if(state.proof_level > 0)
    check_fits(state.nr_cols + la.nrows_total,
               UINT32_MAX,
               "width of the matrix with its transformation block");

  check_fits(idxs_red.rows(), INT32_MAX, "number of reducer rows");
  check_fits(idxs_spol.rows(), INT32_MAX, "number of s-polynomial rows");

  set_up_reducer_mat(la, state, idxs_red, red_order, entries_red);
  set_reducer_pivots(la);
  sparse_mat_init_trace(la.red_mat, idxs_spol.rows());

  std::pair<std::vector<std::pair<size_t, size_t>>, fmpz*> res;
  if(characteristic == 0)
    res = multimodular_gauss_elim(
      la, state, idxs_spol, entries_spol, entries_red, pool);

  else
    res = nmod_gauss_elim(
      la, state, idxs_spol, entries_spol, entries_red, characteristic, pool);

  sparse_mat_ro_clear(la.red_mat);

  return res;
}
}
