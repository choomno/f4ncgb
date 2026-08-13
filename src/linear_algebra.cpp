#include <algorithm>
#include <atomic>
#include <boost/unordered/unordered_flat_map.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "coeff.hpp"
#include "diagnostics.hpp"
#include "primes.hpp"
#include "sparse_rref/sparse_mat.h"
#include "store.hpp"

#include "linear_algebra.hpp"

extern int verbose;

namespace f4ncgb {

/// Scratch state of one call to linear_algebra(), shared by its helpers and by
/// the elimination tasks. One per call, so concurrent solves stay apart.
struct la_context {
  /// Reducer matrix layout and column indices, built once and reused for every
  /// prime. Its coefficients live in red_values, one array per prime that may
  /// be in flight at the same time.
  uint32_mat_ro_t red_mat = {};
  std::vector<std::vector<uint32_t>> red_values;
  /// Column -> reducer row that has its pivot there, -1 if there is none.
  std::vector<int32_t> red_pivots;
  /// Reducer rows share coefficient arrays. For each row, the first row using
  /// the same array -- the only one whose entries are actually stored.
  std::vector<uint32_t> ref_row;
  size_t nrows_total = 0;
  bool use_trace = true;

  /// Enough coefficient arrays for `slots` primes at once.
  void reserve_values(size_t slots) {
    if(red_values.size() < slots)
      red_values.resize(slots);
    for(auto& v : red_values)
      v.resize(red_mat->unique_nnz);
  }
};

//------------------------------------------------------------------------------

/// `order` puts the rows in pivot order: the CSR is still in the order symbolic
/// preprocessing discovered them, and row i here is idxs[order[i]]. Reading
/// through it costs one indirection per row and saves permuting every nonzero,
/// which for serre-e6-17 is a few hundred megabytes copied per iteration.
void inline set_up_reducer_mat(la_context& la,
                               const solve_state& state,
                               const csr_indices& idxs,
                               const std::vector<size_t>& order,
                               const std::vector<std::span<coeff>>& entries) {
  auto& red_mat = la.red_mat;
  auto& ref_row = la.ref_row;
  const size_t nrows = idxs.rows();
  size_t ncols = state.nr_cols;
  assert(order.size() == nrows);

  // first pass: compute total nnz and unique_nnz, and resolve every row to the
  // first one sharing its coefficients. Hashing the pointers once here saves a
  // lookup per row per prime later on.
  boost::unordered_flat_map<const coeff*, uint32_t> first_row;
  first_row.reserve(nrows);
  ref_row.resize(nrows);

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

  // idx_k below ends at nnz and entry_k at unique_nnz, so checking the totals
  // here covers every offset either of them writes.
  check_offset(nnz, "number of nonzeros in the reducer matrix");
  check_offset(unique_nnz, "number of stored reducer coefficients");
  check_fits(ncols, UINT32_MAX, "width of the reducer matrix");

  sparse_mat_ro_init(red_mat, nrows, ncols, nnz, unique_nnz);
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
    index_start[i] = static_cast<mat_offset_t>(idx_k);
    row_nnz[i] = row.size();

    // store indices
    for(const auto j : row)
      indices[idx_k++] = j;

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
    if(la.ref_row[i] != i)
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

/// A prime is unusable if it kills the leading coefficient of a row, because
/// that row loses its pivot.
///
/// This has to be decided before the matrix is built: bailing out of
/// set_up_spol_matrix() half-way would leave the remaining rows uninitialized,
/// while sparse_mat_clear() goes on to free every one of them.
bool inline spol_pivots_survive(const la_context& la,
                                const std::vector<std::span<coeff>>& entries,
                                nmod_t mod,
                                const unsigned char* trace) {
  (void)la;
  for(size_t i = 0; i < entries.size(); i++) {
    // rows known to reduce to zero are never filled
    if(trace[i] or entries[i].empty())
      continue;
    if(!fmpz_get_nmod(entries[i][0].value, mod))
      return false;
  }
  return true;
}

//------------------------------------------------------------------------------

/// Only call once spol_pivots_survive() has accepted this prime.
void
set_up_spol_matrix(const la_context& la,
                   const solve_state& state,
                   uint32_mat_t mat,
                   nmod_t mod,
                   const csr_indices& idxs,
                   const std::vector<std::span<coeff>>& entries,
                   const unsigned char* trace) {
  const size_t nrows = idxs.rows();

  // ---- initialize matrix ----
  if(state.proof_level > 0)
    sparse_mat_init(mat, nrows, state.nr_cols + la.nrows_total);
  else
    sparse_mat_init(mat, nrows, state.nr_cols);

  if(verbose > 2)
    msg("S-Pol matrix has size (%d, %d) with %d nonzeros",
        mat->nrow,
        mat->ncol,
        idxs.data.size());

  // ---- fill rows ----
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
    // --- insert reduced entries ---
    for(size_t k = 0; k < coeffs.size(); k++) {
      uint32_t c = fmpz_get_nmod(coeffs[k].value, mod);
      if(c != 0) {
        row->indices[nnz] = cols[k];
        row->entries[nnz] = c;
        nnz++;
      }
    }
    // the leading entry survives, spol_pivots_survive() made sure of it
    assert(coeffs.empty() or nnz > 0);

    // ---- transformation block ----
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

inline int
cmp_pivots(const pivots& x, const pivots& y) {
  if(x.size() < y.size())
    return -1;
  else if(x.size() > y.size())
    return 1;
  else if(x == y)
    return 0;
  else if(x < y)
    return 1;

  return -1;
}
//------------------------------------------------------------------------------

/// Largest coefficient in absolute value, which bounds how big the modulus has
/// to grow before the result can be reconstructed.
static inline coeff
height(const la_context& la,
       const std::vector<std::span<coeff>>& entries_spol,
       const std::vector<std::span<coeff>>& entries_red) {

  coeff h(0L);
  auto visit = [&h](const std::span<coeff>& row) {
    for(const auto& c : row) {
      if(fmpz_cmpabs(c.value, h.value) > 0)
        fmpz_abs(h.value, c.value);
    }
  };

  for(const auto& row : entries_spol)
    visit(row);

  // Reducer rows share coefficient arrays -- only 14-33 % of them are distinct
  // on these benchmarks -- and ref_row already says which row owns each one.
  for(size_t i = 0; i < entries_red.size(); i++)
    if(la.ref_row[i] == i)
      visit(entries_red[i]);

  return h;
}
//------------------------------------------------------------------------------

static void
crt_reconstruction(fmpz*& entries,
                   std::vector<std::pair<size_t, size_t>>& idxs,
                   std::vector<sparse_mat_struct<uint32_t>*>& rrefs,
                   std::vector<uint32_t>& primes,
                   size_t n_piv) {
  if(rrefs.size() == 0)
    return;

  // Columns nonzero in at least one of the rrefs, per row. Rows are dense in i,
  // so this is a vector of sorted vectors rather than a map of sets: the
  // entries are then read off with one cursor per rref instead of a binary
  // search each.
  std::vector<std::vector<uint32_t>> nnz_pos(n_piv);
  for(size_t i = 0; i < n_piv; i++) {
    auto& cols = nnz_pos[i];
    const auto first = sparse_mat_row(rrefs[0], i);

    // Two rrefs of the same matrix disagree about where the nonzeros are only
    // where a prime divides an entry, which is rare. Noticing that costs one
    // comparison per row per prime and saves sorting the same support as many
    // times as there are primes.
    bool same = true;
    for(size_t k = 1; k < rrefs.size() and same; k++) {
      const auto row = sparse_mat_row(rrefs[k], i);
      same
        = row->nnz == first->nnz
          and std::memcmp(
                row->indices, first->indices, first->nnz * sizeof(uint32_t))
                == 0;
    }
    if(same) {
      cols.assign(first->indices, first->indices + first->nnz);
      continue;
    }

    for(const auto& rref : rrefs) {
      auto row = sparse_mat_row(rref, i);
      // we have cleared all syzygy rows completely (also the transformation
      // matrix) thus, any row within n_piv must have nonzer polynomial part
      cols.insert(cols.end(), row->indices, row->indices + row->nnz);
    }
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
  }

  size_t nnz = 0;
  for(const auto& cols : nnz_pos)
    nnz += cols.size();
  idxs.reserve(nnz);

  size_t len = primes.size();
  fmpz* moduli = new fmpz[len];
  entries = new fmpz[nnz];
  for(size_t i = 0; i < len; i++)
    fmpz_init_set_ui(moduli + i, primes[i]);
  for(size_t i = 0; i < nnz; i++)
    fmpz_init(entries + i);

  fmpz_multi_CRT_t crt_base;
  fmpz_multi_CRT_init(crt_base);
  int res = fmpz_multi_CRT_precompute(crt_base, moduli, static_cast<long>(len));
  if(!res)
    die(12, "Problem with CRT");

  fmpz* inputs = new fmpz[rrefs.size()];
  for(size_t i = 0; i < rrefs.size(); i++)
    fmpz_init(inputs + i);

  // one cursor per rref, walked forward in lockstep with the column list
  std::vector<uint32_t> cursor(rrefs.size());

  size_t idx = 0;
  for(size_t i = 0; i < n_piv; i++) {
    std::fill(cursor.begin(), cursor.end(), 0);
    for(const uint32_t j : nnz_pos[i]) {
      idxs.emplace_back(i, j);
      for(size_t k = 0; k < rrefs.size(); k++) {
        auto row = sparse_mat_row(rrefs[k], i);
        uint32_t& at = cursor[k];
        while(at < row->nnz and row->indices[at] < j)
          at++;
        const bool hit = at < row->nnz and row->indices[at] == j;
        fmpz_set_ui(inputs + k, hit ? row->entries[at] : 0);
      }
      fmpz_multi_CRT_precomp(entries + idx++, crt_base, inputs, 0);
    }
  }

  fmpz_multi_CRT_clear(crt_base);
  fmpz_cleanup(moduli, primes.size());
  fmpz_cleanup(inputs, rrefs.size());
}
//------------------------------------------------------------------------------

inline bool
verify_result(fmpz* nums,
              fmpz* denoms,
              size_t len,
              coeff& height,
              size_t n,
              coeff& P) {

  fmpz_t d;
  fmpz_init_set_ui(d, 1);
  for(size_t i = 0; i < len; i++)
    fmpz_lcm(d, d, denoms + i);

  // height_rref = max | (d / denom_i) * num_i |
  fmpz_t tmp;
  fmpz_t height_rref;
  fmpz_init(tmp);
  fmpz_init_set_ui(height_rref, 0);

  for(size_t i = 0; i < len; i++) {
    fmpz_divexact(tmp, d, denoms + i);
    fmpz_mul(tmp, tmp, nums + i);
    fmpz_abs(tmp, tmp);
    if(fmpz_cmp(tmp, height_rref) > 0)
      fmpz_set(height_rref, tmp);
  }

  // Now compute lhs = height_rref * height * n
  fmpz_t lhs_fmpz;
  fmpz_init(lhs_fmpz);

  fmpz_mul(lhs_fmpz, height_rref, height.value);
  fmpz_mul_ui(lhs_fmpz, lhs_fmpz, n);

  bool result = (fmpz_cmp(lhs_fmpz, P.value) < 0);

  // cleanup
  fmpz_clear(d);
  fmpz_clear(tmp);
  fmpz_clear(height_rref);
  fmpz_clear(lhs_fmpz);

  return result;
}
//------------------------------------------------------------------------------

/// Recovers each entry as a fraction with numerator and denominator below
/// sqrt(prod/2), the bound that makes the result unique. Fails as soon as one
/// entry has no such representation, which means more primes are needed.
static inline bool
rational_reconstruction(fmpz*& nums,
                        fmpz*& denoms,
                        fmpz* crt_entries,
                        coeff& prod,
                        size_t N) {

  nums = new fmpz[N];
  denoms = new fmpz[N];
  for(size_t i = 0; i < N; i++) {
    fmpz_init(nums + i);
    fmpz_init(denoms + i);
  }

  bool res = true;
  fmpq_t q;
  fmpq_init(q);
  for(size_t i = 0; i < N; i++) {
    if(!fmpq_reconstruct_fmpz(q, crt_entries + i, prod.value)) {
      if(verbose > 2)
        msg("Rational reconstruction does not exist");
      res = false;
      break;
    }
    fmpz_set(nums + i, fmpq_numref(q));
    fmpz_set(denoms + i, fmpq_denref(q));
  }
  fmpq_clear(q);
  return res;
}
//------------------------------------------------------------------------------

void
clear_denoms(std::vector<std::pair<size_t, size_t>>& idxs,
             fmpz* nums,
             fmpz* denoms) {

  if(idxs.empty())
    return;

  fmpz_t L;
  fmpz_t tmp;
  fmpz_init(L);
  fmpz_init(tmp);

  size_t row_start = 0;

  while(row_start < idxs.size()) {

    const size_t cur_row = idxs[row_start].first;
    fmpz_set_ui(L, 1);

    // find row range and compute denom-lcm
    size_t row_end = row_start;
    while(row_end < idxs.size() && idxs[row_end].first == cur_row) {
      fmpz_lcm(L, L, denoms + row_end);
      row_end++;
    }

    // scale nums if necessary
    if(!fmpz_is_one(L)) {
      for(size_t i = row_start; i < row_end; i++) {
        fmpz_divexact(tmp, L, denoms + i);
        fmpz_mul(nums + i, nums + i, tmp);
      }
    }

    row_start = row_end;
  }

  fmpz_clear(L);
  fmpz_clear(tmp);
}
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------

/// Reduces v modulo the Mersenne prime 2^31-1 without dividing: algorithm 4 of
/// https://arxiv.org/pdf/2008.08654, with b = 31. Valid for v below 2^62, which
/// the elimination asserts before calling.
[[nodiscard]] static constexpr inline int64_t
mersenne_mod(int64_t v) noexcept {
  const int64_t v_prime = v + 1;
  const int64_t z = ((v_prime >> 31) + v_prime) >> 31;
  return (v + z) & 2'147'483'647;// 2^31-1
}

/// Reduction modulo a fixed p by a precomputed inverse, replacing the hardware
/// division the accumulator scan used to do per visited column.
///
/// With m = floor(2^64 / p), the quotient estimate is off by at most one for
/// any v below 2^62 -- the elimination's own bound on the accumulator -- so a
/// single conditional subtraction finishes it.
///
/// Reducing on every write instead, so that the accumulator is uint32_t rather
/// than int64_t, was tried and is slower. It halves the one array the scan
/// reads at random -- 13 MB on braid3-16, one per thread -- and removes this
/// reduction from the scan entirely, but it costs a Shoup multiplication per
/// nonzero in xmay() where lazy accumulation costs a multiply and a
/// conditional add. Measured against this version: braidXY-12 15.14 s against
/// 14.26 s and braid3-16 54.44 s against 52.71 s at -T 1, and at -T 8 the
/// elimination 2.50 s against 2.13 s and 10.03 s against 10.28 s. So the extra
/// arithmetic outweighs the traffic saved even where the accumulator is far
/// past L2, and the one figure that came out ahead is inside the run-to-run
/// spread these benchmarks have (about 3 % on braid3-16).
struct barrett {
  uint64_t p;
  uint64_t m;

  explicit barrett(uint64_t p_) noexcept
    : p(p_)
    // p is an odd prime, so it never divides 2^64 and this is floor(2^64 / p)
    , m(~uint64_t(0) / p_) {}

  [[nodiscard]] inline uint64_t operator()(uint64_t v) const noexcept {
    const uint64_t q
      = static_cast<uint64_t>((static_cast<__uint128_t>(v) * m) >> 64);
    const uint64_t r = v - q * p;
    return r >= p ? r - p : r;
  }
};
//------------------------------------------------------------------------------

/// Finding the next nonzero of the dense accumulator means walking it column by
/// column, 8 bytes at a time, over ground the row itself never touches. So
/// every write also sets a bit in a mask, one bit per column, and a word of
/// that mask reading zero lets the walk jump over 64 columns at once.
///
/// The mask is not free: every accumulator update sets its bit, so a row pays
/// in proportion to the elimination work and is repaid in proportion to the
/// columns skipped. That is overwhelmingly worth it where the accumulator is
/// large and stays sparse -- lascala_neuh-13 (682903 columns) and
/// malle_G12h-100 run 4-5x faster, 4nilp5s-10 twice as fast. It is not
/// universal: braidXY-12 does enough arithmetic per row that the marks cost
/// more than the scan saves, and pays about 12% for it. Attempts to pick per
/// row from measured density did not separate the two cases, so the mask is
/// unconditional.
///
/// Marks are conservative: an entry can become zero again after a subtraction,
/// so a mark means "maybe nonzero". The invariant that matters is the other
/// direction -- a nonzero column is always marked -- plus leaving both mask and
/// accumulator clean for the next row.
///
/// A second level over this -- one bit per mask word, skipping 4096 columns at
/// a time -- was tried, on the grounds that the mask is 202 KB on braid3-16
/// (1618887 columns) and 578 KB on 4nilp5s-10 (4622141), well past L1. It lost
/// everywhere: 10% on braid3-16, 6% on 4nilp5s-10, and 10% on braidXY-12, which
/// at 79475 columns has a 10 KB mask and was measured with the second level
/// switched off altogether -- so that last figure is the bookkeeping alone.
///
/// Two reasons. The walk the summary shortens is a sequential read that the
/// hardware prefetcher already covers, and the upkeep (a store per mark, a
/// branch per unmark, a test in the innermost step) lands on the hottest path
/// there is -- including in the batched write below, which it breaks up.
static inline void
mark(uint64_t* mask, uint32_t i) {
  mask[i >> 6] |= uint64_t(1) << (i & 63);
}

static inline void
unmark(uint64_t* mask, uint32_t i) {
  mask[i >> 6] &= ~(uint64_t(1) << (i & 63));
}

/// First marked column at or after `i`, or `limit + 1` if there is none.
static inline uint32_t
next_marked(const uint64_t* mask, uint32_t i, uint32_t limit) {
  uint32_t w = i >> 6;
  const uint32_t last_w = limit >> 6;
  uint64_t bits = mask[w] & (~uint64_t(0) << (i & 63));

  while(!bits) {
    if(++w > last_w)
      return limit + 1;
    bits = mask[w];
  }

  uint32_t found = (w << 6) + static_cast<uint32_t>(__builtin_ctzll(bits));
  return found > limit ? limit + 1 : found;
}
//------------------------------------------------------------------------------

template<typename T>
static void inline copy_to_buffer(T& buffer, uint64_t* mask, uint32_vec_t vec) {
  for(size_t i = 0; i < vec->nnz; i++) {
    buffer[vec->indices[i]] = vec->entries[i];
    mark(mask, vec->indices[i]);
  }
}
//------------------------------------------------------------------------------

static void inline copy_from_buffer_and_clear(int64_t* buffer,
                                              uint64_t* mask,
                                              std::vector<uint32_t>& buffer_ids,
                                              uint32_vec_t vec) {
  size_t nnz = buffer_ids.size();

  sparse_vec_realloc(vec, nnz);
  vec->nnz = nnz;

  std::move(buffer_ids.begin(), buffer_ids.end(), vec->indices);
  size_t j = 0;
  for(size_t i : buffer_ids) {
    assert(buffer[i] != 0);
    vec->entries[j++] = static_cast<uint32_t>(buffer[i]);
    buffer[i] = 0;
    unmark(mask, static_cast<uint32_t>(i));
  }
}
//------------------------------------------------------------------------------

static void inline normalize_row(uint32_vec_t vec, nmod_t mod) {
  uint32_t c = vec->entries[0];
  uint32_t inv = nmod_inv(c, mod);
  for(size_t i = 0; i < vec->nnz; i++) {
    uint32_t v = nmod_mul(inv, vec->entries[i], mod);
    vec->entries[i] = v;
  }
}
//------------------------------------------------------------------------------

// Compute x - ay mod p
// but leave out the 0th entry of y
// because that will be zero anyway
//
// Prefetching the pivot entry for each column written here was tried -- per
// column the scan chains mask read -> accumulator read -> pivot read -> reducer
// row read, and this would break one link of it. It cost 9% on braid3-16 and 7%
// on 4nilp5s-10: the scan goes on to visit only the smallest of the columns a
// row touches, so most of the prefetched lines were evicted before anything
// asked for them.
static inline void
xmay(int64_t* __restrict x,
     uint64_t* __restrict mask,
     int64_t a,
     const uint32_t* __restrict idx,
     const uint32_t* __restrict val,
     uint32_t nnz,
     int64_t p2) {
  uint32_t i = 1;
  for(; i + 3 < nnz; i += 4) {
    uint32_t j0 = idx[i];
    uint32_t j1 = idx[i + 1];
    uint32_t j2 = idx[i + 2];
    uint32_t j3 = idx[i + 3];

    int64_t v0 = x[j0];
    int64_t v1 = x[j1];
    int64_t v2 = x[j2];
    int64_t v3 = x[j3];

    int64_t t0 = v0 - a * val[i];
    int64_t t1 = v1 - a * val[i + 1];
    int64_t t2 = v2 - a * val[i + 2];
    int64_t t3 = v3 - a * val[i + 3];

    t0 += (t0 >> 63) & p2;
    t1 += (t1 >> 63) & p2;
    t2 += (t2 >> 63) & p2;
    t3 += (t3 >> 63) & p2;

    x[j0] = t0;
    x[j1] = t1;
    x[j2] = t2;
    x[j3] = t3;

    // one read-modify-write instead of four dependent ones, whenever the four
    // columns land in the same mask word
    if((j0 >> 6) == (j3 >> 6)) {
      mask[j0 >> 6] |= (uint64_t(1) << (j0 & 63)) | (uint64_t(1) << (j1 & 63))
                       | (uint64_t(1) << (j2 & 63))
                       | (uint64_t(1) << (j3 & 63));
    } else {
      mark(mask, j0);
      mark(mask, j1);
      mark(mask, j2);
      mark(mask, j3);
    }
  }

  // Handle remaining elements
  for(; i < nnz; i++) {
    uint32_t j = idx[i];
    int64_t t = x[j] - a * val[i];
    t += (t >> 63) & p2;
    x[j] = t;
    mark(mask, j);
  }
}
//------------------------------------------------------------------------------

template<bool mersenne = false>
static pivots
reverse_solve(uint32_mat_t mat, nmod_t mod) {
  uint64_t p = mod.n;
  int64_t p2 = static_cast<int64_t>(p * p);

  const barrett red(p);

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
    // the leading entry is already normalized, the scan starts after it
    buffer_ids.push_back(row->indices[0]);
    int64_t cc;
    int32_t rr;
    uint32_t i = row->indices[1];
    uint32_t max_col = row->indices[row->nnz - 1];
    while(true) {
      i = next_marked(mask, i, max_col);
      if(i > max_col)
        break;

      // the mark may be stale
      if(buff[i] == 0) {
        unmark(mask, i);
        i++;
        continue;
      }

      if constexpr(mersenne)
        cc = mersenne_mod(buff[i]);
      else
        cc = static_cast<int64_t>(red(static_cast<uint64_t>(buff[i])));
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
        xmay(buff, mask, cc, row_rr->indices, row_rr->entries, row_rr->nnz, p2);
      }
      i++;
    }
    copy_from_buffer_and_clear(buff, mask, buffer_ids, row);
  }

  return piv;
}

//------------------------------------------------------------------------------

/// `red_values` holds the reducer coefficients for this prime and `trace` the
/// rows already known to vanish, both supplied by the caller: several primes can
/// be in flight at once, so neither may be read out of shared state.
/// Picks the Mersenne specialisation of the elimination when the prime allows.
static pivots
gauss_elim_dispatch(la_context& la,
                    const solve_state& state,
                    uint32_mat_t mat,
                    nmod_t mod,
                    BS::thread_pool<BS::none>* pool,
                    const uint32_t* red_values,
                    const unsigned char* trace);

template<bool mersenne = false>
pivots
gauss_elim(la_context& la,
           const solve_state& state,
           uint32_mat_t mat,
           nmod_t mod,
           BS::thread_pool<BS::none>* pool,
           const uint32_t* red_values,
           const unsigned char* trace) {

  uint64_t p = mod.n;
  int64_t p2 = static_cast<int64_t>(p * p);
  const barrett red(p);

  thread_local std::vector<int64_t> buffer_local;
  thread_local std::vector<uint64_t> buffer_mask_local;
  thread_local std::vector<uint32_t> buffer_ids_local;

  std::vector<std::atomic_int32_t> atomic_pivots(mat->ncol);
  // Nothing else can see this yet -- the tasks below are detached afterwards --
  // so plain stores do. std::fill would emit a sequentially consistent one per
  // column, of which there are 1.6 M on the widest matrices.
  for(auto& pivot : atomic_pivots)
    pivot.store(-1, std::memory_order_relaxed);

  // reduce row with all already known pivots
  auto reduce_row
    = [&la, &state, &mat, &atomic_pivots, p2, red, &mod, red_values, trace](
        size_t r) {
      if(trace[r])
        return;
      F4NCGB_TIME(elim_task_cpu);
      auto row = sparse_mat_row(mat, r);
      int32_t rr;
      int64_t cc;
      int32_t expected = -1;
      uint32_t* indices;
      const uint32_t* entries;
      uint32_t nnz;

      buffer_local.resize(mat->ncol, 0);
      buffer_mask_local.resize(mat->ncol / 64 + 1, 0);
      int64_t* buff = buffer_local.data();
      uint64_t* mask = buffer_mask_local.data();

      do {
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
          if constexpr(mersenne) {
            (void)red;// the precomputed inverse is not used in this case.
            cc = mersenne_mod(buff[i]);
          } else
            cc = static_cast<int64_t>(red(static_cast<uint64_t>(buff[i])));
          buff[i] = cc;
          if(cc == 0) {
            unmark(mask, i);
            i++;
            continue;
          }

          // first check red pivots
          rr = la.red_pivots[i];
          if(rr >= 0) {
            buff[i] = 0;
            unmark(mask, i);
            indices = red_mat_indices(la.red_mat, rr);
            entries = red_mat_entries(la.red_mat, red_values, rr);
            nnz = red_mat_nnz(la.red_mat, rr);
            max_col = std::max(max_col, indices[nnz - 1]);
            xmay(buff, mask, cc, indices, entries, nnz, p2);
            i++;
            continue;
          }

          // then check atomic pivots. Acquire, to pair with the release on the
          // compare_exchange below: seeing the pivot has to mean seeing the row
          // the thread that claimed it wrote first.
          rr = atomic_pivots[i].load(std::memory_order_acquire);
          if(rr >= 0) {
            buff[i] = 0;
            unmark(mask, i);
            auto row = sparse_mat_row(mat, rr);
            indices = row->indices;
            entries = row->entries;
            nnz = row->nnz;
            max_col = std::max(max_col, indices[nnz - 1]);
            xmay(buff, mask, cc, indices, entries, nnz, p2);
            i++;
            continue;
          }

          // no reducer found
          buffer_ids_local.push_back(i);
          i++;
        }

        // we have a zero row
        if(buffer_ids_local.empty()
           or (state.proof_level > 0
               and buffer_ids_local[0] >= mat->ncol - la.nrows_total)) {
          sparse_vec_clear(row);
          for(const auto id : buffer_ids_local) {
            buff[id] = 0;
            unmark(mask, id);
          }
          // Only set if we use the tracer. Relaxed: concurrent primes all
          // store the same value here, and nobody reads it before the batch
          // they belong to has finished.
          if(la.use_trace)
            __atomic_store_n(&la.red_mat->trace[r], 1, __ATOMIC_RELAXED);
          return;
        }
        copy_from_buffer_and_clear(buff, mask, buffer_ids_local, row);
        normalize_row(row, mod);
        expected = -1;
        // Strong, not weak: a failure here sends the row through the whole
        // reduction again, which is the right answer when another thread really
        // did claim the column and pure waste when the LL/SC just lost its
        // reservation.
      } while(!atomic_pivots[row->indices[0]].compare_exchange_strong(
        expected,
        static_cast<int32_t>(r),
        std::memory_order_release,
        std::memory_order_relaxed));
    };

  if(pool) {
    // Rows go out in blocks, not one task apiece. A row is often far too small
    // to be worth a trip through the queue: lv2d10-100 hands out up to 85928 of
    // them per prime, each carrying four nonzeros, and paid 44% for it at -T 8
    // against -T 1. There are still dozens of blocks per thread, so the queue
    // keeps balancing rows whose cost varies wildly, and they are handed out in
    // row order exactly as before -- a row that waits for pivots the rows above
    // it claim still tends to run after them.
    const size_t rows = mat->nrow;
    const size_t block
      = std::max<size_t>(1, rows / (pool->get_thread_count() * 32));
    for(size_t begin = 0; begin < rows; begin += block) {
      const size_t end = std::min(begin + block, rows);
      pool->detach_task([begin, end, &reduce_row]() {
        for(size_t r = begin; r < end; r++)
          reduce_row(r);
      });
    }
    pool->wait();
  } else {
    for(size_t r = 0; r < mat->nrow; r++)
      reduce_row(r);
  }

  if(mod.n == PRIMES[0]) {
    return reverse_solve<true>(mat, mod);
  } else {
    return reverse_solve<false>(mat, mod);
  }
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

/// One prime's contribution to the modulus. Filled in by a batch task and read
/// back in prime order, so which task finished first cannot change the result.
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

  // Every rref held here was computed with the best pivot set seen so far; the
  // moment a prime reveals a better one the rest are thrown away, see below.
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

  /// Primes to gather before trying to reconstruct again. Zero while the
  /// initial bound M is in charge; from the first failure on, this drives the
  /// loop and grows geometrically.
  ///
  /// It used to grow by `M = prod * p * p`, which is exactly two more primes per
  /// failure. Every round then re-ran the CRT over every prime gathered so far,
  /// at a cost superlinear in their number, so braidXY-12 spent 51 rounds and
  /// 2908 prime-passes producing a result that needs 300 -- 37% of its runtime.
  /// Growing by half means a handful of rounds, and the last one dominates.
  size_t target = 0;

  const size_t threads = pool ? pool->get_thread_count() : 1;
  std::vector<prime_result> batch;
  std::vector<unsigned char> trace_snapshot;

  while(true) {
    while(prod < M or used_primes.size() < target) {
      // How many primes are still wanted. Primes are all just under 2^31, so
      // the shortfall in the bound converts straight into a count.
      size_t wanted = used_primes.size() < target ? target - used_primes.size()
                                                  : 0;
      if(prod < M) {
        const flint_bitcnt_t missing
          = fmpz_bits(M.value) - std::min(fmpz_bits(M.value),
                                          fmpz_bits(prod.value));
        wanted = std::max<size_t>(wanted, (missing + 30) / 31);
      }
      wanted = std::max<size_t>(wanted, 1);
      wanted = std::min(wanted, MAX_PRIMES - std::min(i, MAX_PRIMES));
      if(wanted == 0)
        die(-1, "Multimodular Gaussian elimination is not converging");

      // Reducing several primes at once beats splitting one prime's rows over
      // the same threads only when there are enough of them to fill the pool.
      // With two primes and eight threads a batch would leave six idle, while
      // row-level parallelism uses all of them -- and worse, batching those two
      // would forfeit the tracer, which is what makes every prime after the
      // first several times cheaper than it. So the test is on what this call
      // has shown it needs in total, not on what is left: the tail of a long
      // run still belongs in a batch.
      //
      // The first prime always runs alone whatever the count. It is the one
      // that fills in the tracer, and every prime batched beside it would do
      // the work it is about to make unnecessary.
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

      // One trace for the whole batch. Taking a copy is what makes it safe to
      // run several primes at once: a prime that discovers a new zero row
      // publishes it for the next batch, but nobody may see the trace change
      // between building its matrix and reducing it -- that would leave a row
      // filled in and never reduced.
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
           or !spol_pivots_survive(
             la, entries_spol, m, trace_snapshot.data()))
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
            pool->detach_task([&run_prime, slot]() {
              run_prime(slot, nullptr);
            });
          pool->wait();
        } else {
          // A batch of one gets the whole pool for its rows.
          run_prime(0, pool.get());
        }
      }

      // Consumed in prime order, never in completion order, so the batch
      // reaches the same verdict the primes would have reached one at a time.
      // The elimination does not depend on best_piv, so a prime computed before
      // a better pivot set turned up is still the rref it would have been.
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
          // This prime found more pivots than every earlier one, so those were
          // computed at a rank the true result does not have. Their primes have
          // to leave the modulus with them: counting them would let the loop
          // below stop gathering while there is still too little to reconstruct
          // from.
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

    // Half as many again, never past the primes there are: overshooting into
    // die() would turn a run that converges into one that does not.
    target = used_primes.size() + std::max<size_t>(2, used_primes.size() / 2);
    target = std::min(target, MAX_PRIMES);
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

  // Input coefficients are not reduced modulo the characteristic when they are
  // parsed, so a leading coefficient divisible by p can reach us here. Unlike
  // in the multimodular case there is no other prime to fall back on.
  if(!fill_reducer_mat(la, state, entries_red, mod, values)
     or !spol_pivots_survive(la, entries_spol, mod, trace))
    die(13,
        "Leading coefficient of an input polynomial is divisible by the "
        "characteristic %lu.",
        p);

  uint32_mat_t nmod_mat;
  set_up_spol_matrix(
    la, state, nmod_mat, mod, idxs_spol, entries_spol, trace);

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

  // Both matrices carry a column per row when proofs are on -- the
  // transformation block -- and those columns are indexed as nr_cols + row, in
  // uint32_t like every other column index.
  if(state.proof_level > 0)
    check_fits(state.nr_cols + la.nrows_total,
               UINT32_MAX,
               "width of the matrix with its transformation block");

  // red_pivots and atomic_pivots hold a row per column as an int32_t, with -1
  // for "no pivot here".
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
