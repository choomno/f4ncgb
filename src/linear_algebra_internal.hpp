#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#if defined(__AVX512F__) or defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <flint/nmod.h>

#include "linear_algebra.hpp"
#include "solve_state.hpp"
#include "sparse_rref/sparse_mat.h"
#include "sparse_rref/sparse_vec.h"

// Internals of linear_algebra.cpp: the per-call scratch state, the modular
// arithmetic and the dense accumulation buffer that the row-at-a-time
// elimination and the blocked one share, and then the blocked pass itself.
//
// Blocked reduction of the s-polynomial rows against the reducer rows.
//
// The row-at-a-time elimination in linear_algebra.cpp keeps one dense
// accumulator of the full matrix width, so every element of every applied
// reducer is a scattered read-modify-write into tens of megabytes, and each row
// re-streams the reducer data it needs. This does the same arithmetic, but
// splits the columns into blocks and only ever accumulates into one block at a
// time, and reduces `lanes` rows together so a reducer's data is read once for
// all of them.

namespace f4ncgb {

//------------------------------------------------------------------------------

// Scratch state of one call to linear_algebra(),
// shared by helpers and by the elimination tasks.
struct la_context {
  uint32_mat_ro_t red_mat = {};
  std::vector<std::vector<uint32_t>> red_values;
  std::vector<int32_t> red_pivots;
  // One bit per reducer row, set when the row is the first to use its
  // coefficient span and so the one whose entries are actually stored.
  std::vector<uint64_t> rep_mask;
  size_t nrows_total = 0;
  bool use_trace = true;

  bool represents(size_t i) const { return (rep_mask[i >> 6] >> (i & 63)) & 1; }

  void set_represents(size_t i) { rep_mask[i >> 6] |= uint64_t(1) << (i & 63); }

  // One buffer of reducer coefficients per prime running at once. Slots that
  // are no longer wanted are dropped rather than kept at the current width.
  void reserve_values(size_t slots) {
    red_values.resize(slots);
    for(auto& v : red_values)
      v.resize(red_mat->unique_nnz);
  }
};

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
//------------------------------------------------------------------------------

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

template<bool mersenne>
struct modulus {
  barrett red{ 2 };
  uint64_t p = 2;
  int64_t p2 = 4;

  modulus() = default;
  explicit modulus(nmod_t mod) { set(mod); }

  void set(nmod_t mod) {
    red = barrett(mod.n);
    p = mod.n;
    p2 = static_cast<int64_t>(mod.n * mod.n);
  }

  [[nodiscard]] inline int64_t reduce(int64_t v) const {
    if constexpr(mersenne)
      return mersenne_mod(v);
    else
      return static_cast<int64_t>(red(static_cast<uint64_t>(v)));
  }

  [[nodiscard]] inline uint32_t negate(uint32_t a) const {
    return static_cast<uint32_t>(p) - a;
  }
};
//------------------------------------------------------------------------------

static inline void
mark(uint64_t* mask, uint32_t i) {
  mask[i >> 6] |= uint64_t(1) << (i & 63);
}

static inline void
unmark(uint64_t* mask, uint32_t i) {
  mask[i >> 6] &= ~(uint64_t(1) << (i & 63));
}

// First marked column at or after `i`, or `limit + 1` if there is none.
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

  // handle remaining elements
  for(; i < nnz; i++) {
    uint32_t j = idx[i];
    int64_t t = x[j] - a * val[i];
    t += (t >> 63) & p2;
    x[j] = t;
    mark(mask, j);
  }
}
//------------------------------------------------------------------------------

// Rows reduced together
constexpr uint32_t blocked_lanes = 32;

// The accumulator a block works in
constexpr size_t block_budget = size_t(1) << 20;

// The block width filling that budget, a power of two so that the block a
// column belongs to is a shift
constexpr uint32_t blocked_block_cols
  = block_budget / (blocked_lanes * sizeof(int64_t));
static_assert(std::has_single_bit(blocked_block_cols));
constexpr uint32_t blocked_log_cols
  = static_cast<uint32_t>(std::bit_width(blocked_block_cols) - 1);

//------------------------------------------------------------------------------

namespace detail {

// x[l] -= a[l] * v for `width` lanes, keeping every lane in [0, p2). Both
// factors are below 2^31, so this is a widening 32x32 -> 64 multiply, which is
// the one integer multiply every SIMD extension has. The conditional add
// replaces the branch that would otherwise bring the lanes back in range.
#if defined(__AVX512F__)
#define F4NCGB_HAVE_AXMY_SIMD 1
static constexpr uint32_t simd_width = 8;
inline void
axmy_simd(int64_t* x, const uint32_t* a, int64_t v, int64_t p2) {
  const __m512i av = _mm512_cvtepu32_epi64(
    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
  __m512i t = _mm512_sub_epi64(_mm512_loadu_si512(x),
                               _mm512_mul_epu32(av, _mm512_set1_epi64(v)));
  t = _mm512_mask_add_epi64(t,
                            _mm512_cmplt_epi64_mask(t, _mm512_setzero_si512()),
                            t,
                            _mm512_set1_epi64(p2));
  _mm512_storeu_si512(x, t);
}
#elif defined(__AVX2__)
#define F4NCGB_HAVE_AXMY_SIMD 1
static constexpr uint32_t simd_width = 4;
inline void
axmy_simd(int64_t* x, const uint32_t* a, int64_t v, int64_t p2) {
  // _mm256_mul_epu32 reads the even 32-bit halves, which is where values below
  // 2^31 already sit once they are widened.
  const __m256i av = _mm256_cvtepu32_epi64(
    _mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
  __m256i t
    = _mm256_sub_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)),
                       _mm256_mul_epu32(av, _mm256_set1_epi64x(v)));
  // AVX2 has no arithmetic 64-bit shift, so the sign mask comes from a compare
  const __m256i neg = _mm256_cmpgt_epi64(_mm256_setzero_si256(), t);
  t = _mm256_add_epi64(t, _mm256_and_si256(neg, _mm256_set1_epi64x(p2)));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(x), t);
}
#elif defined(__ARM_NEON)
#define F4NCGB_HAVE_AXMY_SIMD 1
static constexpr uint32_t simd_width = 2;
inline void
axmy_simd(int64_t* x, const uint32_t* a, int64_t v, int64_t p2) {
  const uint64x2_t prod
    = vmull_u32(vld1_u32(a), vdup_n_u32(static_cast<uint32_t>(v)));
  int64x2_t t = vsubq_s64(vld1q_s64(x), vreinterpretq_s64_u64(prod));
  t = vaddq_s64(t, vandq_s64(vshrq_n_s64(t, 63), vdupq_n_s64(p2)));
  vst1q_s64(x, t);
}
#else
#define F4NCGB_HAVE_AXMY_SIMD 0
#endif

}

// x[l] -= a[l] * v over all B lanes.
template<uint32_t B>
inline void
axmy_lanes(int64_t* x, const uint32_t* a, int64_t v, int64_t p2) {
  uint32_t l = 0;
#if F4NCGB_HAVE_AXMY_SIMD
  for(; l + detail::simd_width <= B; l += detail::simd_width)
    detail::axmy_simd(x + l, a + l, v, p2);
#endif
  for(; l < B; l++) {
    int64_t t = x[l] - static_cast<int64_t>(a[l]) * v;
    t += (t >> 63) & p2;
    x[l] = t;
  }
}

//------------------------------------------------------------------------------

// The modular arithmetic the lanes share. Everything the pass below does with a
// modulus goes through here, which is the seam a prime-batched variant would
// replace: batching the lanes over primes instead of over rows needs `reduce`
// and `axmy` to take their modulus and their p^2 per lane rather than once,
// and nothing else in the sweep, the buckets or the reduction matrix changes.
template<bool mersenne>
struct shared_modulus : modulus<mersenne> {
  // The accumulator entry of one lane, brought back below p.
  [[nodiscard]] inline uint32_t reduce(int64_t v, uint32_t) const {
    return static_cast<uint32_t>(modulus<mersenne>::reduce(v));
  }

  // x[l] -= a[l] * v over all B lanes, each staying in [0, p^2)
  template<uint32_t B>
  inline void axmy(int64_t* x, const uint32_t* a, int64_t v) const {
    axmy_lanes<B>(x, a, v, this->p2);
  }
};

// A reducer row waiting for the column block it next has data in.
struct block_carry {
  uint32_t row;
  uint32_t cursor;
  uint32_t slot;
};

// Per-thread scratch plus the invariants of one matrix; see run_batch().
template<bool mersenne, uint32_t B, typename Mod = shared_modulus<mersenne>>
struct blocked_reducer {
  static_assert(B >= 1 and B <= 32, "lane count must fit the multiplier array");

  const la_context* la = nullptr;
  sparse_mat_struct<uint32_t>* mat = nullptr;
  const uint32_t* red_values = nullptr;
  Mod arith;
  uint32_t mon_cols
    = 0;// columns holding monomials; the rest is the proof block
  uint32_t trans_extra = 0;// trailing entries of a reducer row outside mon_cols
  uint32_t nblocks = 0;
  bool use_trace = true;

  std::vector<int64_t> acc;
  std::vector<uint64_t> mask;
  std::vector<uint32_t> mult;    // multiplier vectors, slot * B + lane
  std::vector<uint32_t> slot_red;// reducer row behind each slot
  // Slots of reducers that have run out of data. Without a proof block nothing
  // reads them again, so reusing them holds the arena down to the reducers
  // that are live at once rather than every one the batch has touched.
  std::vector<uint32_t> free_slots;
  std::vector<std::vector<block_carry>> pending;
  std::vector<uint32_t> out_cols[B];
  std::vector<uint32_t> out_vals[B];
  std::vector<std::pair<uint32_t, uint32_t>> trans;

  // Input rows of the current batch, with the proof-block tail split off.
  const uint32_t* in_idx[B];
  const uint32_t* in_val[B];
  uint32_t in_nnz[B];
  uint32_t in_cur[B];
  uint32_t tail_col[B];
  uint32_t tail_val[B];

  void configure(const la_context& la_,
                 const solve_state& state,
                 sparse_mat_struct<uint32_t>* mat_,
                 nmod_t mod_,
                 const uint32_t* red_values_) {
    la = &la_;
    mat = mat_;
    red_values = red_values_;
    arith.set(mod_);
    mon_cols = static_cast<uint32_t>(state.nr_cols);
    trans_extra = state.proof_level > 0 ? 1 : 0;
    use_trace = la_.use_trace;

    nblocks = mon_cols == 0 ? 0 : ((mon_cols - 1) >> blocked_log_cols) + 1;

    // The buffer is all zeros between batches, so growing it only has to zero
    // what was not there before.
    const size_t width
      = std::min<size_t>(blocked_block_cols, std::max<uint32_t>(1, mon_cols));
    if(acc.size() < width * B)
      acc.resize(width * B, 0);
    if(mask.size() < width / 64 + 1)
      mask.resize(width / 64 + 1, 0);
    // Buckets are empty between batches, so keeping the ones a wider matrix
    // needed costs nothing and saves reallocating them.
    if(pending.size() < nblocks)
      pending.resize(nblocks);
  }

  // acc[c] -= a * val for every lane, over the part of reducer row `row` that
  // falls into [c0, c1). Returns the cursor just past it.
  inline uint32_t apply_slice(uint32_t row,
                              uint32_t k,
                              uint32_t c0,
                              uint32_t c1,
                              const uint32_t* a) {
    const uint32_t* idx = red_mat_indices(la->red_mat, row);
    const uint32_t* val = red_mat_entries(la->red_mat, red_values, row);
    const uint32_t nnz = red_mat_nnz(la->red_mat, row) - trans_extra;
    int64_t* base = acc.data();
    uint64_t* m = mask.data();

    while(k < nnz and idx[k] < c1) {
      const uint32_t rel = idx[k] - c0;
      int64_t* x = base + size_t(rel) * B;
      arith.template axmy<B>(x, a, val[k]);
      mark(m, rel);
      k++;
    }
    return k == nnz ? UINT32_MAX : k;
  }

  // Reduces up to B rows of mat against the reducer pivots. The rows come out
  // unnormalized and supported on columns that carry no reducer pivot.
  void run_batch(const size_t* rows, uint32_t count) {
    assert(count > 0 and count <= B);

    mult.clear();
    slot_red.clear();
    free_slots.clear();
    for(uint32_t l = 0; l < B; l++) {
      out_cols[l].clear();
      out_vals[l].clear();
      in_idx[l] = nullptr;
      in_val[l] = nullptr;
      in_nnz[l] = 0;
      in_cur[l] = 0;
      tail_val[l] = 0;
      tail_col[l] = 0;
    }

    for(uint32_t l = 0; l < count; l++) {
      const auto row = sparse_mat_row(mat, rows[l]);
      in_idx[l] = row->indices;
      in_val[l] = row->entries;
      in_nnz[l] = row->nnz;
      // The row's own proof-block entry is its last one; the rest of that
      // block is rebuilt from T below, so the sweep never sees any of it.
      if(trans_extra and in_nnz[l] > 0
         and in_idx[l][in_nnz[l] - 1] >= mon_cols) {
        in_nnz[l]--;
        tail_col[l] = in_idx[l][in_nnz[l]];
        tail_val[l] = in_val[l][in_nnz[l]];
      }
      assert(in_nnz[l] == 0 or in_idx[l][in_nnz[l] - 1] < mon_cols);
    }

    uint32_t min_in_block = next_input_block();

    for(uint32_t w = 0; w < nblocks; w++) {
      if(w < min_in_block and pending[w].empty())
        continue;

      const uint32_t c0 = w << blocked_log_cols;
      const uint32_t c1 = std::min(c0 + blocked_block_cols, mon_cols);
      int64_t* base = acc.data();
      uint64_t* m = mask.data();

      // 1. the still untouched part of the original rows
      for(uint32_t l = 0; l < count; l++) {
        while(in_cur[l] < in_nnz[l] and in_idx[l][in_cur[l]] < c1) {
          const uint32_t rel = in_idx[l][in_cur[l]] - c0;
          base[size_t(rel) * B + l] = in_val[l][in_cur[l]];
          mark(m, rel);
          in_cur[l]++;
        }
      }
      min_in_block = next_input_block();

      // 2. the reduction matrix carried over from earlier blocks
      auto& bucket = pending[w];
      for(size_t e = 0; e < bucket.size(); e++) {
        const block_carry c = bucket[e];
        const uint32_t k = apply_slice(
          c.row, c.cursor, c0, c1, mult.data() + size_t(c.slot) * B);
        if(k != UINT32_MAX)
          carry_forward(c.row, k, c.slot);
        else
          release_slot(c.slot);
      }
      bucket.clear();

      // 3. eliminate inside the block
      const uint32_t last = c1 - c0 - 1;
      uint32_t rel = 0;
      uint32_t a[B];
      while(true) {
        rel = next_marked(m, rel, last);
        if(rel > last)
          break;

        int64_t* x = base + size_t(rel) * B;
        uint32_t nz = 0;
        for(uint32_t l = 0; l < B; l++) {
          const int64_t v = x[l];
          // v must be smaller than 2^2b, i.e. 2^62
          assert(v >= 0 and v < 4611686018427387904);
          x[l] = 0;
          a[l] = v ? arith.reduce(v, l) : 0;
          nz |= a[l];
        }
        unmark(m, rel);
        if(!nz) {
          rel++;
          continue;
        }

        const uint32_t c = c0 + rel;
        const int32_t rr = la->red_pivots[c];
        if(rr >= 0) {
          const uint32_t slot = claim_slot(static_cast<uint32_t>(rr));
          std::copy(a, a + B, mult.data() + size_t(slot) * B);
          // entry 0 of the reducer is the pivot itself, already cleared
          const uint32_t k = apply_slice(static_cast<uint32_t>(rr),
                                         1,
                                         c0,
                                         c1,
                                         mult.data() + size_t(slot) * B);
          if(k != UINT32_MAX)
            carry_forward(static_cast<uint32_t>(rr), k, slot);
          else
            release_slot(slot);
        } else {
          for(uint32_t l = 0; l < count; l++)
            if(a[l]) {
              out_cols[l].push_back(c);
              out_vals[l].push_back(a[l]);
            }
        }
        rel++;
      }
    }

    for(uint32_t w = 0; w < nblocks; w++)
      pending[w].clear();

    for(uint32_t l = 0; l < count; l++)
      store_row(rows[l], l);
  }

  private:
  inline uint32_t claim_slot(uint32_t row) {
    if(!free_slots.empty()) {
      const uint32_t slot = free_slots.back();
      free_slots.pop_back();
      slot_red[slot] = row;
      return slot;
    }
    const uint32_t slot = static_cast<uint32_t>(slot_red.size());
    slot_red.push_back(row);
    mult.resize(mult.size() + B);
    return slot;
  }

  // A proof block is read off the whole reduction matrix at the end of the
  // batch, so there the slots have to stay.
  inline void release_slot(uint32_t slot) {
    if(!trans_extra)
      free_slots.push_back(slot);
  }

  inline uint32_t next_input_block() const {
    uint32_t best = UINT32_MAX;
    for(uint32_t l = 0; l < B; l++)
      if(in_cur[l] < in_nnz[l])
        best = std::min(best, in_idx[l][in_cur[l]] >> blocked_log_cols);
    return best;
  }

  inline void carry_forward(uint32_t row, uint32_t k, uint32_t slot) {
    const uint32_t* idx = red_mat_indices(la->red_mat, row);
    pending[idx[k] >> blocked_log_cols].push_back({ row, k, slot });
  }

  // Writes lane `l` back into row `r`, appending the proof block read off the
  // reduction matrix: reducer j contributes 1 to column mon_cols + j, so the
  // row picks up -T[l][j] there.
  void store_row(size_t r, uint32_t l) {
    const auto row = sparse_mat_row(mat, r);

    if(out_cols[l].empty()) {
      sparse_vec_clear(row);
      if(use_trace)
        __atomic_store_n(&la->red_mat->trace[r], 1, __ATOMIC_RELAXED);
      return;
    }

    trans.clear();
    if(trans_extra) {
      for(size_t s = 0; s < slot_red.size(); s++) {
        const uint32_t a = mult[s * B + l];
        if(a)
          trans.emplace_back(slot_red[s], arith.negate(a));
      }
      // reducers come up in decreasing row order, columns have to ascend
      std::sort(trans.begin(), trans.end());
    }

    const size_t extra = trans.size() + (tail_val[l] ? 1 : 0);
    const size_t nnz = out_cols[l].size() + extra;
    check_fits(nnz, UINT32_MAX, "number of nonzeros in a reduced row");
    sparse_vec_realloc(row, static_cast<uint32_t>(nnz));
    row->nnz = static_cast<uint32_t>(nnz);
    std::copy(out_cols[l].begin(), out_cols[l].end(), row->indices);
    std::copy(out_vals[l].begin(), out_vals[l].end(), row->entries);

    size_t k = out_cols[l].size();
    for(const auto& [j, v] : trans) {
      row->indices[k] = mon_cols + j;
      row->entries[k] = v;
      k++;
    }
    if(tail_val[l]) {
      row->indices[k] = tail_col[l];
      row->entries[k] = tail_val[l];
    }
  }
};
}
