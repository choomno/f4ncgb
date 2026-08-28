#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <flint/fmpq.h>
#include <flint/fmpz.h>

#include "coeff.hpp"
#include "diagnostics.hpp"
#include "linear_algebra.hpp"
#include "linear_algebra_internal.hpp"
#include "sparse_rref/sparse_mat.h"

namespace f4ncgb {

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

inline coeff
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

  for(size_t i = 0; i < entries_red.size(); i++)
    if(la.represents(i))
      visit(entries_red[i]);

  return h;
}
//------------------------------------------------------------------------------

inline void
crt_reconstruction(fmpz*& entries,
                   std::vector<std::pair<size_t, size_t>>& idxs,
                   std::vector<sparse_mat_struct<uint32_t>*>& rrefs,
                   std::vector<uint32_t>& primes,
                   size_t n_piv) {
  if(rrefs.size() == 0)
    return;

  std::vector<std::vector<uint32_t>> nnz_pos(n_piv);
  for(size_t i = 0; i < n_piv; i++) {
    auto& cols = nnz_pos[i];
    const auto first = sparse_mat_row(rrefs[0], i);

    bool same = true;
    for(size_t k = 1; k < rrefs.size() and same; k++) {
      const auto row = sparse_mat_row(rrefs[k], i);
      same = row->nnz == first->nnz
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

inline bool
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

inline void
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
}
