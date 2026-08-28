#pragma once

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

namespace f4ncgb {
namespace internal {

//------------------------------------------------------------------------------

// Everything but the coefficient of a cofactor term c*a*[i]*b.
template<typename I>
struct cofactor_key {
  I a;
  int i;
  I b;

  inline bool operator==(const cofactor_key& other) const noexcept {
    return a == other.a and i == other.i and b == other.b;
  }
};

template<typename I>
struct cofactor_key_hash {
  inline size_t operator()(const cofactor_key<I>& k) const noexcept {
    size_t seed = 0;
    boost::hash_combine(seed, k.a);
    boost::hash_combine(seed, k.i);
    boost::hash_combine(seed, k.b);
    return seed;
  }
};

//------------------------------------------------------------------------------

// Divides a row by the gcd of its denominator and all of its coefficients,
// so that the pair is the primitive representation of the rational vector
// it stands for.
template<typename Cofactor, typename C>
void
cancel_content(std::vector<Cofactor>& cf, C& denom) {
  if(cf.empty())
    return;

  C g = denom;
  for(const auto& c : cf) {
    g.gcd_inplace(c.c);
    if(g.is_one())
      return;
  }

  for(auto& c : cf)
    c.c /= g;
  denom /= g;
}

//------------------------------------------------------------------------------

template<typename I, typename C, typename Cofactor, typename MonStore>
struct cofactor_expander {
  using key = cofactor_key<I>;
  using map = boost::unordered_flat_map<key, C, cofactor_key_hash<I>>;

  MonStore& mons;
  std::vector<std::vector<Cofactor>>& rows;
  std::vector<C>& denoms;

  // Reused across rows so that the composition does not reallocate per row.
  map acc;

  cofactor_expander(MonStore& mons_,
                    std::vector<std::vector<Cofactor>>& rows_,
                    std::vector<C>& denoms_)
    : mons(mons_)
    , rows(rows_)
    , denoms(denoms_) {}

  //----------------------------------------------------------------------------

  static inline bool points_into(const Cofactor& t,
                                 size_t lo,
                                 size_t hi) noexcept {
    return t.i() >= 0 and static_cast<size_t>(t.i()) >= lo
           and static_cast<size_t>(t.i()) < hi;
  }

  inline void add_term(const C& c, I a, int i, I b) {
    if(c.is_zero())
      return;
    auto [it, inserted] = acc.try_emplace(key{ a, i, b });
    if(inserted)
      it->second = c;
    else
      it->second += c;
  }

  //----------------------------------------------------------------------------

  // Rewrites row n over what preceded lo, given that every row in [lo, hi)
  // is already written that way. Terms pointing below lo are carried through.
  void substitute(size_t n, size_t lo, size_t hi) {
    auto& terms = rows[n];

    bool any = false;
    for(const auto& t : terms)
      if(points_into(t, lo, hi)) {
        any = true;
        break;
      }
    if(!any)
      return;

    // Substituting row j multiplies by a coefficient of rows[j], which is
    // written over denoms[j] rather than over 1. Collecting those into one
    // denominator for the whole row keeps every coefficient an integer, as each
    // of them divides it.
    C common(1L);
    for(const auto& t : terms)
      if(points_into(t, lo, hi))
        common.lcm_inplace(denoms[static_cast<size_t>(t.i())]);

    acc.clear();
    acc.reserve(2 * terms.size());

    for(const auto& t : terms) {
      // carried through, but scaled to the common denominator
      if(!points_into(t, lo, hi)) {
        add_term(t.c * common, t.a(), t.i(), t.b());
        continue;
      }

      const size_t j = static_cast<size_t>(t.i());
      const C scale = common / denoms[j];
      for(const auto& u : rows[j])
        add_term(t.c * u.c * scale,
                 mons.get_product_id(t.a(), u.a()),
                 u.i(),
                 mons.get_product_id(u.b(), t.b()));
    }

    terms.clear();
    terms.reserve(acc.size());
    for(auto& [k, c] : acc)
      if(!c.is_zero())
        terms.emplace_back(std::move(c), k.a, k.i, k.b);

    denoms[n] *= common;
    cancel_content(terms, denoms[n]);
  }

  //----------------------------------------------------------------------------

  /// Rewrites every row created in windows [lo, hi) over what preceded window
  /// lo. window_start[w] is the first row index of window w.
  void expand(const std::vector<size_t>& window_start, size_t lo, size_t hi) {
    if(hi - lo <= 1)
      return;

    const size_t m = lo + (hi - lo) / 2;
    expand(window_start, lo, m);
    expand(window_start, m, hi);

    // Both halves are now written over what preceded them
    for(size_t n = window_start[m]; n < window_start[hi]; n++)
      substitute(n, window_start[lo], window_start[m]);
  }

  //----------------------------------------------------------------------------

  /// Rewrites every row over the input.
  void expand_all(const std::vector<size_t>& window_start) {
    const size_t windows = window_start.size() - 1;
    if(windows == 0)
      return;

    // Mark the input polynomials negative so that substitute() can tell the
    // two apart.
    for(size_t n = window_start[0]; n < window_start[1]; n++)
      for(auto& t : rows[n])
        t.t.i = -t.t.i - 1;

#ifndef NDEBUG
    for(size_t n = window_start[0]; n < window_start[1]; n++)
      for(const auto& t : rows[n])
        assert(t.i() < 0);

    for(size_t w = 1; w < windows; w++)
      for(size_t n = window_start[w]; n < window_start[w + 1]; n++)
        for(const auto& t : rows[n])
          assert(t.i() >= 0 and static_cast<size_t>(t.i()) < window_start[w]);
#endif

    expand(window_start, 0, windows);
  }
};
}
}
