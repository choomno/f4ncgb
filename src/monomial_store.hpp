#pragma once

#include "ambiguity.hpp"
#include "store.hpp"

namespace f4ncgb {
namespace internal {

template<internal::metadata_concept M = internal::metadata<uint8_t>,
         internal::value_concept V = uint8_t,
         typename I = uint32_t,
         uint16_t Nblocks = 0>
class monomial_store;

template<metadata_concept M, value_concept V, typename I, uint16_t Nblocks>
class monomial_store : public store<monomial_store<M, V, I>, M, V, I> {
  using self = monomial_store<M, V, I>;
  using base = store<self, M, V, I>;
  friend base;

  using ambiguity_ = ambiguity<I>;

#ifdef F4NCGB_ENABLE_STORE_TRACE
  store_tracer tracer_ = store_tracer("monomial_store");
#endif

  /// Interns monomials: maps the variables to the id they were stored under.
  boost::unordered_flat_map<std::span<const V>,
                            I,
                            boost::hash<std::span<const V>>,
                            typename base::V_span_equality_struct>
    map_;

  protected:
  /// Interning is the largest single cost of a run and it is bound by memory
  /// latency, not by hashing: the table reaches tens of millions of entries
  /// (48 M on serre-e6-17, 5.5 GB resident) so every probe is a DRAM access.
  /// Reserving it up front to remove all rehashing was measured at 2 % for an
  /// extra gigabyte, and the two structural alternatives -- keying by id, or
  /// caching products by their factors -- were both tried before and found
  /// slower, since each adds an indirection or a second large table. Getting
  /// past this needs fewer interned monomials, not a faster table.
  inline void new_entry(I id) { map_.insert(std::pair((*this)[id + 1], id)); }

  std::optional<I> find(const std::span<const V>& v) const {
    F4NCGB_PROFILE(gstats.store_find_calls++);
    if(v.size() == 0)
      return 0;

    const auto it = map_.find(v);
    if(it == map_.end())
      return std::nullopt;

    F4NCGB_PROFILE(gstats.store_find_hits++);
    return it->second + 1;
  }

  public:
  inline const I getid(const std::span<const V>& v) {
    auto id = find(v);
    if(!id)
      id = base::insert(v);
    return *id;
  }
  inline const I getid(std::vector<V> v) {
    std::span<const V> s(v.begin(), v.size());
    return getid(s);
  }

  /// Id of a*b*c. Absent factors are passed as id 0, the empty monomial, so
  /// this covers products of two as well.
  inline I get_product_id(I a, I b, I c = 0) {
    const size_t length_combined = static_cast<size_t>(base::get_length(a))
                                   + base::get_length(b) + base::get_length(c);
    if(length_combined
       > std::numeric_limits<typename base::length_type>::max()) {
      throw length_overrun_exception(
        length_combined,
        std::numeric_limits<typename base::length_type>::max());
    }

    auto [m, vv] = base::new_scratch(length_combined);
    m.length = length_combined;
    auto a_it = (*this)[a];
    auto b_it = (*this)[b];
    auto c_it = (*this)[c];
    auto it = std::copy(a_it.begin(), a_it.end(), vv);
    it = std::copy(b_it.begin(), b_it.end(), it);
    std::copy(c_it.begin(), c_it.end(), it);

    auto prod_idx = find(std::span(vv, length_combined));
    if(!prod_idx)
      prod_idx = base::insert_scratch();
    return *prod_idx;
  }

  std::ostream& print_monomial(I i, std::ostream& o) const {
    auto m = (*this)[i];
    o << "(";
    for(const auto& v : m)
      o << static_cast<int>(v) << ", ";
    o << ")";
    return o;
  }

  //-----------------------------------------------------------------
  // if v is  blocks[i] <= v <= blocks[i+1], then v is in block i
  std::vector<size_t> blocks;
  inline void set_blocks(std::vector<size_t> block_sizes) {
    blocks.resize(block_sizes.size() + 1);
    size_t n = 0;
    for(size_t i = 0; i < block_sizes.size(); i++) {
      blocks[i] = n;
      n += block_sizes[i];
    }
    blocks[blocks.size() - 1] = n;
  }

  template<bool block_order>
  inline bool cmp(I a, I b) {
    // this is a strict order
    if(a == b)
      return false;

    size_t la, lb;

    if constexpr(block_order) {
      auto a_it = (*this)[a];
      auto b_it = (*this)[b];

      // Counts the variables of both monomials per block on every comparison,
      // so sorting under a block order costs O(blocks * length) per compare
      // where the plain order costs a length lookup. Caching the per-block
      // degrees in the metadata would fix that; no benchmark uses a block
      // order, so the cost is recorded rather than guessed at.
      for(size_t i = 0; i < Nblocks; i++) {
        size_t n = Nblocks - i;
        la = (size_t)std::ranges::count_if(
          a_it,
          [l = blocks[n - 1], u = blocks[n]](V v) { return l < v && v <= u; });
        lb = (size_t)std::ranges::count_if(
          b_it,
          [l = blocks[n - 1], u = blocks[n]](V v) { return l < v && v <= u; });
        if(la != lb)
          return la < lb;
      }
    } else {
      // compare lengths
      la = base::get_length(a);
      lb = base::get_length(b);
      if(la != lb)
        return la < lb;
    }

    // compare monomials lexicographically
    //
    // Deliberately not memcmp: that compares the byte representation, which
    // only coincides with the order of the variables while V is a single byte.
    // For V = uint16_t on a little-endian host it would order the variables by
    // (v & 0xff, v >> 8) instead of by v. It also lets the two monomials differ
    // in length, which the block order does not otherwise guarantee.
    auto a_it = (*this)[a];
    auto b_it = (*this)[b];

    return std::lexicographical_compare(
      a_it.begin(), a_it.end(), b_it.begin(), b_it.end());
  }
  //-----------------------------------------------------------------
  std::ostream& print_ambiguity(const ambiguity_& a, std::ostream& o) const {
    o << "(" << a.degree() << ", ";
    print_monomial(a.ai(), o);
    o << ", ";
    print_monomial(a.ci(), o);
    o << ", ";
    print_monomial(a.aj(), o);
    o << ", ";
    print_monomial(a.cj(), o);
    o << ", " << static_cast<int>(a.i()) << ", " << static_cast<int>(a.j())
      << ")\n";
    return o;
  }
};
}
}
