#pragma once

#include <array>
#include <bit>
#include <cstring>
#include <optional>
#include <vector>

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
class monomial_store : public store<monomial_store<M, V, I, Nblocks>, M, V, I> {
  using self = monomial_store<M, V, I, Nblocks>;
  using base = store<self, M, V, I>;
  friend base;

  using ambiguity_ = ambiguity<I>;

#ifdef F4NCGB_ENABLE_STORE_TRACE
  store_tracer tracer_ = store_tracer("monomial_store");
#endif

  //  `tag` holds 32 bits of the word hash and doubles as the key the index is
  //  derived from, so growing the table never has to walk the words again.
  struct word_entry {
    I id = 0;
    uint32_t tag = 0;
  };

  static constexpr uint64_t table_mixer = 0x9e3779b97f4a7c15ull;

  // Linear probing degrades sharply near a full table, and eight entries share
  // a cache line, so the chains this leaves are cheap to walk.
  static constexpr size_t table_load_num = 7;
  static constexpr size_t table_load_den = 10;

  std::vector<word_entry> table_;
  size_t table_mask_ = 0;
  size_t table_shift_ = 64;
  size_t table_limit_ = 0;
  size_t table_size_ = 0;

  boost::hash<std::span<const V>> hasher_;

  // Staging for get_product_ids(), reused across rows.
  std::vector<V> row_;

  inline size_t table_index(uint32_t tag) const {
    return static_cast<size_t>((static_cast<uint64_t>(tag) * table_mixer)
                               >> table_shift_);
  }

  void table_rehash(size_t want) {
    std::vector<word_entry> old;
    old.swap(table_);

    table_.assign(want, word_entry{});
    table_mask_ = want - 1;
    table_shift_ = 64 - static_cast<size_t>(std::countr_zero(want));
    table_limit_ = want * table_load_num / table_load_den;

    for(const word_entry& e : old) {
      if(e.id == 0)
        continue;
      size_t i = table_index(e.tag);
      while(table_[i].id != 0)
        i = (i + 1) & table_mask_;
      table_[i] = e;
    }
  }

  // Finds the word, or reports the free slot it would be inserted into.
  inline I table_find(const std::span<const V>& v,
                      uint32_t tag,
                      size_t& pos) const {
    size_t i = table_index(tag);
    for(;;) {
      const word_entry& e = table_[i];
      if(e.id == 0) {
        pos = i;
        return 0;
      }
      if(e.tag == tag and base::V_equal((*this)[e.id], v))
        return e.id;
      i = (i + 1) & table_mask_;
    }
  }

  // `pos` comes from the table_find() that just missed; a rehash invalidates
  // it, so the slot is looked up again in that case.
  inline void table_insert(size_t pos, I id, uint32_t tag) {
    if(table_size_ >= table_limit_) {
      table_rehash(table_.size() * 2);
      pos = table_index(tag);
      while(table_[pos].id != 0)
        pos = (pos + 1) & table_mask_;
    }
    table_[pos] = word_entry{ id, tag };
    table_size_++;
  }

  protected:
  inline I commit_scratch(size_t pos, uint32_t tag) {
    const I res = base::insert_scratch();
    table_insert(pos, res, tag);
    return res;
  }

  std::optional<I> find(const std::span<const V>& v) const {
    F4NCGB_PROFILE(gstats.store_find_calls++);
    if(v.size() == 0)
      return 0;

    size_t pos;
    const I id = table_find(v, static_cast<uint32_t>(hasher_(v)), pos);
    if(id == 0)
      return std::nullopt;

    F4NCGB_PROFILE(gstats.store_find_hits++);
    return id;
  }

  static inline V* copy_values(const std::span<const V>& src, V* dst) {
    const size_t n = src.size();
    const V* s = src.data();
    if(n <= 16) {
      for(size_t i = 0; i < n; i++)
        dst[i] = s[i];
    } else {
      std::memcpy(dst, s, n * sizeof(V));
    }
    return dst + n;
  }

  public:
  monomial_store() { table_rehash(1024); }

  inline I getid(const std::span<const V>& v) {
    F4NCGB_PROFILE(gstats.store_find_calls++);
    if(v.size() == 0)
      return 0;

    const uint32_t tag = static_cast<uint32_t>(hasher_(v));
    size_t pos;
    if(const I id = table_find(v, tag, pos)) {
      F4NCGB_PROFILE(gstats.store_find_hits++);
      return id;
    }

    auto [m, vv] = base::new_scratch(v.size());
    copy_values(v, vv);
    m.length = v.size();
    return commit_scratch(pos, tag);
  }
  inline I getid(std::vector<V> v) {
    std::span<const V> s(v.begin(), v.size());
    return getid(s);
  }

  inline I get_product_id(I a, I b, I c = 0) {
    return get_product_id((*this)[a], b, (*this)[c]);
  }

  inline I get_product_id(const std::span<const V>& a,
                          I b,
                          const std::span<const V>& c) {
    const std::span<const V> b_v = (*this)[b];
    const size_t length_combined = a.size() + b_v.size() + c.size();
    if(length_combined == 0)
      return 0;
    if(length_combined
       > std::numeric_limits<typename base::length_type>::max()) {
      throw length_overrun_exception(
        length_combined,
        std::numeric_limits<typename base::length_type>::max());
    }

    auto [m, vv] = base::new_scratch(length_combined);
    m.length = length_combined;
    V* it = copy_values(a, vv);
    it = copy_values(b_v, it);
    copy_values(c, it);

    const std::span<const V> w(vv, length_combined);

    F4NCGB_PROFILE(gstats.store_find_calls++);
    const uint32_t tag = static_cast<uint32_t>(hasher_(w));
    size_t pos;
    if(const I id = table_find(w, tag, pos)) {
      F4NCGB_PROFILE(gstats.store_find_hits++);
      // the scratch tip is simply left uncommitted
      return id;
    }

    return commit_scratch(pos, tag);
  }

  // The ids of a*m*c for every m in `ms`, with a and c fixed.
  inline void get_product_ids(const std::span<const V>& a,
                              const std::span<const I>& ms,
                              const std::span<const V>& c,
                              I* out) {
    // Bounded so that the words staged ahead of a probe are still in cache by
    // the time that probe runs.
    constexpr size_t chunk = 32;
    struct staged {
      uint32_t off;
      uint32_t len;
      uint32_t tag;
    };
    std::array<staged, chunk> st;

    const size_t na = a.size(), nc = c.size();

    for(size_t lo = 0; lo < ms.size(); lo += chunk) {
      const size_t hi = std::min(lo + chunk, ms.size());
      row_.clear();

      for(size_t i = lo; i < hi; i++) {
        const std::span<const V> b_v = (*this)[ms[i]];
        const size_t n = na + b_v.size() + nc;

        staged& e = st[i - lo];
        e.off = static_cast<uint32_t>(row_.size());
        e.len = static_cast<uint32_t>(n);
        if(n == 0)
          continue;

        if(n > std::numeric_limits<typename base::length_type>::max()) {
          throw length_overrun_exception(
            n, std::numeric_limits<typename base::length_type>::max());
        }

        row_.resize(row_.size() + n);
        V* it = copy_values(a, row_.data() + e.off);
        it = copy_values(b_v, it);
        copy_values(c, it);

        e.tag = static_cast<uint32_t>(
          hasher_(std::span<const V>(row_.data() + e.off, n)));
        __builtin_prefetch(&table_[table_index(e.tag)]);
      }

      for(size_t i = lo; i < hi; i++) {
        const staged& e = st[i - lo];
        if(e.len == 0) {
          out[i] = 0;
          continue;
        }

        const std::span<const V> w(row_.data() + e.off, e.len);

        F4NCGB_PROFILE(gstats.store_find_calls++);
        size_t pos;
        if(const I id = table_find(w, e.tag, pos)) {
          F4NCGB_PROFILE(gstats.store_find_hits++);
          out[i] = id;
          continue;
        }

        auto [m, vv] = base::new_scratch(e.len);
        copy_values(w, vv);
        m.length = e.len;
        out[i] = commit_scratch(pos, e.tag);
      }
    }
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
