#pragma once

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
// Older GCC compilers warn about unknown warnings, which is really annoying.
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Walloc-size"
#endif

#include <cassert>
#include <tuple>

#include "coeff.hpp"
#include "monomial_store.hpp"
#include "store.hpp"

namespace f4ncgb {
namespace internal {

template<metadata_concept M, typename I>
struct polynomial_metadata : public M {
  I coefficients;
  /// Position in the basis, used for proof logging. Wide enough for any basis
  /// the index type I can address, and free: the struct is padded to 12 bytes
  /// either way.
  uint32_t idx;
};

template<metadata_concept PM = internal::metadata<uint8_t>,
         metadata_concept MM = internal::metadata<uint8_t>,
         value_concept V = uint8_t,
         typename I = uint32_t,
         typename C = coeff,
         uint16_t Nblocks = 0>
class polynomial_store
  : public store<polynomial_store<PM, MM, V, I, C, Nblocks>,
                 polynomial_metadata<PM, I>,
                 I,
                 I> {

#ifdef F4NCGB_ENABLE_STORE_TRACE
  store_tracer tracer_ = store_tracer("polynomial_store");
#endif

  public:
  using self = polynomial_store<PM, MM, V, I, C, Nblocks>;
  using base = store<polynomial_store<PM, MM, V, I, C, Nblocks>,
                     polynomial_metadata<PM, I>,
                     I,
                     I>;
  using monomial_store_ = monomial_store<MM, V, I, Nblocks>;
  using monomial = monomial_store_::value_type;
  using metadata = polynomial_metadata<PM, I>;

  using polynomial_vec = std::vector<std::pair<C, I>>;
  using polynomial_nested_vec = std::vector<std::pair<C, std::vector<V>>>;

  friend base;

  constexpr static bool block_order = Nblocks > 1;

  consteval static size_t capacity() {
    return addressable_capacity<I, sizeof(C)>();
  }

  size_t cpool_capacity_ = 0;

  /// Two pools: the base class holds the monomial ids, cpool_ the coefficients.
  inline polynomial_store(monomial_store_& store)
    : base::store()
    , store_(store) {
    cpool_.reset(allocate_pool(capacity(), cpool_capacity_));
    // allocate_pool() rounds up to whole huge pages, so never let the bound
    // reach past what an id of type I can index.
    cpool_capacity_ = std::min(cpool_capacity_, capacity());
  }

  ~polynomial_store() {
    for(I i = 0; i < cpool_size_; ++i) {
      C* c = get_coefficients_raw(i);
      // Explicitly destruct the coefficent again.
      c->~C();
    }
  }

  /// Initialize a new polynomial. Remember to initialize the coefficents!
  inline std::tuple<metadata&, I*, C*> add(I length = 32) {
    auto [m, vv] = this->new_scratch(length);
    return std::tuple<metadata&, I*, C*>(
      reinterpret_cast<metadata&>(m), vv, get_coefficients_raw(cpool_size_));
  }
  inline I commit() { return this->insert_scratch(); }

  /// Consumes the coefficients of p, which is left holding zeros.
  inline I add_polynomial(polynomial_vec& p) {
    if(p.size() > std::numeric_limits<typename base::length_type>::max()) {
      throw length_overrun_exception(
        p.size(), std::numeric_limits<typename base::length_type>::max());
    }
    auto [m, monomials, coefficients] = add(p.size());
    for(size_t i = 0; i < p.size(); ++i) {
      C* c = new(coefficients + i) C;
      *c = std::move(p[i].first);
      monomials[i] = p[i].second;
    }
    m.length = p.size();
    return commit();
  }
  inline I add_polynomial(const polynomial_nested_vec& p) {
    auto [m, monomials, coefficients] = add(p.size());
    for(size_t i = 0; i < p.size(); ++i) {
      C* c = new(coefficients + i) C;
      *c = p[i].first;
      monomials[i] = store_.getid(p[i].second);
    }
    m.length = p.size();
    return commit();
  }

  inline std::span<C> get_coefficients(I id) {
    if(id == 0)
      return std::span<C>();
    metadata& m = this->get_metadata(id);
    return std::span<C>(get_coefficients_raw(m.coefficients), m.length);
  }

  inline std::span<const C> get_coefficients(I id) const {
    return const_cast<self*>(this)->get_coefficients(id);
  }

  inline I get_monomial_id(I id, I idx) const {
    if(id == 0)
      return 0;
    assert(idx < this->get_metadata(id).length);
    return (*this)[id][idx];
  }

  inline void sort_polynomial(std::vector<std::pair<C, I>>& p) {
    std::stable_sort(p.begin(), p.end(), [this](const auto& a, const auto& b) {
      return this->store_.template cmp<block_order>(b.second, a.second);
    });
  }

  inline I get_lm_id(I id) const {
    // discuss if polynomials are sorted
    // increasing or decreasing -- decreasing
    // is probably better
    return get_monomial_id(id, 0);
  }

  inline void set_idx(I id, uint32_t idx) {
    assert(idx != 0);
    if(id == 0)
      return;
    metadata& m = this->get_metadata(id);
    m.idx = idx;
  }

  inline uint32_t get_idx(I id) {
    if(id == 0)
      return 0;
    metadata& m = this->get_metadata(id);
    return m.idx;
  }

  private:
  /// f * p * b, as a new polynomial sharing p's coefficient array: only the
  /// monomials are built. A factor of 1 is passed as id 0, the empty monomial,
  /// so one-sided products go through here too.
  ///
  /// The result is scratch -- real_size_ and cpool_size_ are rolled back -- so
  /// it lives only until the next reset().
  template<bool have_lm>
  inline I multiply(I f, I p, I b, I known_lm) {
    assert(p != 0);

    size_t cpool_size_old = cpool_size_;
    size_t size_old = this->real_size_;
    metadata& p_metadata = this->get_metadata(p);
    const I length = p_metadata.length;
    auto [new_m, new_i, new_c] = add(length);
    new_m.length = length;

    const std::span<const I> p_mons = (*this)[p];

    // f and b are the same for every term, so their values are looked up once
    // here rather than once per term inside get_product_id().
    const auto f_v = store_[f];
    const auto b_v = store_[b];

    I i = 0;
    if constexpr(have_lm) {
      assert(length > 0);
      new_i[0] = known_lm;
      i = 1;
    }
    for(; i < length; ++i)
      new_i[i] = store_.get_product_id(f_v, p_mons[i], b_v);

    I g = commit();
    this->get_metadata(g).coefficients = p_metadata.coefficients;
    cpool_size_ = cpool_size_old;
    this->real_size_ = size_old;
    return g;
  }

  public:
  inline I multiply_front_and_back(I f, I p, I b) {
    return multiply<false>(f, p, b, 0);
  }
  /// Same product, with the leading monomial handed in rather than looked up.
  /// Both callers pick p precisely because they already hold f * lm(p) * b, so
  /// interning it again is a probe into the largest hash table in the program
  /// that can only return the id the caller started from. It is a lookup that
  /// always hits, never an insertion, so ids are assigned exactly as before.
  inline I multiply_front_and_back(I f, I p, I b, I known_lm) {
    return multiply<true>(f, p, b, known_lm);
  }

  const monomial_store_& get_monomial_store() const { return store_; }
  monomial_store_& get_monomial_store() { return store_; }

  std::ostream& print_polynomial(I i, std::ostream& o) const {
    auto c = this->get_coefficients(i);
    size_t j = 0;
    for(auto m : (*this)[i]) {
      o << c[j++];
      o << "*";
      store_.print_monomial(m, o);
    }
    return o;
  }

  protected:
  monomial_store_& store_;
  std::unique_ptr<std::byte[], typename base::pool_deleter> cpool_;
  size_t cpool_size_ = 0;

  inline C* get_coefficients_raw(I id) {
    if(static_cast<size_t>(id) * sizeof(C) + sizeof(C) > cpool_capacity_) {
      throw coefficient_overrun_exception(id * sizeof(C), cpool_capacity_);
    }
    return reinterpret_cast<C*>(cpool_.get()
                                + static_cast<size_t>(id) * sizeof(C));
  }

  inline void new_entry(I id) {
    // Commit the new coefficients to the coefficient array.
    metadata& m = this->get_metadata(id + 1);
    m.coefficients = cpool_size_;
    cpool_size_ += m.length;
  }
};
}
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
