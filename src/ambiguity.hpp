#ifndef AMBIGUITY_H
#define AMBIGUITY_H

#include <algorithm>
#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <span>

namespace f4ncgb {

template<typename I>
size_t inline compute_hash(const std::array<I, 7>& values) noexcept {
  size_t seed = 0;
  for(I value : values)
    boost::hash_combine(seed, value);
  return seed;
}

template<typename I>
struct ambiguity {
  std::array<I, 7> values;
  size_t hash_;
  mutable I fi_;
  mutable I fj_;
  mutable I lm_;

  using mon_id = I;
  using poly_id = I;

  inline ambiguity() = default;

  inline ambiguity(I d, I i, I j, I ai, I ci, I aj, I cj)
    : values{ d, i, j, ai, ci, aj, cj }
    , hash_(compute_hash(values)) {}

  inline I degree() const noexcept { return values[0]; }
  inline I i() const noexcept { return values[1]; }
  inline I j() const noexcept { return values[2]; }
  inline I ai() const noexcept { return values[3]; }
  inline I ci() const noexcept { return values[4]; }
  inline I aj() const noexcept { return values[5]; }
  inline I cj() const noexcept { return values[6]; }
  inline I fi() const noexcept { return fi_; }
  inline I fj() const noexcept { return fj_; }
  inline I lm() const noexcept { return lm_; }

  // Equality operator for unordered_set
  inline bool operator==(const ambiguity& other) const noexcept {
    return (hash_ == other.hash_) and (values == other.values);
  }

  template<typename PS, typename LmMap>
  void prepare(PS& poly, LmMap& lm_to_poly) const {
    poly_id ii = lm_to_poly[i()];
    poly_id jj = lm_to_poly[j()];

    fi_ = poly.multiply_front_and_back(ai(), ii, ci());
    lm_ = poly.get_lm_id(fi_);
    // ai * lm(f_i) * ci == aj * lm(f_j) * cj, 
    // so the second lm does not have to be computed again.
    fj_ = poly.multiply_front_and_back(aj(), jj, cj(), lm_);
  }
};

template<class T, std::size_t N, std::size_t M>
constexpr inline bool
endswith(std::span<T, N> data, std::span<T, M> suffix) noexcept {
  return data.size() >= suffix.size()
         && std::equal(data.end() - static_cast<long>(suffix.size()),
                       data.end(),
                       suffix.begin());
}
}

#endif// AMBIGUITY_H
