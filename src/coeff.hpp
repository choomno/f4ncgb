#pragma once

#include <cstddef>
#include <flint/fmpq.h>
#include <flint/fmpz.h>
#include <iostream>
#include <utility>

namespace f4ncgb {

inline void
fmpz_cleanup(fmpz* arr, size_t len) {
  for(size_t i = 0; i < len; i++)
    fmpz_clear(arr + i);
  delete[] arr;
}

struct coeff {
  fmpz_t value;

  coeff() { fmpz_init(value); }
  ~coeff() { fmpz_clear(value); }

  coeff(const coeff& other) {
    fmpz_init(value);
    fmpz_set(value, other.value);
  }

  coeff(coeff&& other) noexcept {
    fmpz_init(value);
    fmpz_swap(value, other.value);
  }

  coeff(long x) {
    fmpz_init(value);
    fmpz_set_si(value, x);
  }

  friend std::ostream& operator<<(std::ostream& os, const coeff& c) {
    char* s = fmpz_get_str(nullptr, 10, c.value);
    os << s;
    flint_free(s);
    return os;
  }

  bool operator==(const coeff& other) const {
    return fmpz_equal(value, other.value) != 0;
  }

  bool operator!=(const coeff& other) const { return !(*this == other); }

  coeff& operator=(const coeff& other) {
    if(this != &other)
      fmpz_set(value, other.value);
    return *this;
  }

  coeff& operator=(coeff&& other) noexcept {
    if(this != &other)
      fmpz_swap(value, other.value);
    return *this;
  }

  coeff& operator*=(const coeff& other) {
    fmpz_mul(value, value, other.value);
    return *this;
  }

  coeff& operator/=(long b) {
    fmpz_divexact_si(value, value, b);
    return *this;
  }

  coeff& operator/=(const coeff& b) {
    fmpz_divexact(value, value, b.value);
    return *this;
  }

  friend coeff operator+(const coeff& a, const coeff& b) {
    coeff r;
    fmpz_add(r.value, a.value, b.value);
    return r;
  }

  friend coeff operator+(const coeff& a, ulong b) {
    coeff r;
    fmpz_add_ui(r.value, a.value, b);
    return r;
  }

  friend coeff operator*(const coeff& a, const coeff& b) {
    coeff r;
    fmpz_mul(r.value, a.value, b.value);
    return r;
  }

  friend coeff operator/(const coeff& a, const coeff& b) {
    coeff r;
    fmpz_divexact(r.value, a.value, b.value);
    return r;
  }

  friend coeff operator*(const coeff& a, ulong b) {
    coeff r;
    fmpz_mul_ui(r.value, a.value, b);
    return r;
  }

  friend coeff operator*(ulong a, const coeff& b) {
    coeff r;
    fmpz_mul_ui(r.value, b.value, a);
    return r;
  }

  friend coeff operator+(ulong a, const coeff& b) { return b + a; }

  friend bool operator<(const coeff& a, const coeff& b) {
    return fmpz_cmp(a.value, b.value) < 0;
  }

  int sign() const { return fmpz_sgn(value); }

  bool is_zero() const { return fmpz_is_zero(value); }

  bool is_unit() const { return fmpz_is_pm1(value); }

  coeff abs() const {
    coeff r;
    fmpz_abs(r.value, value);
    return r;
  }

  void lcm_inplace(const coeff& b) { fmpz_lcm(value, value, b.value); }

  void lcm_inplace(long b) {
    coeff tmp(b);
    lcm_inplace(tmp);
  }

  void gcd_inplace(const coeff& b) { fmpz_gcd(value, value, b.value); }

  bool is_one() const { return fmpz_is_one(value); }
};

struct rat_coeff {
  fmpq_t value;

  rat_coeff() { fmpq_init(value); }
  ~rat_coeff() { fmpq_clear(value); }

  void update(const coeff& num, const coeff& den) {
    fmpq_set_fmpz_frac(value, num.value, den.value);
  }

  friend std::ostream& operator<<(std::ostream& os, const rat_coeff& c) {
    char* s = fmpq_get_str(nullptr, 10, c.value);
    os << s;
    flint_free(s);
    return os;
  }
  int sign() const { return fmpq_sgn(value); }

  bool is_zero() const { return fmpq_is_zero(value); }

  bool is_unit() const { return fmpq_is_pm1(value); }

  rat_coeff abs() const {
    rat_coeff r;
    fmpq_abs(r.value, value);
    return r;
  }
};

}
