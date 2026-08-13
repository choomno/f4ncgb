#ifndef F4NCGB_HPP
#define F4NCGB_HPP

#include <flint/fmpz.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const char*
f4ncgb_version();

typedef struct f4ncgb_handle f4ncgb_handle;

typedef enum f4ncgb_state {
  F4NCGB_STATE_READY,
  F4NCGB_STATE_SOLVING,
} f4ncgb_state;

typedef enum f4ncgb_result {
  F4NCGB_OK = 0,
  F4NCGB_ERROR,
  F4NCGB_ARGERROR,
  F4NCGB_UNKNOWN,
} f4ncgb_result;

void
f4ncgb_set_msg_printing(bool);

f4ncgb_handle*
f4ncgb_init();

void
f4ncgb_free(f4ncgb_handle*);

f4ncgb_state
f4ncgb_get_state(const f4ncgb_handle*);

// Add a monomial with some coefficient.
//
// This builds up the current polynomial. Polynomials are
// automatically sorted after they are added like this.
const char*
f4ncgb_add(f4ncgb_handle*,
           long numerator,
           long denominator,
           size_t varcount,
           uint32_t* vars);

const char*
f4ncgb_end_poly(f4ncgb_handle*);

const char*
f4ncgb_set_blocks(f4ncgb_handle*, uint32_t blockcount, uint32_t* blocklengths);

const char*
f4ncgb_set_nvars(f4ncgb_handle*, uint32_t);

const char*
f4ncgb_set_characteristic(f4ncgb_handle*, uint32_t);

const char*
f4ncgb_set_maxiter(f4ncgb_handle*, uint32_t);

const char*
f4ncgb_set_maxdeg(f4ncgb_handle*, uint32_t);

const char*
f4ncgb_set_threads(f4ncgb_handle*, uint32_t);

const char*
f4ncgb_set_tracer(f4ncgb_handle*, bool);

const char*
f4ncgb_set_expanded_proof(f4ncgb_handle*, bool);

const char*
f4ncgb_set_output_file(f4ncgb_handle*, const char*);

const char*
f4ncgb_set_proof_file(f4ncgb_handle*, const char*);

typedef void (*f4ncgb_add_cb)(void* userdata,
                              fmpz_t numerator,
                              fmpz_t denominator,
                              size_t varcount,
                              const uint32_t* vars);

typedef void (*f4ncgb_end_poly_cb)(void* userdata);

f4ncgb_result
f4ncgb_solve(f4ncgb_handle*,
             void* userdata,
             f4ncgb_add_cb add_cb,
             f4ncgb_end_poly_cb end_cb);

f4ncgb_result
f4ncgb_reduce(f4ncgb_handle*,
              void* userdata,
              f4ncgb_add_cb add_cb,
              f4ncgb_end_poly_cb end_cb);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "coeff.hpp"

namespace f4ncgb {
class Solver {
  struct Deleter {
    void operator()(f4ncgb_handle* h) { f4ncgb_free(h); }
  };
  std::unique_ptr<f4ncgb_handle, Deleter> handle_;

  /// Every setter reports failure the same way: a message, or null on success.
  static void check(const char* msg) {
    if(msg)
      throw std::runtime_error(msg);
  }

  public:
  Solver() { handle_.reset(f4ncgb_init()); }
  ~Solver() = default;

  f4ncgb_state state() const { return f4ncgb_get_state(handle_.get()); }

  void add(long numerator, long denominator, std::vector<uint32_t> vars) {
    check(f4ncgb_add(
      handle_.get(), numerator, denominator, vars.size(), vars.data()));
  }

  void end_poly() { check(f4ncgb_end_poly(handle_.get())); }

  void set_blocks(std::vector<uint32_t> b) {
    check(f4ncgb_set_blocks(handle_.get(), b.size(), b.data()));
  }
  void set_nvars(uint32_t nvars) {
    check(f4ncgb_set_nvars(handle_.get(), nvars));
  }
  void set_characteristic(uint32_t characteristic) {
    check(f4ncgb_set_characteristic(handle_.get(), characteristic));
  }
  void set_maxiter(uint32_t maxiter) {
    check(f4ncgb_set_maxiter(handle_.get(), maxiter));
  }
  void set_maxdeg(uint32_t maxdeg) {
    check(f4ncgb_set_maxdeg(handle_.get(), maxdeg));
  }
  void set_threads(uint32_t threads) {
    check(f4ncgb_set_threads(handle_.get(), threads));
  }
  void set_tracer(bool tracer) {
    check(f4ncgb_set_tracer(handle_.get(), tracer));
  }
  void set_expanded_proof(bool expanded) {
    check(f4ncgb_set_expanded_proof(handle_.get(), expanded));
  }
  void set_output_file(const std::string& output_file) {
    check(f4ncgb_set_output_file(handle_.get(), output_file.c_str()));
  }
  void set_proof_file(const std::string& proof_file) {
    check(f4ncgb_set_proof_file(handle_.get(), proof_file.c_str()));
  }

  using add_cb = std::function<void(fmpz_t, fmpz_t, std::span<const uint32_t>)>;
  using end_poly_cb = std::function<void()>;

  using monomial = std::vector<uint32_t>;
  using polynomial = std::vector<std::tuple<coeff, coeff, monomial>>;

  /// The C entry points, which solve() and reduce() differ only by.
  using entry_point = f4ncgb_result (*)(f4ncgb_handle*,
                                        void*,
                                        f4ncgb_add_cb,
                                        f4ncgb_end_poly_cb);

  f4ncgb_result solve(add_cb add, end_poly_cb end) {
    return run(f4ncgb_solve, add, end);
  }
  f4ncgb_result reduce(add_cb add, end_poly_cb end) {
    return run(f4ncgb_reduce, add, end);
  }

  std::pair<f4ncgb_result, std::vector<polynomial>> solve() {
    return collect(f4ncgb_solve);
  }
  std::pair<f4ncgb_result, std::vector<polynomial>> reduce() {
    return collect(f4ncgb_reduce);
  }

  private:
  /// Hands the two std::functions to the C API as a pair of plain callbacks.
  f4ncgb_result run(entry_point entry, add_cb add, end_poly_cb end) {
    struct meta {
      add_cb a;
      end_poly_cb e;
    };
    meta m{ add, end };

    auto c_add_cb = [](void* userdata,
                       fmpz_t numerator,
                       fmpz_t denominator,
                       size_t varcount,
                       const uint32_t* vars) {
      meta& m = *static_cast<meta*>(userdata);
      return m.a(
        numerator, denominator, std::span<const uint32_t>(vars, varcount));
    };
    auto c_end_poly_cb = [](void* userdata) {
      meta& m = *static_cast<meta*>(userdata);
      return m.e();
    };

    return entry(
      handle_.get(), static_cast<void*>(&m), c_add_cb, c_end_poly_cb);
  }

  /// Gathers what the callbacks report into one polynomial per end_poly.
  std::pair<f4ncgb_result, std::vector<polynomial>> collect(entry_point entry) {
    std::vector<polynomial> polys;
    bool create = true;

    f4ncgb_result res = run(
      entry,
      [&polys, &create](
        fmpz_t numerator, fmpz_t denominator, std::span<const uint32_t> vars) {
        if(create) {
          polys.emplace_back();
          create = false;
        }
        coeff num;
        coeff den;
        if(numerator)
          fmpz_set(num.value, numerator);
        else
          fmpz_set_ui(num.value, 0);
        if(denominator)
          fmpz_set(den.value, denominator);
        else
          fmpz_set_ui(den.value, 1);

        polys.back().emplace_back(
          num, den, std::vector<uint32_t>(vars.begin(), vars.end()));
      },
      [&create]() { create = true; });

    return std::make_pair(res, std::move(polys));
  }
};
}
#endif

#endif
