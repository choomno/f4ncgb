#include "diagnostics.hpp"
#include "f4ncgb.hpp"
#include "parser.hpp"
#include "solve.hpp"
#include <config.hpp>
#include <exception>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

#include <ulong_extras.h>

using namespace f4ncgb;

#define REQUIRE_HANDLE(h)     \
  if(!h) {                    \
    return "no handle given"; \
  }

/// Every setter needs a handle that is not in the middle of a computation.
#define REQUIRE_READY(h)                      \
  REQUIRE_HANDLE(h)                           \
  if(h->state != F4NCGB_STATE_READY) {        \
    return "invalid state, must be in READY"; \
  }

using monomial = std::vector<uint32_t>;
using polynomial = std::vector<std::tuple<long, long, monomial>>;

typedef struct f4ncgb_handle {
  f4ncgb_state state = F4NCGB_STATE_READY;
  const char* output_file = "";
  const char* proof_file = "";

  parser_context ctx;
  solve_options opts;

  polynomial current_polynomial;
  std::vector<polynomial> polynomials;
} f4ncgb_handle;

extern "C" void
f4ncgb_set_msg_printing(bool printing) {
  msg_printing = printing ? 1 : 0;
}

extern "C" const char*
f4ncgb_version() {
  return F4NCGB_VERSION;
}

extern "C" f4ncgb_handle*
f4ncgb_init() {
  f4ncgb_handle* h = new f4ncgb_handle;

  h->ctx.impl = [h](parse_add_cb add_cb,
                    void* add_cb_userdata,
                    parse_monomial_boundary_cb boundary_cb,
                    void* boundary_cb_userdata) -> parse_res {
    for(const auto& poly : h->polynomials) {
      for(const auto& [numerator, denominator, mono] : poly) {
        for(uint32_t v : mono) {
          if(parse_res err = add_cb(add_cb_userdata, v))
            return err;
        }
        boundary_cb(boundary_cb_userdata,
                    numerator,
                    denominator,
                    numerator == denominator);
      }
      if(parse_res err = add_cb(add_cb_userdata, 0))
        return err;
    }
    return std::nullopt;
  };

  return h;
}

extern "C" void
f4ncgb_free(f4ncgb_handle* h) {
  delete h;
}

extern "C" f4ncgb_state
f4ncgb_get_state(const f4ncgb_handle* h) {
  return h->state;
}

extern "C" const char*
f4ncgb_add(f4ncgb_handle* h,
           long numerator,
           long denominator,
           size_t varcount,
           uint32_t* vars) {
  REQUIRE_READY(h);

  if(varcount) {
    h->current_polynomial.emplace_back(
      numerator, denominator, monomial(vars, vars + varcount));
  } else {
    h->current_polynomial.emplace_back(numerator, denominator, monomial());
  }
  return nullptr;
}

extern "C" const char*
f4ncgb_end_poly(f4ncgb_handle* h) {
  REQUIRE_READY(h);
  h->polynomials.emplace_back(h->current_polynomial);
  h->current_polynomial.clear();
  return nullptr;
}

extern "C" const char*
f4ncgb_set_blocks(f4ncgb_handle* h,
                  uint32_t blockcount,
                  uint32_t* blocklengths) {
  REQUIRE_READY(h);

  h->opts.blocks.resize(blockcount);
  uint32_t v = 1;
  for(size_t i = 0; i < blockcount; ++i) {
    h->opts.blocks[i].resize(blocklengths[i]);
    auto begin = h->opts.blocks[i].begin();
    auto end = h->opts.blocks[i].end();
    std::iota(begin, end, v);
    v += blocklengths[i];
  }

  // v has run past the last variable, which is numbered v - 1
  if(h->opts.num_vars == 0) {
    h->opts.num_vars = v - 1;
  }

  return nullptr;
}

extern "C" const char*
f4ncgb_set_weights(f4ncgb_handle* h,
                   uint32_t nweights,
                   const uint32_t* weights) {
  REQUIRE_READY(h);

  h->opts.weights.assign(weights, weights + nweights);

  if(h->opts.num_vars == 0) {
    h->opts.num_vars = nweights;
  }

  return nullptr;
}

extern "C" const char*
f4ncgb_set_nvars(f4ncgb_handle* h, uint32_t nvars) {
  REQUIRE_READY(h);
  h->opts.num_vars = nvars;
  return nullptr;
}

extern "C" const char*
f4ncgb_set_characteristic(f4ncgb_handle* h, uint32_t characteristic) {
  REQUIRE_READY(h);

  if(characteristic > 2147483647l) {// 2^31 -1
    return "provided characteristic is too large. Only p < 2^31 supported";
  } else if(characteristic != 0 and !n_is_prime(characteristic)) {
    return "provided nonzero characteristic is not prime";
  }

  h->opts.characteristic = characteristic;
  return nullptr;
}

extern "C" const char*
f4ncgb_set_maxiter(f4ncgb_handle* h, uint32_t maxiter) {
  REQUIRE_READY(h);
  h->opts.maxiter = maxiter;
  return nullptr;
}

extern "C" const char*
f4ncgb_set_maxdeg(f4ncgb_handle* h, uint32_t maxdeg) {
  REQUIRE_READY(h);
  h->opts.maxdeg = maxdeg;
  return nullptr;
}

extern "C" const char*
f4ncgb_set_threads(f4ncgb_handle* h, uint32_t threads) {
  REQUIRE_READY(h);
  h->opts.threads = threads;
  return nullptr;
}

extern "C" const char*
f4ncgb_set_output_file(f4ncgb_handle* h, const char* output_file) {
  REQUIRE_READY(h);
  h->output_file = output_file;
  return nullptr;
}

extern "C" const char*
f4ncgb_set_proof_file(f4ncgb_handle* h, const char* proof_file) {
  REQUIRE_READY(h);
  h->proof_file = proof_file;
  h->opts.proof_file = proof_file;
  h->opts.proof_level = std::max(size_t(1), h->opts.proof_level);
  return nullptr;
}

extern "C" const char*
f4ncgb_set_expanded_proof(f4ncgb_handle* h, bool expanded) {
  REQUIRE_READY(h);
  if(expanded)
    h->opts.proof_level = 2;
  return nullptr;
}

extern "C" const char*
f4ncgb_set_tracer(f4ncgb_handle* h, bool tracer) {
  REQUIRE_READY(h);
  h->opts.tracer = tracer;
  return nullptr;
}

f4ncgb_result
f4ncgb_solve_intern_(f4ncgb_handle* h,
                     void* userdata,
                     f4ncgb_add_cb add_cb,
                     f4ncgb_end_poly_cb end_cb,
                     bool reduce) {
  if(h->opts.num_vars > 0 && h->opts.num_blocks() == 0) {
    h->opts.blocks.emplace_back();
    h->opts.blocks[0].resize(h->opts.num_vars);
    auto begin = h->opts.blocks[0].begin();
    auto end = h->opts.blocks[0].end();
    std::iota(begin, end, 1);
  }

  if(h->opts.num_vars == 0)
    return F4NCGB_ARGERROR;
  if(h->opts.num_vars >= std::numeric_limits<uint16_t>::max())
    return F4NCGB_ARGERROR;
  for(const auto& poly : h->polynomials)
    for(const auto& mono : poly)
      for(uint32_t v : std::get<2>(mono))
        if(v == 0 || v > h->opts.num_vars)
          return F4NCGB_ARGERROR;
  if(h->opts.validate_blocks())
    return F4NCGB_ARGERROR;
  if(h->opts.validate_weights())
    return F4NCGB_ARGERROR;
  if(h->opts.characteristic > 2147483647l)// 2^31 -1
    return F4NCGB_ARGERROR;
  if(h->opts.characteristic != 0 and !n_is_prime(h->opts.characteristic))
    return F4NCGB_ARGERROR;
  if(!h->proof_file and h->opts.proof_level > 0)
    return F4NCGB_ARGERROR;

  h->opts.reduce = reduce;

  h->state = F4NCGB_STATE_SOLVING;

  int res;
  try {
    res = f4ncgb::f4ncgb_main(h->ctx,
                              h->opts,
                              { .output_name = h->output_file,
                                .userdata = userdata,
                                .add_cb = add_cb,
                                .end_poly_cb = end_cb });
  } catch(const std::exception& e) {
    msg("f4ncgb_solve failed: %s", e.what());
    h->state = F4NCGB_STATE_READY;
    return F4NCGB_ERROR;
  }

  h->state = F4NCGB_STATE_READY;

  switch(res) {
    case 0:
      return F4NCGB_OK;
    default:
      return F4NCGB_ERROR;
  }
}

extern "C" f4ncgb_result
f4ncgb_solve(f4ncgb_handle* h,
             void* userdata,
             f4ncgb_add_cb add_cb,
             f4ncgb_end_poly_cb end_cb) {
  return f4ncgb_solve_intern_(h, userdata, add_cb, end_cb, false);
}

extern "C" f4ncgb_result
f4ncgb_reduce(f4ncgb_handle* h,
              void* userdata,
              f4ncgb_add_cb add_cb,
              f4ncgb_end_poly_cb end_cb) {
  return f4ncgb_solve_intern_(h, userdata, add_cb, end_cb, true);
}
