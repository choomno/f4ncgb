#pragma once

#include <cstddef>
#include <vector>

#include "coeff.hpp"

namespace f4ncgb {

// The way linear_algebra.cpp reduces the s-polynomials against the
// reducers. `calibrate` times a sample of the matrix both ways and keeps the
// one that won; the other two fix the choice (needed for the testing)
enum class reducer_pass { calibrate, blocked, row_at_a_time };

// The part of a single computation's state that reaches beyond the f4 object:
// the parser fills in the input denominators, the linear algebra reads the
// proof and normalization settings.
struct solve_state {
  // Denominator cleared out of each input polynomial. In input order until
  // sort_rows() permutes it to match the sorted s-polynomial rows.
  std::vector<coeff> input_denoms;

  // Coefficient the result is normalized by in reduce_mode.
  coeff reduce_denom_c = 1;

  // 0 = off, 1 = cofactors, 2 = cofactors expanded in terms of the input.
  size_t proof_level = 0;

  // Number of monomial columns of the current matrix, which is also where the
  // transformation block of the proof matrix starts.
  size_t nr_cols = 0;

  // The current matrix interreduces the input instead of reducing
  // s-polynomials.
  bool interreduce = false;

  // Computing a normal form instead of a basis.
  bool reduce_mode = false;

  // Copied from solve_options
  reducer_pass la_pass = reducer_pass::calibrate;
};
}
