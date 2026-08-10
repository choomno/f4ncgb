#pragma once

#include <cstddef>
#include <vector>

#include "coeff.hpp"

namespace f4ncgb {

/// The part of a single computation's state that reaches beyond the f4 object:
/// the parser fills in the input denominators, the linear algebra reads the
/// proof and normalization settings.
///
/// These used to be file-scope globals, which meant two concurrent
/// f4ncgb_solve() calls silently corrupted each other. The f4 object owns one
/// of these and hands it to everyone who needs it.
struct solve_state {
  /// Denominator cleared out of each input polynomial. In input order until
  /// sort_rows() permutes it to match the sorted s-polynomial rows.
  std::vector<coeff> input_denoms;

  /// Coefficient the result is normalized by in reduce_mode.
  coeff reduce_denom_c = 1;

  /// 0 = off, 1 = cofactors, 2 = cofactors expanded in terms of the input.
  size_t proof_level = 0;

  /// Number of monomial columns of the current matrix, which is also where the
  /// transformation block of the proof matrix starts.
  size_t nr_cols = 0;

  /// The current matrix interreduces the input instead of reducing
  /// s-polynomials.
  bool interreduce = false;

  /// Computing a normal form instead of a basis.
  bool reduce_mode = false;
};
}
