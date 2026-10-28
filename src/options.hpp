#pragma once

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "solve_state.hpp"

namespace f4ncgb {

// Fixed for the duration of a computation -- the part that changes while one
// runs is solve_state, in solve_state.hpp.
struct solve_options {
  using var_id = uint32_t;

  // 0 computes over the rationals, otherwise a prime below 2^31
  size_t characteristic = 0;

  // Variables are numbered 1..num_vars
  size_t num_vars = 0;

  // The variables of each block, smallest block first. A single block unless
  // the input asks for an elimination order.
  std::vector<std::vector<var_id>> blocks;

  // Weights of the variables. Empty means every weight is 1.
  std::vector<uint32_t> weights;

  // Ambiguities above this degree are dropped.
  size_t maxdeg = UINT_MAX;

  // Upper bound on F4 iterations.
  size_t maxiter = 10;

  size_t threads = 1;

  // 0 = off, 1 = cofactors, 2 = cofactors expanded in terms of the input.
  size_t proof_level = 0;
  std::string proof_file;
  // Optional caller-owned destination, used in preference to proof_file.
  std::ostream* proof_output = nullptr;

  // Remember which rows reduced to zero for the first prime and skip them for
  // the rest. Faster, and correct with high probability.
  bool tracer = true;

  // Compute the normal form of the last input polynomial instead of a basis.
  bool reduce = false;

  // Decides between the blocked reducer pass and the one-row-at-a-time
  // elimination. Only the test set it to compare the two against each other.
  reducer_pass la_pass = reducer_pass::calibrate;

  size_t num_blocks() const { return blocks.size(); }

  const std::vector<var_id>& block(size_t id) const {
    assert(id < blocks.size());
    return blocks[id];
  }

  std::vector<size_t> block_sizes() const {
    std::vector<size_t> res(blocks.size());
    for(size_t i = 0; i < blocks.size(); i++)
      res[i] = blocks[i].size();
    return res;
  }

  bool weighted() const { return !weights.empty(); }

  // unweighted deglex
  bool default_order() const { return num_blocks() <= 1 and !weighted(); }

  std::optional<std::string> validate_weights() const {
    if(weights.empty())
      return std::nullopt;

    if(weights.size() != num_vars)
      return std::format(
        "{} weights were given but the problem declares {} variables",
        weights.size(),
        num_vars);

    for(size_t i = 0; i < weights.size(); i++)
      if(weights[i] == 0 or weights[i] > 65535)
        return std::format("variable {} has weight {}: weights have to lie "
                           "in 1..65535",
                           i + 1,
                           weights[i]);

    return std::nullopt;
  }

  std::optional<std::string> validate_blocks() const {
    if(blocks.empty())
      return std::nullopt;

    var_id expected = 1;
    for(size_t i = 0; i < blocks.size(); i++)
      for(var_id v : blocks[i]) {
        if(v != expected)
          return std::format("block {} names variable {} where {} was "
                             "expected: blocks have to cover 1..{} in order",
                             i,
                             v,
                             expected,
                             num_vars);
        expected++;
      }

    if(expected - 1 != num_vars)
      return std::format(
        "the blocks name {} variables but the problem declares {}",
        expected - 1,
        num_vars);

    return std::nullopt;
  }
};
}
