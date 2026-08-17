#pragma once

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <vector>

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

  // Ambiguities above this degree are dropped.
  size_t maxdeg = UINT_MAX;

  // Upper bound on F4 iterations.
  size_t maxiter = 10;

  size_t threads = 1;

  // 0 = off, 1 = cofactors, 2 = cofactors expanded in terms of the input.
  size_t proof_level = 0;
  std::string proof_file;

  // Remember which rows reduced to zero for the first prime and skip them for
  // the rest. Faster, and correct with high probability.
  bool tracer = true;

  // Compute the normal form of the last input polynomial instead of a basis.
  bool reduce = false;

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
