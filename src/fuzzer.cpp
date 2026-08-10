#include <limits>
#include <numeric>
#include <optional>

#include "f4ncgb.hpp"
#include "fuzztest/fuzztest.h"
#include "fuzztest/internal/any.h"
#include "options.hpp"
#include "parser.hpp"
#include "solve.hpp"
#include "store.hpp"
#include <gtest/gtest.h>

using namespace f4ncgb;
using namespace fuzztest;

using block = std::vector<uint8_t>;
using monomial = std::tuple<long, long, std::vector<uint8_t>>;
using polynomial = std::vector<monomial>;

Domain<std::vector<std::vector<uint8_t>>>
UniqueAcrossVectors(uint8_t max_value = 254,
                    size_t max_vectors = F4NCGB_MAX_BLOCKS - 1) {
  auto number_of_values = InRange<uint8_t>(2, max_value);

  return Map(
    [max_vectors](uint8_t nr) -> std::vector<std::vector<uint8_t>> {
      std::vector<uint8_t> all_values(nr - 1);
      std::iota(all_values.begin(), all_values.end(), static_cast<uint8_t>(1));
      std::vector<std::vector<uint8_t>> result;

      assert(!all_values.empty());

      // Randomly determine how many sub-vectors there will be
      static std::random_device rd;
      static std::mt19937 gen(rd());
      std::uniform_int_distribution<> dist(
        1, std::min(max_vectors, all_values.size()));
      size_t num_sub_vectors = dist(gen);

      // Shuffle the values to ensure randomness
      // No shuffling, as this is unexpected by the later code!
      // std::shuffle(all_values.begin(), all_values.end(), gen);

      // Divide the all_values into num_sub_vectors parts
      size_t part_size = all_values.size() / num_sub_vectors;
      size_t index = 0;
      for(size_t i = 0; i < num_sub_vectors; ++i) {
        size_t additional = (i < all_values.size() % num_sub_vectors) ? 1 : 0;
        size_t current_size = part_size + additional;
        assert(current_size > 0);
        result.emplace_back(all_values.begin() + index,
                            all_values.begin() + index + current_size);
        index += current_size;
      }

      // Check generated vectors
      for(auto& v : result) {
        assert(v.size() > 0);
      }

      return result;
    },
    number_of_values);
}

size_t
block_size(const block& b) {
  return b.size();
}

size_t
blocks_to_numvars(const std::vector<block>& blocks) {
  return std::transform_reduce(
    blocks.begin(), blocks.end(), 0L, std::plus{}, block_size);
}

Domain<std::vector<uint8_t>>
AnyMonomial(uint8_t max_var) {
  return NonEmpty(VectorOf(InRange<uint8_t>(1, max_var)));
}

Domain<polynomial>
AnyPolynomial(size_t max_var) {
  return NonEmpty(
    VectorOf(TupleOf(OneOf(InRange<long>(-10, -1), InRange<long>(1, 11)),
                     InRange<long>(1, 11),
                     AnyMonomial(max_var))));
}

Domain<std::pair<std::vector<block>, std::vector<polynomial>>>
AnyProblem() {
  auto blocks_to_pair = [](const std::vector<block>& blocks)
    -> Domain<std::pair<std::vector<block>, std::vector<polynomial>>> {
    size_t num_vars = blocks_to_numvars(blocks);
    assert(num_vars < std::numeric_limits<uint8_t>::max());
    assert(num_vars >= 1);
    return PairOf(Just(blocks), NonEmpty(VectorOf(AnyPolynomial(num_vars))));
  };
  return FlatMap(blocks_to_pair, UniqueAcrossVectors());
}

class fuzz_parser : public parser_context {
  const std::vector<polynomial>& polys;

  public:
  /// The generated problem, in the form the solver expects it.
  solve_options opts;

  fuzz_parser(
    const std::pair<std::vector<::block>, std::vector<polynomial>>& problem)
    : parser_context()
    , polys(problem.second) {
    auto& [blocks, polys] = problem;
    opts.characteristic = 0;
    opts.num_vars = blocks_to_numvars(blocks);
    opts.blocks.resize(blocks.size());
    assert(polys.size() == this->polys.size());
    assert(polys.size() > 0);
    for(size_t i = 0; i < blocks.size(); ++i) {
      opts.blocks[i].resize(blocks[i].size());
      std::transform(blocks[i].begin(),
                     blocks[i].end(),
                     opts.blocks[i].begin(),
                     [](uint8_t e) -> uint32_t { return e; });
    }

    this->impl = [this](parse_add_cb add_cb,
                        void* add_cb_userdata,
                        parse_monomial_boundary_cb boundary_cb,
                        void* boundary_cb_userdata) -> parse_res {
      for(auto& p : this->polys) {
        assert(p.size() > 0);
        for(auto& m : p) {
          auto& [num, denom, mon] = m;
          assert(mon.size() > 0);
          for(auto v : mon) {
            assert(v > 0);
            add_cb(add_cb_userdata, v);
          }
          boundary_cb(boundary_cb_userdata,
                      num,
                      denom,
                      num == 1 && denom == 1 || denom == 1);
        }
        add_cb(add_cb_userdata, 0);
      }
      return std::nullopt;
    };
  }
};

void
fuzz_f4ncgb_main(
  const std::pair<std::vector<block>, std::vector<polynomial>>& problem,
  size_t maxdeg,
  size_t maxiter) {
  size_t maxvar = blocks_to_numvars(problem.first);

  for(auto& p : problem.second) {
    assert(p.size() > 0);

    for(auto& m : p) {
      for(auto v : std::get<2>(m)) {
        assert(v > 0);
        assert(v <= maxvar);
      }
    }
  }

  fuzz_parser ctx(problem);
  ctx.opts.maxiter = maxiter;
  ctx.opts.maxdeg = maxdeg;

  int res = f4ncgb_main(ctx, ctx.opts, run_options{});
}

FUZZ_TEST(f4ncgb, fuzz_f4ncgb_main)
  .WithDomains(AnyProblem(), InRange(8, 1000), InRange(4, 10));

#ifdef __linux__
void
fuzz_parser(std::string input) {
  int add_cb_count = 0;
  int boundary_cb_count = 0;

  auto add_cb = [&add_cb_count](uint32_t i) {
    (void)i;
    if(add_cb_count++ > 1000000)
      throw std::runtime_error("too many ADDs");
    return std::nullopt;
  };
  auto boundary_cb
    = [&boundary_cb_count](long numerator, long denominator, bool is_rational) {
        (void)numerator;
        (void)denominator;
        (void)is_rational;
        if(boundary_cb_count++ > 1000000)
          throw std::runtime_error("too many BOUNDARYs");
        return std::nullopt;
      };

  FILE* f = fmemopen((void*)input.c_str(), input.size(), "r");

  if(!f) {
    std::cout << "Could not make a memory-backed file. Error: "
              << strerror(errno) << std::endl;
    exit(1);
  }

  parser_context ctx(f);

  try {
    parse(ctx, add_cb, boundary_cb);
  } catch(std::runtime_error& e) {
    EXPECT_LT(add_cb_count, 1000000);
    EXPECT_LT(boundary_cb_count, 1000000);
    FAIL() << e.what();
  } catch(...) {
    FAIL() << "Some error";
  }
}

FUZZ_TEST(f4ncgb, fuzz_parser);
#endif
