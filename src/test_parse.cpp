#include <array>
#include <boost/test/unit_test_log.hpp>
#include <filesystem>
#include <string>

#include <boost/test/unit_test.hpp>

#include "options.hpp"
#include "parser.hpp"
#include "polynomial_store.hpp"

using namespace f4ncgb;

std::optional<std::filesystem::path>
get_parse_tests_location() {
  std::array<std::string, 2> candidates = { "parse_tests/", "../parse_tests/" };

  auto curr = std::filesystem::current_path();

  for(auto& c : candidates) {
    auto path = curr / std::filesystem::path(c);
    if(std::filesystem::exists(path)) {
      return path;
    }
  }
  return std::nullopt;
}

// These tests only test the parser code itself, not the mechanics underneath.
// They are fed using some input files in the parse_tests subdirectory.

static std::optional<std::string>
invoke_parser(parser_context& ctx,
              internal::polynomial_store<>& ps,
              std::filesystem::path p) {
  if(auto err = ctx.open(p); err) {
    return err;
  }
  solve_options opts;
  std::optional<std::string> err = ctx.parse_header(opts);
  if(err)
    return err;
  std::vector<coeff> input_denoms;
  return parse_rest_into_polynomial_store(
    ctx, ps, input_denoms, opts.num_vars);
}

BOOST_AUTO_TEST_CASE(parse_files) {
  auto dir_optional = get_parse_tests_location();
  BOOST_REQUIRE(dir_optional.has_value());
  std::filesystem::path dir = *dir_optional;

  for(auto& e : std::filesystem::directory_iterator(dir)) {
    if(e.path().extension() != ".ms" && e.path().extension() != ".poly") {
      // Only ms and poly files should be tested. Others may
      // be temporaries.
      BOOST_TEST_MESSAGE("Skipping file " << e << " because of extension "
                                          << e.path().extension());
      continue;
    }
    BOOST_REQUIRE(e.is_regular_file() || e.is_symlink());
    BOOST_TEST_MESSAGE("Parsing file " << e);
    parser_context ctx;
    internal::monomial_store<> ms;
    internal::polynomial_store<> ps(ms);
    if(e.path().filename().string().starts_with("fail_")) {
      // Expect some parse error.
      auto ret = invoke_parser(ctx, ps, e);
      BOOST_CHECK_MESSAGE(ret.has_value(),
                          "parsing the input "
                            << e.path().filename()
                            << " did not lead to an error as expected!");
    } else {
      // Expect the parsing to be successful.
      auto ret = invoke_parser(ctx, ps, e);
      if(ret) {
        BOOST_CHECK_MESSAGE(!ret.has_value(),
                            "parsing the input " << e.path().filename()
                                                 << " did lead to an error:\n"
                                                 << *ret);
      }
    }
  }
}

// parse_files only sees that a file parses; these check what the optional `w`
// line actually put into the options.
static solve_options
parse_header_of(const std::string& name) {
  auto dir_optional = get_parse_tests_location();
  BOOST_REQUIRE(dir_optional.has_value());

  parser_context ctx;
  BOOST_REQUIRE(!ctx.open(*dir_optional / name).has_value());

  solve_options opts;
  auto err = ctx.parse_header(opts);
  BOOST_REQUIRE_MESSAGE(!err.has_value(), err.value_or(""));
  return opts;
}

BOOST_AUTO_TEST_CASE(parse_weights) {
  const std::vector<uint32_t> w123{ 1, 2, 3 };
  const solve_options ms = parse_header_of("weighted.ms");
  BOOST_TEST(ms.num_vars == 3u);
  BOOST_TEST(ms.num_blocks() == 1u);
  BOOST_TEST(ms.weighted());
  BOOST_TEST(!ms.default_order());
  BOOST_TEST((ms.weights == w123));

  // several blocks, extra spacing, and a variable that is itself called w
  const std::vector<uint32_t> w4112{ 4, 1, 1, 2 };
  const solve_options blocks = parse_header_of("weighted_blocks.ms");
  BOOST_TEST(blocks.num_vars == 4u);
  BOOST_TEST(blocks.num_blocks() == 2u);
  BOOST_TEST((blocks.weights == w4112));

  const std::vector<uint32_t> w511{ 5, 1, 1 };
  const solve_options poly = parse_header_of("weighted.poly");
  BOOST_TEST(poly.num_vars == 3u);
  BOOST_TEST((poly.weights == w511));

  // no w line leaves the weights empty and the fast path selected
  const solve_options plain = parse_header_of("braid3_test.poly");
  BOOST_TEST(!plain.weighted());
  BOOST_TEST(plain.weights.empty());
}
