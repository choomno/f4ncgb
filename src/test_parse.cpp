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
