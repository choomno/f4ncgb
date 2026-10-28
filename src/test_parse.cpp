#include <array>
#include <boost/test/unit_test_log.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

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
              std::filesystem::path p,
              solve_options* parsed_options = nullptr) {
  if(auto err = ctx.open(p); err) {
    return err;
  }
  solve_options opts;
  std::optional<std::string> err = ctx.parse_header(opts);
  if(err)
    return err;
  if(parsed_options)
    *parsed_options = opts;
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

// Each invocation owns a directory created atomically, so concurrent test
// processes neither overwrite one another's inputs nor remove one another's files.
class temporary_parse_directory {
  std::filesystem::path path_;

  public:
  temporary_parse_directory() {
    const auto root = std::filesystem::temp_directory_path();
    std::random_device random;
    for(unsigned attempt = 0; attempt < 64; ++attempt) {
      auto candidate = root / ("f4ncgb-parse-" + std::to_string(random())
                                + "-" + std::to_string(random()));
      std::error_code error;
      if(std::filesystem::create_directory(candidate, error)) {
        path_ = std::move(candidate);
        return;
      }
      if(error && error != std::errc::file_exists)
        throw std::filesystem::filesystem_error(
          "Cannot create parser test directory", candidate, error);
    }
    throw std::runtime_error("Cannot allocate a unique parser test directory");
  }

  temporary_parse_directory(const temporary_parse_directory&) = delete;
  temporary_parse_directory& operator=(const temporary_parse_directory&) = delete;

  ~temporary_parse_directory() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  const std::filesystem::path& path() const { return path_; }
};

// Inputs whose exact bytes matter (CRLF line endings) are written in binary
// mode instead of living in parse_tests, where editors or Git may normalize them.
static std::optional<std::string>
parse_msolve_text(const std::string& text,
                 solve_options* parsed_options = nullptr) {
  temporary_parse_directory directory;
  const auto path = directory.path() / "input.ms";
  {
    std::ofstream out(path, std::ios::binary);
    BOOST_REQUIRE_MESSAGE(out.is_open(), "Cannot open parser test input " << path);
    out << text;
    out.close();
    BOOST_REQUIRE_MESSAGE(out.good(), "Cannot write parser test input " << path);
  }
  parser_context ctx;
  internal::monomial_store<> ms;
  internal::polynomial_store<> ps(ms);
  return invoke_parser(ctx, ps, path, parsed_options);
}

BOOST_AUTO_TEST_CASE(parse_msolve_line_endings_and_variable_names) {
  // A last line ending in "\r\n" must parse like the "\n" version.
  const auto crlf = parse_msolve_text(
    "x,y\r\n0\r\nx*y - y*x,\r\nx*x - 1\r\n");
  BOOST_CHECK_MESSAGE(!crlf.has_value(), crlf.value_or(""));

  // Both line endings must preserve the variable count and the same two
  // elimination blocks, not merely accept the header without an error.
  solve_options lf_options, crlf_options;
  const auto lf_blocks = parse_msolve_text(
    "x,y,\nz,w\n0\nx*z - z*x\n", &lf_options);
  const auto crlf_blocks = parse_msolve_text(
    "x,y,\r\nz,w\r\n0\r\nx*z - z*x\r\n", &crlf_options);
  BOOST_REQUIRE_MESSAGE(!lf_blocks.has_value(), lf_blocks.value_or(""));
  BOOST_REQUIRE_MESSAGE(!crlf_blocks.has_value(), crlf_blocks.value_or(""));
  const std::vector<std::vector<solve_options::var_id>> expected_blocks{
    { 1, 2 }, { 3, 4 }
  };
  BOOST_TEST(lf_options.num_vars == 4u);
  BOOST_TEST(crlf_options.num_vars == lf_options.num_vars);
  BOOST_TEST(lf_options.num_blocks() == 2u);
  BOOST_TEST((lf_options.blocks == expected_blocks));
  BOOST_TEST((crlf_options.blocks == lf_options.blocks));
  BOOST_TEST(crlf_options.characteristic == lf_options.characteristic);
  BOOST_TEST((crlf_options.weights == lf_options.weights));

  const auto declared_zero_power = parse_msolve_text("x,t\n0\nx*t^0 - 1\n");
  BOOST_CHECK_MESSAGE(!declared_zero_power.has_value(),
                      declared_zero_power.value_or(""));

  // A missing or undeclared variable name is reported as such. A zero
  // exponent must not let an undeclared name escape validation.
  const std::array<std::pair<std::string, std::string>, 7> bad = { {
    { "x,\r\n0\r\nx*x\r\n", "Expected a digit" },
    { "x,y\n0\nx*y - y*x,\nx*x - 1,\n", "Expected a variable name" },
    { "x,y\n0\nx*y + -1*y*x\n", "Expected a variable name" },
    { "x,y\n0\nx*-y\n", "Expected a variable name" },
    { "x,y\n0\nx*y - y*x,\n\nx*x - 1\n", "Expected a variable name" },
    { "x,y\n0\nx*z - 1\n", "Undeclared variable 'z'" },
    { "x,y\n0\nx*t^0 - 1\n", "Undeclared variable 't'" },
  } };
  for(const auto& [text, expected] : bad) {
    const auto err = parse_msolve_text(text);
    BOOST_REQUIRE_MESSAGE(err.has_value(), "no error for input:\n" << text);
    BOOST_CHECK_MESSAGE(err->find(expected) != std::string::npos,
                        "expected '" << expected << "', got: " << *err);
  }
}
