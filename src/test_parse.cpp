#include <array>
#include <boost/test/unit_test_log.hpp>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <boost/test/unit_test.hpp>

#include "options.hpp"
#include "output.hpp"
#include "parser.hpp"
#include "polynomial_store.hpp"
#include "solve.hpp"

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

  // A one-variable header ends immediately after its first character. CRLF
  // must follow the same path as LF rather than being mistaken for an identifier.
  solve_options single_lf_options, single_crlf_options;
  const auto single_lf = parse_msolve_text(
    "x\n7\nx*x - 1\n", &single_lf_options);
  const auto single_crlf = parse_msolve_text(
    "x\r\n7\r\nx*x - 1\r\n", &single_crlf_options);
  BOOST_REQUIRE_MESSAGE(!single_lf.has_value(), single_lf.value_or(""));
  BOOST_REQUIRE_MESSAGE(!single_crlf.has_value(), single_crlf.value_or(""));
  const std::vector<std::vector<solve_options::var_id>> single_block{ { 1 } };
  BOOST_TEST(single_lf_options.num_vars == 1u);
  BOOST_TEST(single_crlf_options.num_vars == single_lf_options.num_vars);
  BOOST_TEST((single_lf_options.blocks == single_block));
  BOOST_TEST((single_crlf_options.blocks == single_lf_options.blocks));
  BOOST_TEST(single_lf_options.characteristic == 7u);
  BOOST_TEST(single_crlf_options.characteristic == single_lf_options.characteristic);

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
  const std::array<std::pair<std::string, std::string>, 9> bad = { {
    { "x\r7\r\nx*x - 1\r\n", "Unexpected character" },
    { "x\r\n7\r\nx*x - 1\r", "Expected" },
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
    BOOST_CHECK_MESSAGE(err->find("input.ms:") != std::string::npos,
                        "parser error lacks its input location: " << *err);
    BOOST_CHECK_MESSAGE(err->find(expected) != std::string::npos,
                        "expected '" << expected << "', got: " << *err);
  }
}

namespace {
struct parsed_input {
  parse_res error;
  solve_options options;
  std::string polynomials;
};

parsed_input
parse_context(parser_context& context) {
  parsed_input result;
  internal::monomial_store<> monomials;
  internal::polynomial_store<> polynomials(monomials);
  result.error = context.parse_header(result.options);
  if(!result.error) {
    std::vector<coeff> denominators;
    result.error = parse_rest_into_polynomial_store(
      context, polynomials, denominators, result.options.num_vars);
    if(!result.error) {
      std::ostringstream text;
      write_all(text, context, result.options, polynomials);
      result.polynomials = text.str();
    }
  }
  return result;
}

class parser_character_probe : public parser_context {
  public:
  using parser_context::getc;
  using parser_context::ungetc;
};

class failing_output_buffer : public std::streambuf {
  int_type overflow(int_type) override { return traits_type::eof(); }
};
}

BOOST_AUTO_TEST_CASE(memory_input_matches_file_parser_for_all_fixtures) {
  auto fixtures = get_parse_tests_location();
  BOOST_REQUIRE(fixtures.has_value());
  for(const auto& entry : std::filesystem::directory_iterator(*fixtures)) {
    if(entry.path().extension() != ".ms" && entry.path().extension() != ".poly")
      continue;
    std::ifstream source(entry.path(), std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(source)), {});
    parser_context file, memory;
    BOOST_REQUIRE(!file.open(entry.path()));
    BOOST_REQUIRE(!memory.open_text(bytes, entry.path().string()));
    const auto expected = parse_context(file);
    const auto actual = parse_context(memory);
    BOOST_CHECK_MESSAGE(actual.error == expected.error, entry.path());
    BOOST_TEST(actual.polynomials == expected.polynomials);
    BOOST_TEST(actual.options.num_vars == expected.options.num_vars);
    BOOST_TEST(actual.options.characteristic == expected.options.characteristic);
    BOOST_TEST((actual.options.blocks == expected.options.blocks));
    BOOST_TEST((actual.options.weights == expected.options.weights));
  }
}

BOOST_AUTO_TEST_CASE(memory_input_matches_file_line_endings_and_diagnostics) {
  temporary_parse_directory directory;
  const auto path = directory.path() / "input.ms";
  const std::array<std::string, 8> inputs{ {
    "x\n0\nx*x - 1\n",
    "x\r\n7\r\nx*x - 1\r\n",
    "x,y\n0\nx*y - y*x,\nx*x - 1\n",
    "x,y\r\n0\r\nx*y - y*x,\r\nx*x - 1\r\n",
    "x,t\n0\nx*t^0 - 1\n",
    "x,y\n0\nx*y + -1*y*x\n",
    "x,y\n0\nx*t^0 - 1\n",
    "x\n0\nx*"
  } };
  // Reuse both contexts to exercise switching/resetting input state.
  parser_context file, memory;
  for(const auto& input : inputs) {
    { std::ofstream output(path, std::ios::binary); output << input; }
    BOOST_REQUIRE(!file.open(path));
    BOOST_REQUIRE(!memory.open_text(input, path.string()));
    const auto expected = parse_context(file);
    const auto actual = parse_context(memory);
    BOOST_CHECK(actual.error == expected.error);
    BOOST_TEST(actual.polynomials == expected.polynomials);
  }
  BOOST_REQUIRE(!memory.open(path));
  BOOST_REQUIRE(!file.open(path));
  BOOST_CHECK(parse_context(memory).error == parse_context(file).error);
}

BOOST_AUTO_TEST_CASE(memory_character_input_preserves_bytes_pushback_and_eof) {
  temporary_parse_directory directory;
  const auto path = directory.path() / "bytes";
  const std::string bytes{ '\0', static_cast<char>(255), 'x', '\n' };
  {
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  parser_character_probe file, memory;
  BOOST_REQUIRE(!file.open(path));
  BOOST_REQUIRE(!memory.open_text(bytes));
  for(unsigned n = 0; n < bytes.size() + 2; ++n)
    BOOST_TEST(memory.getc() == file.getc());
  memory.ungetc(EOF);
  file.ungetc(EOF);
  BOOST_TEST(memory.getc() == file.getc());
  memory.ungetc('y');
  file.ungetc('y');
  BOOST_TEST(memory.getc() == file.getc());
  BOOST_TEST(memory.getc() == file.getc());
}

BOOST_AUTO_TEST_CASE(memory_solver_outputs_match_basis_and_proof_files) {
  f4ncgb_set_msg_printing(false);
  temporary_parse_directory directory;
  const auto input = directory.path() / "input.ms";
  const auto basis_path = directory.path() / "basis.txt";
  const auto proof_path = directory.path() / "proof.txt";
  for(const unsigned characteristic : { 0u, 7u }) {
    const std::string text = "x,y\n" + std::to_string(characteristic)
                             + "\n2*x*x-x,\n3*y*y-y,\nx*y-y*x\n";
    { std::ofstream output(input, std::ios::binary); output << text; }
    for(const unsigned proof_level : { 1u, 2u }) {
      parser_context file, memory;
      solve_options file_options, memory_options;
      BOOST_REQUIRE(!file.open(input));
      BOOST_REQUIRE(!memory.open_text(text));
      BOOST_REQUIRE(!file.parse_header(file_options));
      BOOST_REQUIRE(!memory.parse_header(memory_options));
      std::ostringstream basis_stream, proof_stream;
      file_options.maxiter = memory_options.maxiter = 64;
      file_options.tracer = memory_options.tracer = false;
      file_options.proof_level = memory_options.proof_level = proof_level;
      file_options.proof_file = proof_path.string();
      memory_options.proof_output = &proof_stream;
      const auto unused_basis = directory.path() / "unused-basis.txt";
      const auto unused_proof = directory.path() / "unused-proof.txt";
      memory_options.proof_file = unused_proof.string();
      run_options file_run, memory_run;
      file_run.output_name = basis_path.string();
      memory_run.output_stream = &basis_stream;
      memory_run.output_name = unused_basis.string();
      BOOST_REQUIRE(f4ncgb_main(file, file_options, file_run) == 0);
      BOOST_REQUIRE(f4ncgb_main(memory, memory_options, memory_run) == 0);
      std::ifstream basis_file(basis_path), proof_file(proof_path);
      const std::string basis((std::istreambuf_iterator<char>(basis_file)), {});
      const std::string proof((std::istreambuf_iterator<char>(proof_file)), {});
      BOOST_REQUIRE(!basis.empty());
      BOOST_REQUIRE(!proof.empty());
      BOOST_TEST(basis_stream.str() == basis);
      BOOST_TEST(proof_stream.str() == proof);
      BOOST_TEST(!std::filesystem::exists(unused_basis));
      BOOST_TEST(!std::filesystem::exists(unused_proof));
    }
  }
}

BOOST_AUTO_TEST_CASE(memory_solver_rejects_failed_output_streams) {
  auto make_context = [](parser_context& context, solve_options& options) {
    BOOST_REQUIRE(!context.open_text("x\n7\nx*x-x\n"));
    BOOST_REQUIRE(!context.parse_header(options));
    options.tracer = false;
  };
  {
    parser_context context;
    solve_options options;
    make_context(context, options);
    std::ostringstream failed;
    failed.setstate(std::ios::badbit);
    options.proof_level = 1;
    options.proof_output = &failed;
    run_options run;
    BOOST_CHECK_THROW(f4ncgb_main(context, options, run), std::runtime_error);
  }
  {
    parser_context context;
    solve_options options;
    make_context(context, options);
    failing_output_buffer buffer;
    std::ostream failed(&buffer);
    options.proof_level = 1;
    options.proof_output = &failed;
    std::ostringstream basis;
    run_options run;
    run.output_stream = &basis;
    BOOST_CHECK_THROW(f4ncgb_main(context, options, run), std::runtime_error);
    BOOST_TEST(basis.str().empty());
  }
  {
    parser_context context;
    solve_options options;
    make_context(context, options);
    std::ostringstream failed;
    failed.setstate(std::ios::badbit);
    run_options run;
    run.output_stream = &failed;
    std::function<void()> statistics;
    run.stats_print_function = &statistics;
    BOOST_CHECK_THROW(f4ncgb_main(context, options, run), std::runtime_error);
    BOOST_REQUIRE(static_cast<bool>(statistics));
    BOOST_CHECK_NO_THROW(statistics());
  }
}
