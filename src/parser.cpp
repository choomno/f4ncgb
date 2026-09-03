#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "parser.hpp"
#include "zip_utils.hpp"

#define xstr(s) str(s)
#define str(s) #s

#define RETURN_ERROR(MSG)                                       \
  do {                                                          \
    std::source_location loc = std::source_location::current(); \
    return error_at_current(                                    \
      std::format("{}\n{}:{}:{}: location of last error",       \
                  MSG,                                          \
                  loc.file_name(),                              \
                  loc.line(),                                   \
                  loc.column()));                               \
  } while(false)

#define EXPECT(C)                                                 \
  do {                                                            \
    if(current_c() != C) {                                        \
      std::source_location loc = std::source_location::current(); \
      return error_at_current(                                    \
        std::format("{} '{}'\n{}:{}:{}: location of last error",  \
                    "Expected " str(C) " but received",           \
                    static_cast<char>(current_c()),               \
                    loc.file_name(),                              \
                    loc.line(),                                   \
                    loc.column()));                               \
    }                                                             \
  } while(false)

static inline bool
isdigit_(char c) {
  return c >= '0' && c <= '9';
}

#define EXPECT_NO_OVERFLOW()                          \
  do {                                                \
    if(number_missing_) {                             \
      number_missing_ = false;                        \
      RETURN_ERROR("Expected a digit");               \
    }                                                 \
    if(number_overflow_) {                            \
      number_overflow_ = false;                       \
      RETURN_ERROR("Number is too large to be read"); \
    }                                                 \
  } while(false)

#define EXPECT_DIGIT()                                            \
  do {                                                            \
    if(!isdigit_(current_c())) {                                  \
      std::source_location loc = std::source_location::current(); \
      return error_at_current(                                    \
        std::format("{} {}\n{}:{}:{}: location of last error",    \
                    "Expected a digit but received",              \
                    static_cast<char>(current_c()),               \
                    loc.file_name(),                              \
                    loc.line(),                                   \
                    loc.column()));                               \
    }                                                             \
  } while(false)

#define EXPECT_POSNEGDIGIT()                                      \
  do {                                                            \
    if(!isdigit_(current_c()) && c != '-' && c != '+') {          \
      std::source_location loc = std::source_location::current(); \
      return error_at_current(                                    \
        std::format("{} {}\n{}:{}:{}: location of last error",    \
                    "Expected a digit but received",              \
                    static_cast<char>(current_c()),               \
                    loc.file_name(),                              \
                    loc.line(),                                   \
                    loc.column()));                               \
    }                                                             \
  } while(false)

#define READ(C) \
  c = getc();   \
  EXPECT(C);

#define ADD(V)                                          \
  do {                                                  \
    if(parse_res add_err_ = add_cb(add_cb_userdata, V)) \
      return add_err_;                                  \
  } while(false)

#define BOUNDARY(N, D, R) boundary_cb(boundary_cb_userdata, N, D, R)

namespace f4ncgb {
parse_res
parser_context::open(std::filesystem::path p) {
  if(!std::filesystem::exists(p)) {
    return std::format("File \"{}\" does not exist!", p.string());
  }

  FILE* f_ptr = f4ncgb::open_compressed_or_direct(p);

  if(!f_ptr) {
    return std::format(
      "Could not open file \"{}\", error: {}", p.string(), strerror(errno));
  }
  f.reset(f_ptr);
  filename = p.string();
  return std::nullopt;
}

parse_res
parser_context::impl_msolve(parse_add_cb add_cb,
                            void* add_cb_userdata,
                            parse_monomial_boundary_cb boundary_cb,
                            void* boundary_cb_userdata) {
  int c = 0;
  // Read polynomials.

  // Each polynomial is a line ending with ',' except the last one. They are
  // made up of Monomials. Monomials always start with some coefficient
  // (either rational or integer), optionally followed by variables.

  while(true) {
    // Parse a polynomial.

    // Parse a monomial.
    do {

      // Parse the coefficient.
      bool rational = false;
      long numerator = 1;
      size_t denominator = 1;
      c = swallow_whitespace_no_newline();

      if(c == '-') {
        numerator = -1;
        c = getc();
      } else if(c == '+') {
        numerator = 1;
        c = getc();
      }

      c = swallow_whitespace_no_newline();

      if(isdigit_(c)) {
        EXPECT_POSNEGDIGIT();
        numerator = read_positive_or_negative_int() * numerator;
        EXPECT_NO_OVERFLOW();
        c = swallow_whitespace_no_newline();
        denominator = 1;
        if(c == '/') {
          c = getc();
          EXPECT_DIGIT();
          denominator = read_positive_int();
          EXPECT_NO_OVERFLOW();
          if(denominator == 0) {
            RETURN_ERROR("Denominator must not be 0!");
          }
          rational = true;
        }
        c = swallow_whitespace_no_newline();
      } else {
        // Trick the rest of the parser into thinking it read a coefficient and
        // a *, but it actually read nothing.
        ungetc(c);
        c = '*';
      }

      // Parse the variables
      while(c == '*') {
        c = getc();
        c = swallow_whitespace_no_newline();

        const std::string& ident
          = read_ident(parser_context::msolve_ident_filter);
        auto id = str_to_id(ident);

        c = swallow_whitespace_no_newline();

        if(c == '^') {
          getc();
          c = swallow_whitespace_no_newline();
          EXPECT_DIGIT();
          size_t sup = read_positive_int();
          EXPECT_NO_OVERFLOW();
          if(sup > std::numeric_limits<uint32_t>::max()) {
            return "Exponent too high";
          }
          c = swallow_whitespace_no_newline();
          for(size_t i = 0; i < sup; ++i) {
            ADD(id);
          }
        } else {
          ADD(id);
        }
      }

      // Submit the current monomial.
      BOUNDARY(numerator, static_cast<long>(denominator), rational);
    } while(c == '+' || c == '-');

    // Maybe another polynomial, or break.
    if(c == ',') {
      c = getc();
      if(c == '\r') {
        c = getc();
      }
      if(c == '\n') {
        c = getc();
      }

      // Boundary between polynomials.
      ADD(0);
    } else if(c == '\n' || c == '\r') {
      // Very last polynomial.
      ADD(0);
      return parse_msolve_end();
    } else {
      // Also very last polynomial with some weird ending.
      ADD(0);
      return parse_msolve_end();
    }
  }

  return std::nullopt;
}

parse_res
parser_context::parse_msolve_end() {
  while(c == '\r' || c == '\n') {
    if(c == '\r') {
      c = getc();
      READ('\n');
    } else if(c == '\n') {
      c = getc();
    }
  }

  if(c != EOF) {
    return std::string("Some characters left in input (did not end with EOF, "
                       "ended with '")
           + static_cast<char>(c) + "' in line " + std::to_string(line)
           + "). Check for trailing bytes or wrong format. Note: Check for "
             "',' as poly separator.";
  }

  return std::nullopt;
}

parse_res
parser_context::impl_msolve_header(char first_char, char second_char) {
  // Handle the first two characters first, then process the rest of the input.

  if(first_char == ',') {
    RETURN_ERROR("First character of msolve must be some ident, not ','");
  }
  if(parser_context::msolve_ident_filter(first_char)
     && (second_char == ',' || second_char == ' ' || second_char == '\n')) {
    // First char was some variable that has to be counted.
    std::string ident;
    ident += first_char;
    str_to_id(ident);
    ++opts_->num_vars;
  } else if(parser_context::msolve_ident_filter(first_char)
            && parser_context::msolve_ident_filter(second_char)) {
    c = getc();
    if(c == ',') {
      std::string ident;
      ident += first_char;
      ident += second_char;
      str_to_id(ident);
      // Some variable name was started and finished with the third char.
      ++opts_->num_vars;
    } else if(parser_context::msolve_ident_filter(c)) {
      // Some variable name was started but not finished yet, read until finish.
      std::string ident(read_ident(parser_context::msolve_ident_filter));
      std::string prefix;
      prefix += first_char;
      prefix += second_char;
      ident.insert(0, prefix);
      str_to_id(ident);
      ++opts_->num_vars;
    } else {
      RETURN_ERROR(
        std::format("Unexpected third character for msolve format: '{}'", c));
    }
  } else if(first_char == ' ' && second_char == ' ') {
    // Swallow up to the first new character.
    c = swallow_whitespace_and_newline();
  } else {
    RETURN_ERROR(std::format(
      "Unexpected start for msolve format: \"{}{}\"", first_char, second_char));
  }

  // Now, we have the first ident and are either at a ',' or at the end of the
  // variables with '\n'
  assert(opts_->num_vars == 1);
  if(c != '\n' && c != ',' && c != ' ') {
    RETURN_ERROR(std::format(
      "Unexpected msolve parse state after first variable, have c: '{}'", c));
  }

  opts_->blocks.emplace_back();
  opts_->blocks[0].emplace_back(1);

  auto* current_block = &opts_->blocks[0];

  while(true) {
    c = swallow_whitespace_no_newline();

    if(c == '\r') {
      c = getc();
    }

    // No more variables.
    if(c == '\n') {
      break;
    } else if(c == ',') {
      getc();
      swallow_whitespace_no_newline();
      if(c == '\n') {
        getc();
        opts_->blocks.emplace_back();
        current_block = &opts_->blocks[opts_->blocks.size() - 1];
        // Swallow as much as needed until there is something again.
        swallow_whitespace_and_newline();
      }
      auto id = str_to_id(read_ident(parser_context::msolve_ident_filter));
      current_block->emplace_back(id);
      ++opts_->num_vars;
    } else {
      RETURN_ERROR(std::format("Unexpected character '{}'", c));
    }
  }

  // Step over newline.
  if(c == '\r') {
    c = getc();
  }
  EXPECT('\n');
  c = getc();

  // Read the characteristic.
  EXPECT_DIGIT();
  opts_->characteristic = read_positive_int();
  EXPECT_NO_OVERFLOW();

  // Step over newline.
  if(c == '\r') {
    c = getc();
  }
  EXPECT('\n');
  c = getc();

  if(auto err = opts_->validate_blocks())
    RETURN_ERROR(*err);

  impl = [this](parse_add_cb add_cb,
                void* add_cb_userdata,
                parse_monomial_boundary_cb boundary_cb,
                void* boundary_cb_userdata) {
    return impl_msolve(
      add_cb, add_cb_userdata, boundary_cb, boundary_cb_userdata);
  };

  return std::nullopt;
}

template<parser_symbolic_context::id (*getV)(parser_context&)>
parse_res
parser_context::read_blocks() {
  size_t block = 0;
  while(block < opts_->num_blocks()) {
    if(current_c() == EOF)
      RETURN_ERROR("Input ended before every block was closed by a 0");
    if constexpr(getV == &parser_context::read_numeric_var) {
      EXPECT_DIGIT();
    }
    parser_symbolic_context::id v = getV(*this);
    if(v == 0) {
      ++block;
    } else {
      opts_->blocks[block].push_back(v);
    }
    swallow_whitespace_and_newline();
  }

  if(auto err = opts_->validate_blocks())
    RETURN_ERROR(*err);

  return std::nullopt;
}

template<parser_symbolic_context::id (*getV)(parser_context&)>
parse_res
parser_context::impl_poly_gen(parse_add_cb add_cb,
                              void* add_cb_userdata,
                              parse_monomial_boundary_cb boundary_cb,
                              void* boundary_cb_userdata) {

  while(c != EOF) {
    parser_symbolic_context::id v;

    // Polynomial
    do {
      // Monomial
      do {
        // Monomial Coefficient
        swallow_whitespace_and_newline();
        EXPECT_POSNEGDIGIT();
        long numerator = read_positive_or_negative_int();
        EXPECT_NO_OVERFLOW();
        swallow_whitespace_and_newline();
        EXPECT_DIGIT();
        long denominator = static_cast<long>(read_positive_int());
        EXPECT_NO_OVERFLOW();

        if(denominator == 0) {
          RETURN_ERROR("Denominator must not be 0!");
        }

        swallow_whitespace_and_newline();

        // Monomial Vars
        while(c != '+' && c != '0' && c != EOF) {
          if constexpr(getV == &parser_context::read_numeric_var) {
            EXPECT_DIGIT();
          }
          v = getV(*this);
          ADD(v);
          swallow_whitespace_and_newline();
        }

        BOUNDARY(numerator, denominator, numerator == denominator);

        // Prepare next monomial. If '0', nothing has to be done.
        if(c == '+') {
          getc();
          swallow_whitespace_and_newline();
        } else {
          break;
        }
      } while(c != EOF);

      // Prepare next polynomial.
      if(c == '0') {
        getc();
        swallow_whitespace_and_newline();
        ADD(0);
      } else {
        break;
      }
    } while(c != EOF);
  }

  return std::nullopt;
}

template parse_res
parser_context::impl_poly_gen<&parser_context::read_numeric_var>(
  parse_add_cb add_cb,
  void* add_cb_userdata,
  parse_monomial_boundary_cb boundary_cb,
  void* boundary_cb_userdata);

parse_res
parser_context::impl_poly(parse_add_cb add_cb,
                          void* add_cb_userdata,
                          parse_monomial_boundary_cb boundary_cb,
                          void* boundary_cb_userdata) {
  return impl_poly_gen<&parser_context::read_numeric_var>(
    add_cb, add_cb_userdata, boundary_cb, boundary_cb_userdata);
}

parse_res
parser_context::parse_header(solve_options& opts) {
  opts_ = &opts;
  // Parsing procedure:
  //
  // 1. Decide on format to use based on first few read characters.
  // 2. Call the respective implementation.
  // 3. Let the implementation call the callbacks.
  //
  // Return eventual errors as strings.

  getc();
  if(c != 'p') {
    init_symbols();
    char first = c;
    return impl_msolve_header(first, getc());
  }

  // File reads "p"
  c = getc();
  if(c != ' ') {
    init_symbols();
    return impl_msolve_header('p', c);
  }

  // File reads "p "

  using impl_fun_type
    = std::function<parse_res(parse_add_cb add_cb,
                              void* add_cb_userdata,
                              parse_monomial_boundary_cb boundary_cb,
                              void* boundary_cb_userdata)>;
  impl_fun_type impl_fun = nullptr;

  // Now the file must be one of our own new format types!
  getc();
  if(c == 'p') {
    // File should read "p poly"
    READ('o');
    READ('l');
    READ('y');
    READ(' ');

    impl_fun = [this](parse_add_cb add_cb,
                      void* add_cb_userdata,
                      parse_monomial_boundary_cb boundary_cb,
                      void* boundary_cb_userdata) {
      return impl_poly(
        add_cb, add_cb_userdata, boundary_cb, boundary_cb_userdata);
    };
  } else {
    return std::format("character {} is not a valid start for problem "
                       "specifier, must be p",
                       c);
  }

  getc();
  EXPECT_DIGIT();

  opts_->num_vars = read_positive_int();
  EXPECT_NO_OVERFLOW();
  EXPECT(' ');
  c = getc();
  EXPECT_DIGIT();
  const size_t num_blocks = read_positive_int();
  EXPECT_NO_OVERFLOW();
  if(num_blocks > 100)
    return "too many blocks";
  opts_->blocks.resize(num_blocks);
  EXPECT(' ');
  c = getc();
  EXPECT_DIGIT();
  opts_->characteristic = read_positive_int();
  EXPECT_NO_OVERFLOW();
  EXPECT('\n');
  getc();// Swallow the newline.

  if(auto err = read_blocks<&parser_context::read_numeric_var>())
    return err;

  impl = impl_fun;
  return std::nullopt;
}
}
