#pragma once

#include "coeff.hpp"
#include "options.hpp"

#include <cstdio>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>

#include <boost/bimap.hpp>
#include <boost/bimap/unordered_set_of.hpp>

namespace f4ncgb {
using parse_res = std::optional<std::string>;
using parse_add_cb = std::add_pointer<parse_res(void*, uint32_t)>::type;

using parse_monomial_boundary_cb
  = std::add_pointer<parse_res(void*, long, long, bool)>::type;

class parser_symbolic_context {
  public:
  using id = uint32_t;

  private:
  using map = boost::bimap<boost::bimaps::unordered_set_of<std::string>,
                           boost::bimaps::unordered_set_of<id>>;
  map m;
  id counter = 1;

  public:
  parser_symbolic_context() = default;
  ~parser_symbolic_context() = default;

  inline std::string_view id_to_str(id i) const { return m.right.at(i); }
  inline id str_to_id(const std::string& s) {
    auto it = m.left.find(s);
    if(it != m.left.end()) {
      return it->second;
    }
    m.insert(map::value_type(s, counter));
    // std::cout << "Insert " << s << " at " << counter << std::endl;
    return counter++;
  }
};

class parser_context {
  protected:
  std::unique_ptr<parser_symbolic_context> symbols;
  struct FILE_deleter {
    void operator()(FILE* f) { fclose(f); }
  };
  std::unique_ptr<FILE, FILE_deleter> f;

  size_t last_line = 0;
  size_t last_col = 0;
  size_t line = 1;
  size_t col = 1;
  std::string filename = "";
  std::string ident = "";
  int c = 0;

  parse_res impl_msolve_header(char first_char, char second_char);
  parse_res impl_msolve(parse_add_cb add_cb,
                        void* add_cb_userdata,
                        parse_monomial_boundary_cb boundary_cb,
                        void* boundary_cb_userdata);

  template<parser_symbolic_context::id (*getV)(parser_context&)>
  parse_res read_blocks();

  template<parser_symbolic_context::id (*getV)(parser_context&)>
  parse_res impl_poly_gen(parse_add_cb add_cb,
                          void* add_cb_userdata,
                          parse_monomial_boundary_cb boundary_cb,
                          void* boundary_cb_userdata);

  parse_res impl_poly(parse_add_cb add_cb,
                      void* add_cb_userdata,
                      parse_monomial_boundary_cb boundary_cb,
                      void* boundary_cb_userdata);

  inline int current_c() const { return c; }

  inline int getc() {
    assert(f);
    if(!f || feof(f.get())) {
      this->c = EOF;
      return EOF;
    }
    last_line = line;
    last_col = col;

    int c = fgetc(f.get());

    // Debug aid:
    // std::printf("Read '%c'\n", c);

    this->c = c;
    if(c == '\n') {
      ++line;
      col = 1;
    } else {
      ++col;
    }
    return c;
  }

  inline void ungetc(int c) { ::ungetc(c, f.get()); }

  // Set when a number in the input did not fit
  bool number_overflow_ = false;

  bool number_missing_ = false;

  inline size_t read_digits(size_t limit) {
    if(c < '0' || c > '9') {
      number_missing_ = true;
      return 0;
    }
    size_t num = 0;
    do {
      const size_t digit = static_cast<size_t>(c) - '0';
      if(num > (limit - digit) / 10)
        number_overflow_ = true;
      else
        num = num * 10 + digit;
      getc();
    } while(c >= '0' && c <= '9');
    return num;
  }

  inline size_t read_positive_int() {
    return read_digits(std::numeric_limits<size_t>::max());
  }

  inline long read_positive_or_negative_int() {
    assert((c >= '0' && c <= '9') || c == '-' || c == '+');
    bool neg = false;
    if(c == '-') {
      getc();
      neg = true;
    } else if(c == '+') {
      getc();
      neg = false;
    }
    // -LONG_MIN has no counterpart, so the magnitude stops at LONG_MAX
    const long num = static_cast<long>(
      read_digits(static_cast<size_t>(std::numeric_limits<long>::max())));
    return neg ? -num : num;
  }

  inline int swallow_whitespace_no_newline() {
    while(c == ' ' || c == '\t') {
      getc();
    }
    return c;
  }
  inline int swallow_whitespace_and_newline() {
    while(c == ' ' || c == '\t' || c == '\n' || c == '\r') {
      getc();
    }
    return c;
  }

  static inline bool msolve_ident_filter(char c) {
    return c != '-' && c != '/' && c != '*' && c != '+' && c != ',' && c != ' '
           && c != '\n' && c != '\r' && c != '\t' && c != '^';
  }
  template<typename Filter>
  inline const std::string& read_ident(Filter c_allowed) {
    ident = "";
    do {
      ident.push_back(c);
      getc();
    } while(c_allowed(c) && c != EOF);
    return ident;
  }

  std::string error_at_current(std::string_view msg) {
    if(filename != "") {
      return std::format(
        "{}:{}:{}: error: {}", filename, last_line, last_col, msg);
    } else {
      return std::format("{}:{}: error: {}", last_line, last_col, msg);
    }
  }

  void init_symbols() { symbols = std::make_unique<parser_symbolic_context>(); }

  static parser_symbolic_context::id read_numeric_var(parser_context& ctx) {
    return ctx.read_positive_int();
  }

  parse_res parse_msolve_end();

  public:
  parser_context(FILE* in)
    : f(in) {}
  parser_context() = default;
  ~parser_context() = default;

  solve_options* opts_ = nullptr;

  std::function<parse_res(parse_add_cb add_cb,
                          void* add_cb_userdata,
                          parse_monomial_boundary_cb boundary_cb,
                          void* boundary_cb_userdata)>
    impl;

  parse_res open(std::filesystem::path p);

  inline std::string_view id_to_str(parser_symbolic_context::id i) const {
    return symbols->id_to_str(i);
  }
  inline parser_symbolic_context::id str_to_id(const std::string& s) {
    return symbols->str_to_id(s);
  }

  bool has_symbols() const { return static_cast<bool>(symbols); }

  parse_res parse_header(solve_options& opts);

  const std::string& get_filename() const { return filename; }

  std::ostream& var_to_ostream(std::ostream& o, parser_symbolic_context::id i) {
    if(has_symbols())
      return o << id_to_str(i);
    else
      return o << static_cast<int>(i);
  }
};

template<typename AddCB, typename BoundaryCB>
parse_res
parse_rest(parser_context& ctx, AddCB add_cb, BoundaryCB boundary_cb) {
  parse_add_cb add_cb_impl = [](void* userdata, uint32_t i) -> parse_res {
    AddCB* cb = reinterpret_cast<AddCB*>(userdata);
    return (*cb)(i);
  };
  parse_monomial_boundary_cb boundary_cb_impl
    = [](void* userdata,
         long numerator,
         long denominator,
         bool is_rational) -> parse_res {
    BoundaryCB* cb = reinterpret_cast<BoundaryCB*>(userdata);
    return (*cb)(numerator, denominator, is_rational);
  };

  return ctx.impl(add_cb_impl, &add_cb, boundary_cb_impl, &boundary_cb);
}

template<typename AddCB, typename BoundaryCB>
parse_res
parse(parser_context& ctx,
      solve_options& opts,
      AddCB add_cb,
      BoundaryCB boundary_cb) {
  parse_res err = ctx.parse_header(opts);
  if(err)
    return err;
  return parse_rest(ctx, add_cb, boundary_cb);
}

// Every route into the stores runs through here -- the three input formats and
// the C API
template<class PS>
parse_res
parse_rest_into_polynomial_store(parser_context& ctx,
                                 PS& s,
                                 std::vector<coeff>& input_denoms,
                                 size_t num_vars) {

  using monomial_store = PS::monomial_store_;
  using monomial_ref = PS::value_type;
  using var = PS::monomial;
  std::vector<var> tmp_monomial;
  std::vector<std::pair<coeff, monomial_ref>> tmp_polynomial;
  std::vector<long> tmp_denom;

  auto add_cb
    = [&s, &input_denoms, &tmp_monomial, &tmp_polynomial, &tmp_denom, num_vars](
        uint32_t id) -> parse_res {
    if(id) {
      if(id > num_vars)
        return std::format("variable index {} is out of range, the problem "
                           "declares {} variables",
                           id,
                           num_vars);
      tmp_monomial.push_back(static_cast<var>(id));
    } else {

      coeff common_denom = 1;
      for(long d : tmp_denom)
        common_denom.lcm_inplace(d);

      input_denoms.push_back(common_denom);

      size_t i = 0;
      for(auto& [c, m] : tmp_polynomial) {
        c *= common_denom;
        c /= tmp_denom[i++];
      }

      s.sort_polynomial(tmp_polynomial);
      s.add_polynomial(tmp_polynomial);
      tmp_polynomial.clear();
      tmp_denom.clear();
    }
    return std::nullopt;
  };

  auto boundary_cb = [&s, &tmp_monomial, &tmp_polynomial, &tmp_denom](
                       long numerator, long denominator, bool is_rational) {
    (void)is_rational;

    tmp_denom.push_back(denominator);

    monomial_store& m = s.get_monomial_store();
    auto monomial_id = m.getid(tmp_monomial);
    tmp_polynomial.push_back(std::make_pair(coeff(numerator), monomial_id));

    tmp_monomial.clear();
    return std::nullopt;
  };

  return parse_rest(ctx, add_cb, boundary_cb);
}

}
