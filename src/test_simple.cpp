#include <boost/test/unit_test.hpp>

#include <fstream>
#include <sstream>
#include <thread>

#include "coeff.hpp"
#include "cofactors.hpp"
#include "f4.hpp"
#include "f4ncgb.hpp"
#include "options.hpp"
#include "parser.hpp"
#include "polynomial_store.hpp"
#include "solve.hpp"
#include "store.hpp"

using namespace f4ncgb;

std::optional<std::filesystem::path>
get_test_input_files_location() {
  std::array<std::string, 2> candidates = { "test_inputs/", "../test_inputs/" };

  auto curr = std::filesystem::current_path();

  for(auto& c : candidates) {
    auto path = curr / std::filesystem::path(c);
    if(std::filesystem::exists(path)) {
      return path;
    }
  }
  return std::nullopt;
}

template<internal::metadata_concept MM = internal::metadata<uint8_t>,
         internal::metadata_concept PM = internal::metadata<uint8_t>,
         internal::value_concept V = uint8_t,
         typename I = uint32_t,
         typename C = coeff>
struct impl {
  using var = V;
  using idx = I;
  using coefficient = C;
  using monomial_store = internal::monomial_store<MM, V, I>;
  using polynomial_store = internal::polynomial_store<PM, MM, V, I, C>;
};

BOOST_AUTO_TEST_CASE(std_span_behavior) {
  std::vector<uint8_t> v{ 1, 2, 3, 1, 2, 3 };
  std::span<uint8_t> s1{ v.begin(), 3 };
  std::span<uint8_t> s2{ v.begin() + 3, 3 };

  BOOST_TEST(std::equal(s1.begin(), s1.end(), s2.begin(), s2.end()));
}

BOOST_AUTO_TEST_CASE(simple_store) {
  using I = impl<>;

  I::monomial_store s;
  auto m1_idx_1 = s.getid({ 1, 2, 3 });
  auto m1_idx_2 = s.getid({ 1, 2, 3 });

  BOOST_TEST(m1_idx_1 == m1_idx_2);
  BOOST_TEST(m1_idx_1 > 0);

  auto m2_idx_1 = s.getid({ 1, 2, 3, 3 });

  BOOST_TEST(m2_idx_1 != m1_idx_1);

  auto m1_idx_3 = s.getid({ 1, 2, 3 });

  BOOST_TEST(m1_idx_1 == m1_idx_3);
}

BOOST_AUTO_TEST_CASE(simple_product) {
  using I = impl<>;

  I::monomial_store s;

  I::idx m1_idx = s.getid({ 1 });
  I::idx m23_idx = s.getid({ 2, 3 });
  I::idx m123_idx = s.getid({ 1, 2, 3 });
  I::idx m123123_idx = s.getid({ 1, 2, 3, 1, 2, 3 });

  I::idx prod_idx = s.get_product_id(m1_idx, m23_idx);

  BOOST_TEST(m123_idx == prod_idx);

  I::idx prod2_idx = s.get_product_id(m123_idx, m123_idx);
  BOOST_TEST(m123123_idx == prod2_idx);
}

BOOST_AUTO_TEST_CASE(simple_cmp) {
  using I = impl<>;

  I::monomial_store s;

  I::idx m123_idx = s.getid({ 1, 2, 3 });
  I::idx m123_idx2 = s.getid({ 1, 2, 3 });
  I::idx m132_idx = s.getid({ 1, 3, 2 });
  I::idx m123123_idx = s.getid({ 1, 2, 3, 1, 2, 3 });

  BOOST_TEST(!s.template cmp<false>(m123_idx, m123_idx2));
  BOOST_TEST(s.template cmp<false>(m123_idx, m132_idx));
  BOOST_TEST(!s.template cmp<false>(m123123_idx, m123_idx));
}

BOOST_AUTO_TEST_CASE(simple_polynomials) {
  using I = impl<>;
  I::monomial_store s;
  I::polynomial_store p(s);

  I::idx p1 = p.add_polynomial({ { 1l, { 2, 3 } }, { 2l, { 2, 3 } } });
  I::idx p2 = p.add_polynomial({ { 3l, { 3, 4 } }, { 4l, { 5, 6 } } });

  BOOST_TEST(p1 == 1);
  BOOST_TEST(p2 != p1);

  BOOST_TEST(p.get_coefficients(p1)[0] == I::coefficient(1l));
  BOOST_TEST(p.get_coefficients(p1)[1] == I::coefficient(2l));
  BOOST_TEST(p.get_coefficients(p2)[0] == I::coefficient(3l));
  BOOST_TEST(p.get_coefficients(p2)[1] == I::coefficient(4l));

  // An absent factor is the empty monomial, id 0
  I::idx p3 = p.multiply_front_and_back(0, p2, s.getid({ 1, 2 }));

  BOOST_TEST(p.get_monomial_id(p3, 0) == s.getid({ 3, 4, 1, 2 }));

  // A factor on both sides, which is the only shape the algorithm itself uses
  I::idx f = s.getid({ 1, 2 });
  I::idx b = s.getid({ 4 });
  I::idx p4 = p.multiply_front_and_back(f, p2, b);

  BOOST_TEST(p.get_monomial_id(p4, 0) == s.getid({ 1, 2, 3, 4, 4 }));
  BOOST_TEST(p.get_monomial_id(p4, 1) == s.getid({ 1, 2, 5, 6, 4 }));

  // Handing the leading monomial in instead of interning it has to give the
  // same polynomial back, tail included -- that form skips the first product,
  // so it is the rest of the loop that has to stay in step.
  I::idx p5 = p.multiply_front_and_back(f, p2, b, s.getid({ 1, 2, 3, 4, 4 }));

  BOOST_TEST(
    std::equal(p[p4].begin(), p[p4].end(), p[p5].begin(), p[p5].end()));
}

template<size_t N>
struct metadata_with_tuple {
  uint8_t length = 0;
  uint16_t t[N];
};

template<size_t N>
void
run(/* some input parameters */) {
  using I
    = impl<metadata_with_tuple<N>, metadata_with_tuple<1>, uint8_t, uint16_t>;

  // Do things with impl.
  typename I::monomial_store s;

  s.getid({ 1, 2, 3 });
}

BOOST_AUTO_TEST_CASE(select_impl_from_number) {
  size_t n = 1;

  // Generate a switch over n with at most 60 variables. The implementation is
  // duplicated internally for each N.
  //
  // The implementation could also be a more complex struct or something else.
  // The run function is only a demonstration.
  boost::mp11::mp_with_index<60>(n, [](auto N) { run<N>(); });
}

BOOST_AUTO_TEST_CASE(simple_with_extended_metadata) {
  struct metadata_with_extra {
    uint8_t length = 0;
    uint16_t m = 1;
  };
  static_assert(sizeof(metadata_with_extra) == 4);
  static_assert(alignof(metadata_with_extra) == 2);

  using I = impl<metadata_with_extra, metadata_with_extra, uint8_t, uint16_t>;
  I::monomial_store s;
  I::idx m1_idx = s.getid({ 1 });
  metadata_with_extra& m1_metadata = s.get_metadata(m1_idx);

  BOOST_TEST(m1_metadata.m == 1);

  I::idx m23_idx = s.getid({ 2, 3 });
  metadata_with_extra& m23_metadata = s.get_metadata(m23_idx);

  BOOST_TEST(m23_metadata.m == 1);

  metadata_with_extra& m1_metadata_ = s.get_metadata(m1_idx);

  BOOST_TEST(m1_metadata.m == 1);
  BOOST_TEST(m1_metadata_.m == 1);
  BOOST_TEST(&m1_metadata == &m1_metadata_);

  I::idx prod_id = s.get_product_id(m1_idx, m23_idx);
  metadata_with_extra& prod_metadata = s.get_metadata(prod_id);
  BOOST_TEST(prod_metadata.m == 1);
  I::idx prod_id_expected = s.getid({ 1, 2, 3 });

  BOOST_TEST(prod_id == prod_id_expected);

  I::idx long_id = s.getid({ 1, 1, 3, 3 });
  metadata_with_extra& long_metadata = s.get_metadata(long_id);
  BOOST_TEST(long_metadata.m == 1);

  std::vector<uint8_t> v(s[long_id].begin(), s[long_id].end());
  BOOST_TEST(v[0] == 1);
  BOOST_TEST(v[1] == 1);
  BOOST_TEST(v[2] == 3);
  BOOST_TEST(v[3] == 3);
}

BOOST_AUTO_TEST_CASE(parse_small_ms_into_polynomial) {
  auto dir_optional = get_test_input_files_location();
  BOOST_REQUIRE(dir_optional.has_value());
  std::filesystem::path dir = *dir_optional;

  std::filesystem::path input = dir / "small.ms";

  using I = impl<>;
  I::monomial_store ms;
  I::polynomial_store ps(ms);

  parser_context ctx;
  solve_options opts;
  ctx.open(input);
  parse_res r = ctx.parse_header(opts);
  BOOST_REQUIRE(!r.has_value());

  BOOST_CHECK(opts.characteristic == 0);
  BOOST_CHECK(opts.num_vars == 2);
  BOOST_CHECK(opts.num_blocks() == 1);
  const auto& b0 = opts.block(0);
  BOOST_CHECK(b0[0] == 1);
  BOOST_CHECK(b0[1] == 2);

  std::vector<coeff> input_denoms;
  r = parse_rest_into_polynomial_store(ctx, ps, input_denoms, opts.num_vars);
  BOOST_REQUIRE(!r.has_value());

  std::vector<I::idx> polys(ps.begin(), ps.end());
  BOOST_REQUIRE(polys[0] == 0);
  BOOST_REQUIRE(polys[1] == 1);
  // The only place the id encoding is pinned down. An id counts alignof(M)-
  // sized slots from the start of the pool, plus one for the reserved 0: the
  // first polynomial has two terms, so it occupies
  // sizeof(polynomial_metadata) + 2 * sizeof(I) = 20 bytes = 5 slots.
  BOOST_REQUIRE(polys[2] == 6);
  std::vector<I::idx> poly1(ps[polys[1]].begin(), ps[polys[1]].end());

  BOOST_TEST(poly1[0] == 3);
}

BOOST_AUTO_TEST_CASE(parse_braid3) {
  auto dir_optional = get_test_input_files_location();
  BOOST_REQUIRE(dir_optional.has_value());
  std::filesystem::path dir = *dir_optional;

  std::filesystem::path input = dir / "braid3-16.ms";

  using I = impl<>;
  I::monomial_store ms;
  I::polynomial_store ps(ms);

  parser_context ctx;
  solve_options opts;
  ctx.open(input);
  parse_res r = ctx.parse_header(opts);
  BOOST_REQUIRE(!r.has_value());

  std::vector<coeff> input_denoms;
  r = parse_rest_into_polynomial_store(ctx, ps, input_denoms, opts.num_vars);
  BOOST_REQUIRE(!r.has_value());

  BOOST_CHECK(opts.characteristic == 0);
  BOOST_CHECK(opts.num_vars == 3);
  BOOST_CHECK(opts.num_blocks() == 1);
}

using solver_result = std::pair<f4ncgb_result, std::vector<Solver::polynomial>>;

static solver_result
solve_example() {
  Solver s;
  s.set_blocks({ 3 });

  s.add(2, 1, { 3, 2, 3 });
  s.add(-3, 1, { 2, 1, 2 });
  s.end_poly();

  s.add(4, 1, { 3, 1, 2 });
  s.add(5, 1, { 1, 2, 1 });
  s.end_poly();

  s.add(6, 1, { 3, 1, 3 });
  s.add(-7, 1, { 2, 3, 1 });
  s.end_poly();

  s.add(1, 1, { 3, 3, 3 });
  s.add(1, 1, { 2, 2, 2 });
  s.add(1, 1, { 1, 2, 3 });
  s.add(1, 1, { 1, 1, 1 });
  s.end_poly();

  s.set_maxdeg(4);
  s.set_maxiter(2);

  return s.solve();
}

static solver_result
reduce_example() {
  Solver s;
  s.set_nvars(2);

  s.add(1, 1, { 2, 1 });
  s.add(-1, 3, { 1, 2 });
  s.end_poly();

  s.add(-3, 1, { 1, 2 });
  s.add(29, 16, { 1 });
  s.end_poly();

  s.add(7, 8, { 2, 1 });
  s.add(1, 1, { 1, 2 });
  s.add(-4, 1, {});
  s.end_poly();

  return s.reduce();
}

BOOST_AUTO_TEST_CASE(c_api_use) {
  f4ncgb_set_msg_printing(false);

  Solver check_state;
  BOOST_CHECK(check_state.state() == F4NCGB_STATE_READY);

  auto [res, polies] = solve_example();

  // Expected number of polys is 5.
  BOOST_CHECK(res == F4NCGB_OK);
  BOOST_CHECK(polies.size() == 5);

  // for(auto& p : polies) {
  //   for(auto& m : p) {
  //     std::cout << "[" << std::get<0>(m) << "/" << std::get<1>(m) << "] ";
  //     for(auto v : std::get<2>(m)) {
  //       std::cout << v << " ";
  //     }
  //     std::cout << "+ ";
  //   }
  //   std::cout << "0" << std::endl;
  // }
}

BOOST_AUTO_TEST_CASE(max_blocks_boundary) {
  f4ncgb_set_msg_printing(false);

  // Exactly F4NCGB_MAX_BLOCKS blocks.
  Solver s;
  s.set_blocks(std::vector<uint32_t>(F4NCGB_MAX_BLOCKS, 1));
  s.set_maxiter(2);
  s.set_maxdeg(4);

  // x1*x2 - x3 and x2*x1 - x4, in an order where every variable is its own
  // block
  s.add(1, 1, { 1, 2 });
  s.add(-1, 1, { 3 });
  s.end_poly();
  s.add(1, 1, { 2, 1 });
  s.add(-1, 1, { 4 });
  s.end_poly();

  auto [res, polies] = s.solve();
  BOOST_CHECK(res == F4NCGB_OK);
  BOOST_CHECK(polies.size() == 2);

  // one block more stays refused, and by the setter rather than by a crash
  Solver too_many;
  BOOST_CHECK_THROW(
    too_many.set_blocks(std::vector<uint32_t>(F4NCGB_MAX_BLOCKS + 1, 1)),
    std::runtime_error);
}

namespace {

// The braid3 relations on the top three variables of an `nvars` variable
// problem, serialised with those three numbered from the top so that runs with
// different `nvars` compare.
std::string
solve_on_top_vars(uint32_t nvars, uint32_t characteristic) {
  const uint32_t base = nvars - 3;
  const uint32_t x = base + 1, y = base + 2, z = base + 3;

  Solver s;
  s.set_nvars(nvars);
  s.set_characteristic(characteristic);
  s.set_maxiter(4);

  auto poly = [&s](std::vector<std::pair<long, std::vector<uint32_t>>> terms) {
    for(const auto& [c, m] : terms)
      s.add(c, 1, m);
    s.end_poly();
  };

  poly({ { 1, { z, y, z } }, { -1, { y, x, y } } });
  poly({ { 1, { z, x, y } }, { -1, { x, y, x } } });
  poly({ { 1, { z, x, z } }, { -1, { y, z, x } } });
  poly({ { 1, { z, z, z } },
         { 1, { y, y, y } },
         { 1, { x, x, x } },
         { 1, { x, y, z } } });

  auto [res, polies] = s.solve();
  if(res != F4NCGB_OK)
    return "solve failed";

  std::ostringstream o;
  for(const auto& p : polies) {
    for(const auto& [num, den, mon] : p) {
      o << num << "/" << den << ":";
      for(uint32_t v : mon)
        o << " " << (v - base);
      o << " ";
    }
    o << "\n";
  }
  return o.str();
}

}

BOOST_AUTO_TEST_CASE(uint16_variables_match_uint8) {
  f4ncgb_set_msg_printing(false);

  // 254 variables still fit uint8_t, 255 and above switch V to uint16_t
  for(uint32_t characteristic : { 0u, 32003u }) {
    const std::string want = solve_on_top_vars(254, characteristic);
    BOOST_REQUIRE(!want.empty());
    BOOST_REQUIRE(want != "solve failed");

    BOOST_TEST(solve_on_top_vars(255, characteristic) == want);
    BOOST_TEST(solve_on_top_vars(300, characteristic) == want);
  }
}

//------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(reduce_without_reducers) {
  f4ncgb_set_msg_printing(false);

  Solver s;
  s.set_nvars(2);

  // x1*x2 - x2*x1, the only polynomial, so there is nothing to reduce it by
  s.add(1, 1, { 1, 2 });
  s.add(-1, 1, { 2, 1 });
  s.end_poly();

  auto [res, polies] = s.reduce();

  BOOST_CHECK(res == F4NCGB_OK);
  BOOST_REQUIRE(polies.size() == 1);
  BOOST_REQUIRE(polies[0].size() == 2);

  // the input comes back, leading monomial first
  rat_coeff rc;
  fmpq_t correct;
  fmpq_init(correct);

  auto& [n, d, m] = polies[0][0];
  rc.update(n, d);
  fmpq_set_si(correct, -1, 1);
  BOOST_CHECK(fmpq_equal(rc.value, correct));
  BOOST_CHECK((m == std::vector<uint32_t>{ 2, 1 }));

  auto& [n2, d2, m2] = polies[0][1];
  rc.update(n2, d2);
  fmpq_set_si(correct, 1, 1);
  BOOST_CHECK(fmpq_equal(rc.value, correct));
  BOOST_CHECK((m2 == std::vector<uint32_t>{ 1, 2 }));

  fmpq_clear(correct);
}

BOOST_AUTO_TEST_CASE(reduce_path) {
  f4ncgb_set_msg_printing(false);

  auto [res, polies] = reduce_example();

  // Expected number of polys is 1.
  BOOST_CHECK(res == F4NCGB_OK);
  BOOST_CHECK(polies.size() == 1);

  rat_coeff rc;
  fmpq_t correct;
  fmpq_init(correct);

  auto& [n, d, m] = polies[0][0];
  rc.update(n, d);
  fmpq_set_si(correct, 899, 1152);
  BOOST_CHECK(fmpq_equal(rc.value, correct));

  auto& [n2, d2, m2] = polies[0][1];
  rc.update(n2, d2);
  fmpq_set_si(correct, -4, 1);
  BOOST_CHECK(fmpq_equal(rc.value, correct));

  // for(auto& p : polies) {
  //   for(auto& m : p) {
  //     std::cout << "[" << std::get<0>(m) << "/" << std::get<1>(m) << "] ";
  //     for(auto v : std::get<2>(m)) {
  //       std::cout << v << " ";
  //     }
  //     std::cout << "+ ";
  //   }
  //   std::cout << "0" << std::endl;
  // }
}

static bool
same(const solver_result& a, const solver_result& b) {
  if(a.first != b.first or a.second.size() != b.second.size())
    return false;
  for(size_t i = 0; i < a.second.size(); i++) {
    if(a.second[i].size() != b.second[i].size())
      return false;
    for(size_t j = 0; j < a.second[i].size(); j++) {
      const auto& [an, ad, am] = a.second[i][j];
      const auto& [bn, bd, bm] = b.second[i][j];
      if(an != bn or ad != bd or am != bm)
        return false;
    }
  }
  return true;
}

// Concurrent solves used to share their state
// through file-scope globals, which this catches.
BOOST_AUTO_TEST_CASE(concurrent_solves) {
  f4ncgb_set_msg_printing(false);

  const solver_result expected_solve = solve_example();
  const solver_result expected_reduce = reduce_example();

  for(int round = 0; round < 8; round++) {
    solver_result got_solve;
    solver_result got_reduce;

    std::thread a([&got_solve]() { got_solve = solve_example(); });
    std::thread b([&got_reduce]() { got_reduce = reduce_example(); });
    a.join();
    b.join();

    BOOST_CHECK(same(got_solve, expected_solve));
    BOOST_CHECK(same(got_reduce, expected_reduce));
  }
}

//------------------------------------------------------------------------------
// The blocked reducer pass (linear_algebra_internal.hpp) and the row-at-a-time
// one (linear_algebra.cpp) have to produce the same basis.

namespace {

std::string
solve_ug2(reducer_pass pass, size_t threads) {
  auto dir = get_test_input_files_location();
  BOOST_REQUIRE(dir.has_value());

  parser_context ctx;
  solve_options opts;
  ctx.open(*dir / "ug2-x1x2x3x4-5.ms");
  parse_res r = ctx.parse_header(opts);
  BOOST_REQUIRE(!r.has_value());

  opts.maxdeg = 5;
  opts.maxiter = 1000;
  opts.la_pass = pass;
  opts.threads = threads;

  const auto out = std::filesystem::temp_directory_path()
                   / ("f4ncgb_pass_" + std::to_string(static_cast<int>(pass))
                      + "_" + std::to_string(threads) + ".out");

  run_options run;
  run.output_name = out.string();
  BOOST_REQUIRE(f4ncgb_main(ctx, opts, run) == 0);

  std::ifstream f(out);
  std::string res((std::istreambuf_iterator<char>(f)),
                  std::istreambuf_iterator<char>());
  std::filesystem::remove(out);
  return res;
}

}

BOOST_AUTO_TEST_CASE(blocked_reduction_matches_row_at_a_time) {
  f4ncgb_set_msg_printing(false);

  const std::string want = solve_ug2(reducer_pass::row_at_a_time, 1);
  BOOST_REQUIRE(!want.empty());

  // 31783 columns in the last matrix, so several blocks to carry reducers
  // between and get wrong
  BOOST_TEST(solve_ug2(reducer_pass::blocked, 1) == want);
  BOOST_TEST(solve_ug2(reducer_pass::blocked, 4) == want);

  // and the default, which times a sample of each matrix both ways and may
  // hand the rest of it back to the row-at-a-time pass
  BOOST_TEST(solve_ug2(reducer_pass::calibrate, 1) == want);
  BOOST_TEST(solve_ug2(reducer_pass::calibrate, 4) == want);
}

//------------------------------------------------------------------------------
// Composition of cofactor transformations (cofactors.hpp).

namespace {

using test_f4 = f4ncgb::f4<1, uint8_t>;
using test_cofactor = test_f4::cofactor;
using test_mons = test_f4::monomial_store;

// A term of an expanded row, in a form that is easy to compare.
struct term {
  uint32_t a;
  int i;
  uint32_t b;
  long c;

  bool operator==(const term& o) const {
    return a == o.a and i == o.i and b == o.b and c == o.c;
  }
};

std::vector<term>
sorted(std::vector<term> ts) {
  std::sort(ts.begin(), ts.end(), [](const term& x, const term& y) {
    if(x.i != y.i)
      return x.i < y.i;
    if(x.a != y.a)
      return x.a < y.a;
    return x.b < y.b;
  });
  return ts;
}

std::vector<term>
sorted_terms(const std::vector<test_cofactor>& row) {
  std::vector<term> res;
  for(const auto& t : row)
    res.push_back({ t.a(), t.i(), t.b(), fmpz_get_si(t.c.value) });
  return sorted(std::move(res));
}

constexpr int
in(int i) {
  return -i - 1;
}

}

// g4 = g3 - x*g1 where g3 = x*g1 + g2 expands to f1: the x*f0 terms have to
// meet in the collection and cancel
BOOST_AUTO_TEST_CASE(cofactor_composition_cancels) {
  test_mons mons;
  mons.set_blocks({ 2 });
  const uint32_t X = mons.getid({ 1 });

  // index 0 is the free slot, windows are [1,3), [3,4), [4,5)
  std::vector<std::vector<test_cofactor>> rows(5);
  std::vector<coeff> denoms(5, coeff(1L));

  rows[1].emplace_back(coeff(1L), 0u, 0, 0u);// g1 = f0
  rows[2].emplace_back(coeff(1L), 0u, 1, 0u);// g2 = f1

  rows[3].emplace_back(coeff(1L), X, 1, 0u);// g3 = x*g1 + g2
  rows[3].emplace_back(coeff(1L), 0u, 2, 0u);

  rows[4].emplace_back(coeff(1L), 0u, 3, 0u);// g4 = g3 - x*g1
  rows[4].emplace_back(coeff(-1L), X, 1, 0u);

  std::vector<size_t> window_start{ 1, 3, 4, 5 };
  f4ncgb::internal::cofactor_expander<uint32_t, coeff, test_cofactor, test_mons>
    expander(mons, rows, denoms);
  expander.expand_all(window_start);

  // g3 = x*f0 + f1
  const std::vector<term> want3{ { X, in(0), 0, 1 }, { 0, in(1), 0, 1 } };
  BOOST_TEST((sorted_terms(rows[3]) == sorted(want3)));
  BOOST_TEST(denoms[3].is_one());

  // g4 = f1, with the two x*f0 terms gone
  const std::vector<term> want4{ { 0, in(1), 0, 1 } };
  BOOST_TEST((sorted_terms(rows[4]) == sorted(want4)));
  BOOST_TEST(denoms[4].is_one());
}

BOOST_AUTO_TEST_CASE(cofactor_composition_skips_windows) {
  test_mons mons;
  mons.set_blocks({ 2 });
  const uint32_t Y = mons.getid({ 2 });

  // windows are [1,3), [3,4), [4,5), [5,6)
  std::vector<std::vector<test_cofactor>> rows(6);
  std::vector<coeff> denoms(6, coeff(1L));

  rows[1].emplace_back(coeff(1L), 0u, 0, 0u);// g1 = f0
  rows[2].emplace_back(coeff(1L), 0u, 1, 0u);// g2 = f1

  rows[3].emplace_back(coeff(1L), 0u, 1, 0u);// g3 = g1 + g2
  rows[3].emplace_back(coeff(1L), 0u, 2, 0u);

  rows[4].emplace_back(coeff(1L), Y, 3, 0u);// g4 = y*g3

  rows[5].emplace_back(coeff(1L), 0u, 4, 0u);// g5 = g4 - y*g1, and the g1 term
  rows[5].emplace_back(coeff(-1L), Y, 1, 0u);// reaches back two windows

  std::vector<size_t> window_start{ 1, 3, 4, 5, 6 };
  f4ncgb::internal::cofactor_expander<uint32_t, coeff, test_cofactor, test_mons>
    expander(mons, rows, denoms);
  expander.expand_all(window_start);

  // g4 = y*f0 + y*f1
  const std::vector<term> want4{ { Y, in(0), 0, 1 }, { Y, in(1), 0, 1 } };
  BOOST_TEST((sorted_terms(rows[4]) == sorted(want4)));

  // g5 = y*(f0 + f1) - y*f0 = y*f1, so the y*f0 terms cancel only after the
  // skipped window has been substituted in
  const std::vector<term> want5{ { Y, in(1), 0, 1 } };
  BOOST_TEST((sorted_terms(rows[5]) == sorted(want5)));
  BOOST_TEST(denoms[5].is_one());
}

BOOST_AUTO_TEST_CASE(cofactor_composition_denominators) {
  test_mons mons;
  mons.set_blocks({ 2 });
  const uint32_t X = mons.getid({ 1 });
  const uint32_t Y = mons.getid({ 2 });

  std::vector<std::vector<test_cofactor>> rows(5);
  std::vector<coeff> denoms(5, coeff(1L));

  rows[1].emplace_back(coeff(1L), 0u, 0, 0u);// g1 = f0
  rows[2].emplace_back(coeff(1L), 0u, 1, 0u);// g2 = f1

  // g3 = (x*g1 + g2) / 2
  rows[3].emplace_back(coeff(1L), X, 1, 0u);
  rows[3].emplace_back(coeff(1L), 0u, 2, 0u);
  denoms[3] = coeff(2L);

  // g4 = g3 + x*g1 = (3*x*f0 + f1) / 2
  rows[4].emplace_back(coeff(1L), 0u, 3, 0u);
  rows[4].emplace_back(coeff(1L), X, 1, 0u);

  std::vector<size_t> window_start{ 1, 3, 4, 5 };
  f4ncgb::internal::cofactor_expander<uint32_t, coeff, test_cofactor, test_mons>
    expander(mons, rows, denoms);
  expander.expand_all(window_start);

  const std::vector<term> want{ { X, in(0), 0, 3 }, { 0, in(1), 0, 1 } };
  BOOST_TEST((sorted_terms(rows[4]) == sorted(want)));
  BOOST_TEST((denoms[4] == coeff(2L)));

  // the monomials on both sides have to compose in the right order
  BOOST_TEST(mons.get_product_id(X, Y) != mons.get_product_id(Y, X));
}

//------------------------------------------------------------------------------

namespace {

// Deterministic, so a failure is reproducible.
struct word_source {
  uint64_t state = 0;

  uint64_t next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return state >> 33;
  }

  std::vector<uint8_t> word(size_t vars, size_t len) {
    std::vector<uint8_t> w(len);
    for(size_t i = 0; i < len; i++)
      w[i] = static_cast<uint8_t>(1 + next() % vars);
    return w;
  }
};

}

BOOST_AUTO_TEST_CASE(batched_products_match_term_at_a_time) {
  using I = impl<>;
  constexpr size_t vars = 6;

  I::monomial_store scalar, batched;

  word_source src{ 4242 };
  for(int row = 0; row < 40; row++) {
    const auto a = src.word(vars, src.next() % 4);
    const auto c = src.word(vars, src.next() % 4);
    // rows on both sides of the chunk the store walks a row in
    const size_t len = 1 + src.next() % 80;

    std::vector<uint32_t> ms_scalar, ms_batched;
    for(size_t i = 0; i < len; i++) {
      const auto m = src.word(vars, 1 + src.next() % 12);
      ms_scalar.push_back(scalar.getid(m));
      ms_batched.push_back(batched.getid(m));
    }
    BOOST_REQUIRE(ms_scalar == ms_batched);

    const uint32_t ia = scalar.getid(a), ic = scalar.getid(c);
    const uint32_t ja = batched.getid(a), jc = batched.getid(c);

    std::vector<uint32_t> one(len), all(len);
    for(size_t i = 0; i < len; i++)
      one[i] = scalar.get_product_id(scalar[ia], ms_scalar[i], scalar[ic]);
    batched.get_product_ids(batched[ja], ms_batched, batched[jc], all.data());

    BOOST_TEST(one == all);

    // and the words behind those ids really are the concatenation
    for(size_t i = 0; i < len; i++) {
      std::vector<uint8_t> w(a);
      const auto mv = batched[ms_batched[i]];
      w.insert(w.end(), mv.begin(), mv.end());
      w.insert(w.end(), c.begin(), c.end());
      const auto got = batched[all[i]];
      BOOST_TEST(std::equal(w.begin(), w.end(), got.begin(), got.end()));
    }
  }
}

// -----------------------------------------------------------------------------
// The open-addressed tables lose entries rather than crashing if a rehash goes
// wrong, so growth past several doublings is worth testing directly.

BOOST_AUTO_TEST_CASE(probe_map_survives_growth) {
  internal::probe_map m;
  m.reset(16);

  // well past several growth points, so the rehash path is exercised
  constexpr uint32_t n = 300000;
  for(uint32_t i = 0; i < n; i++) {
    auto [slot, inserted] = m.find_or_insert(i, i * 7);
    BOOST_REQUIRE(inserted);
    BOOST_REQUIRE(*slot == i * 7);
  }
  BOOST_TEST(m.size() == n);

  // nothing was lost or duplicated by a rehash
  for(uint32_t i = 0; i < n; i++) {
    auto [slot, inserted] = m.find_or_insert(i, 0);
    BOOST_REQUIRE(!inserted);
    BOOST_REQUIRE(*slot == i * 7);
  }
  BOOST_TEST(m.size() == n);

  // and reset really empties it, whether or not the buffer is reused
  m.reset(n);
  BOOST_TEST(m.size() == 0u);
  auto [slot, inserted] = m.find_or_insert(5, 99);
  BOOST_TEST(inserted);
  BOOST_TEST(*slot == 99u);
}

BOOST_AUTO_TEST_CASE(monomial_store_survives_table_growth) {
  using I = impl<>;
  I::monomial_store s;

  // i in base 5 over eight letters: 5^8 distinct words, so every one of these
  // is genuinely new and the table has to rehash several times
  constexpr uint32_t n = 200000;
  std::vector<uint8_t> w(8);
  auto word_of = [&w](uint32_t i) {
    for(size_t k = 0; k < 8; k++) {
      w[k] = static_cast<uint8_t>(1 + (i % 5));
      i /= 5;
    }
    return w;
  };

  std::vector<uint32_t> ids;
  ids.reserve(n);
  for(uint32_t i = 0; i < n; i++)
    ids.push_back(s.getid(word_of(i)));

  // distinct words got distinct ids
  auto sorted = ids;
  std::sort(sorted.begin(), sorted.end());
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
  BOOST_TEST(sorted.size() == ids.size());

  // and nothing was lost by a rehash: every word still finds its own id
  for(uint32_t i = 0; i < n; i++)
    BOOST_REQUIRE(s.getid(word_of(i)) == ids[i]);
  BOOST_TEST(s.table_used() == ids.size());
  BOOST_TEST(s.table_slots() > s.table_used());
}
