#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <numeric>
#include <ostream>
#include <utility>
#include <vector>

#include <boost/align/align_down.hpp>
#include <boost/align/align_up.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>

#include "ambiguity.hpp"
#include "coeff.hpp"
#include "diagnostics.hpp"
#include "f4ncgb.hpp"
#include "linear_algebra.hpp"
#include "options.hpp"
#include "output.hpp"
#include "parser.hpp"
#include "solve_state.hpp"
#include "sparse_rref/thread_pool.hpp"

#include "monomial_trie.hpp"
#include "polynomial_store.hpp"
#include "store.hpp"

namespace f4ncgb {

struct metadata_monomial {
  uint32_t length = 0;
};

struct metadata_polynomial {
  uint32_t length = 0;
};

//------------------------------------------------------------------------------

template<size_t Nblocks,
         internal::value_concept V = uint8_t,
         typename I = uint32_t,
         typename C = coeff>
struct f4 {
  using MM = metadata_monomial;
  using PM = metadata_polynomial;
  using monomial_store = internal::monomial_store<MM, V, I, Nblocks>;
  using polynomial_store = internal::polynomial_store<PM, MM, V, I, C, Nblocks>;

  using monomial = std::span<const V>;
  using mon_id = I;
  using poly_id = I;

  using ambiguity_ = ambiguity<mon_id>;
  using crit_pair = std::pair<poly_id, poly_id>;

  using monomial_trie_ = monomial_trie<V, I>;

  public:
  /// Names, for printing monomials back out, and for reading the input.
  parser_context& context;
  /// What this computation was asked to do.
  const solve_options& opts;

  monomial_store mons;
  polynomial_store poly;
  std::vector<poly_id> basis;
  std::map<size_t, std::vector<ambiguity_>> amb;
  std::vector<ambiguity_> crit_pairs;
  boost::unordered_flat_map<mon_id, poly_id> lm_to_poly;
  monomial_trie_ prefix_trie;
  monomial_trie_ suffix_trie;
  monomial_trie_ gm_trie;
  std::unique_ptr<BS::thread_pool<BS::none>> pool;
  std::vector<poly_id> spolies;
  std::vector<poly_id> reducers;
  std::vector<mon_id> columns;

  /// Everything the parser and the linear algebra need to see of this
  /// computation. Owning it here is what keeps concurrent solves apart.
  solve_state state;

  size_t characteristic = 0;
  size_t iter = 0;
  size_t maxiter = UINT_MAX;
  size_t maxdeg = UINT_MAX;
  bool tracer = true;
  bool constant_flag = false;

  static constexpr bool block_order = Nblocks > 1;

  std::ofstream proof_file;

  f4(parser_context& context_, const solve_options& opts_)
    : context(context_)
    , opts(opts_)
    , mons()
    , poly(mons)
    , prefix_trie(opts_.num_vars)
    , suffix_trie(opts_.num_vars)
    , gm_trie(opts_.num_vars)
    , characteristic(opts_.characteristic)
    , maxiter(opts_.maxiter)
    , maxdeg(opts_.maxdeg)
    , tracer(opts_.tracer) {

    state.proof_level = opts.proof_level;

    mons.set_blocks(opts.block_sizes());

    if(opts.threads > 1)
      pool = std::make_unique<BS::thread_pool<BS::none>>(opts.threads);

    if(opts.proof_file != "") {
      proof_file.open(opts.proof_file, std::ios_base::trunc);
      if(!proof_file)
        die(19, "Failed to open proof file.");
    }
  }
  //------------------------------------------------------------------------------
  struct triplet {
    mon_id a;
    int i;
    mon_id b;

    triplet(mon_id a_, int i_, mon_id b_)
      : a(a_)
      , i(i_)
      , b(b_) {}
  };

  struct cofactor {
    coeff c;
    triplet t;

    cofactor(coeff c_, mon_id a_, int i_, mon_id b_)
      : c(std::move(c_))
      , t(a_, i_, b_) {}

    inline mon_id a() const { return t.a; }
    inline int i() const { return t.i; }
    inline mon_id b() const { return t.b; }

    inline void log(std::ostream& o,
                    parser_context& context,
                    monomial_store& mon,
                    const coeff& lc,
                    bool first = false) const {

      rat_coeff rc;
      rc.update(c, lc);

      if(!first)
        o << (rc.sign() > 0 ? " + " : " ");
      o << rc << "*";
      if(t.a != 0) {
        write_monomial(o, context, mon, t.a, false);
        o << "*";
      }
      if(t.i >= 0)
        o << "[" << t.i << "]";
      else
        o << "[i" << -t.i - 1 << "]";
      if(t.b != 0) {
        o << "*";
        write_monomial(o, context, mon, t.b, false);
      }
    }
  };

  std::vector<triplet> extended_spolies;
  std::vector<triplet> extended_reducers;
  std::vector<std::vector<cofactor>> cofactors;
  //------------------------------------------------------------------------------
  inline parse_res read_input(parser_context& context) {
    parse_res res
      = parse_rest_into_polynomial_store(context, poly, state.input_denoms);
    // Print in case of doubt.
    // write_all(std::cout, context, opts, poly);
    return res;
  }
  //------------------------------------------------------------------------------
  inline void write_basis(std::ostream& o) {
    bool first = true;
    const coeff* denom = state.reduce_mode ? &state.reduce_denom_c : nullptr;
    for(size_t n = 1; n < basis.size(); n++) {
      write_polynomial(o, context, poly, basis[n], first, denom);
      first = false;
    }
    o << "\n";
  }
  //------------------------------------------------------------------------------
  inline void write_basis(void* userdata,
                          f4ncgb_add_cb add,
                          f4ncgb_end_poly_cb end) {
    std::vector<uint32_t> data;

    for(size_t n = 1; n < basis.size(); n++) {
      auto poly_id = basis[n];
      // special case: zero polynomial
      if(poly_id == 0) {
        add(userdata, nullptr, nullptr, 0, nullptr);
      } else {
        auto coeff_it = poly.get_coefficients(poly_id).begin();
        coeff lc = state.reduce_mode ? state.reduce_denom_c : *coeff_it;
        for(const auto mon_id : poly[poly_id]) {
          auto vars = mons[mon_id];
          data.clear();
          std::copy(vars.begin(), vars.end(), std::back_inserter(data));
          coeff c = *coeff_it++;
          add(userdata, c.value, lc.value, vars.size(), data.data());
        }
      }
      end(userdata);
    }
  }
  //------------------------------------------------------------------------------
  std::vector<poly_id> new_elements;
  void interreduce_and_add_to_basis(std::vector<poly_id>& polies) {

    state.interreduce = true;

    if(verbose > 1)
      msg("Linearly interreducing input of size %d.", polies.size());

    size_t i = 0;
    for(const auto& p : polies) {
      spolies.push_back(p);
      if(state.proof_level > 0)
        extended_spolies.emplace_back(0, i++, 0);
    }

    // to leave 0th position open; just like in basis
    cofactors.emplace_back();

    // there is no basis yet, so this only collects the columns
    symbolic_preprocessing();

    // reduce and store result in new_elements
    reduction();

    if(verbose > 1)
      msg("Adding %d input elements to basis.", new_elements.size());

    update_basis_and_amb();

    state.interreduce = false;

    if(verbose > 0)
      msg("==== Interreduction finished. Basis has now %d elements ====",
          basis.size() - 1);
  }
  //------------------------------------------------------------------------------

  std::vector<poly_id> input;
  void compute_basis() {
    // add something at 0th position
    // so that index 0 remains free
    basis.push_back(0);
    input.clear();
    input.resize(poly.size() - 1);
    std::copy(poly.begin() + 1, poly.end(), input.begin());

    // add input to critical pairs
    interreduce_and_add_to_basis(input);

    // main loop
    iter = 0;
    while((!amb.empty() or !crit_pairs.empty()) and iter < maxiter
          and !constant_flag) {
      {
        F4NCGB_TIME(crit_pair);
        stage_crit_pairs();
      }

      if(verbose > 1)
        msg("Reducing %d critical pairs.", crit_pairs.size());

      spolies.clear();
      symbolic_preprocessing();

      // store results in new_elements
      reduction();

      if(verbose > 1)
        msg("Adding %d new elements to basis.", new_elements.size());

      update_basis_and_amb();

      iter++;
      if(verbose > 0)
        msg("==== Iteration %d has finished. Basis has now %d elements ====",
            iter,
            basis.size() - 1);
    }
  }
  //------------------------------------------------------------------------------
  void reduced_form() {
    basis.push_back(0);
    input.clear();
    if(poly.size() < 2)
      die(8, "At least one reducer required.");
    input.resize(poly.size() - 1);
    std::copy(poly.begin() + 1, poly.end(), input.begin());

    // separate last element, this the one to be reduced
    poly_id p = input.back();
    input.pop_back();

    // Set the parameters for the reduce path. Both overrides are deliberate:
    // the normal form is read back out of the cofactors, so proof logging has
    // to be on whatever the caller asked for, and the reduction runs for a
    // bounded number of rounds rather than the caller's iteration budget.
    state.reduce_mode = true;
    maxiter = 10;
    state.proof_level = 1;
    state.reduce_denom_c = state.input_denoms.back();
    state.input_denoms.pop_back();

    // interreduce input
    interreduce_and_add_to_basis(input);

    // when input contains 1, normal form is zero
    if(constant_flag) {
      basis.clear();
      basis.push_back(0);
      basis.push_back(0);
      return;
    }

    // perform reduction
    spolies.clear();
    spolies.push_back(p);
    symbolic_preprocessing();
    reduction();

    // reduction to zero
    if(new_elements.empty()) {
      basis.clear();
      basis.push_back(0);
      basis.push_back(0);
      return;
    }

    // nonzero reduction + set up normalization coeff
    poly_id normal_form = new_elements[0];
    state.reduce_denom_c = cofactors.back().back().c;

    basis.clear();
    basis.push_back(0);
    basis.push_back(normal_form);
  }

  //------------------------------------------------------------------------------

  inline void stage_crit_pairs() {
    if(amb.empty())
      return;

    auto minimal_amb = amb.begin();
    size_t d = minimal_amb->first;

    crit_pairs.reserve(minimal_amb->second.size());
    for(const auto& a : minimal_amb->second) {
      a.prepare(poly, lm_to_poly);
      crit_pairs.push_back(a);
    }
    amb.erase(d);

    // sort by increasing lm
    std::sort(crit_pairs.begin(),
              crit_pairs.end(),
              [this](const auto& a, const auto& b) {
                return this->mons.template cmp<block_order>(a.lm(), b.lm());
              });
  }
  //------------------------------------------------------------------------------
  std::vector<char> to_remove;
  std::vector<mon_id> next_same_ci;
  /// Drops the ambiguities that Gebauer and Moeller's criteria make redundant
  /// and files the rest under their degree.
  ///
  /// The two sorts look like one too many -- deduplicating through a hash set
  /// would be cheaper than sorting 48-byte ambiguities by (hash, values). They
  /// are not interchangeable though: that first sort is what puts the list in a
  /// canonical order, the second one can tie, and the filter below depends on
  /// which of two tied ambiguities it sees first. Replacing it changes which
  /// ambiguities get dropped, unless the second sort is given a total
  /// tie-break.
  void gebauer_moeller(std::vector<ambiguity_>& new_amb) {
    // first index is always the newer polynomial

    // sort to group identical elements and remove duplicates
    std::sort(new_amb.begin(),
              new_amb.end(),
              [](const ambiguity_& a, const ambiguity_& b) {
                if(a.hash_ != b.hash_)
                  return a.hash_ < b.hash_;
                return a.values < b.values;
              });
    new_amb.erase(std::unique(new_amb.begin(), new_amb.end()), new_amb.end());

    // sort
    auto cmp = [this](const ambiguity_& a, const ambiguity_& b) {
      if(a.degree() != b.degree())
        return a.degree() < b.degree();
      if(a.j() != b.j())
        return a.j() < b.j();
      return this->mons.template cmp<block_order>(a.aj(), b.aj());
    };
    std::sort(new_amb.begin(), new_amb.end(), cmp);

    to_remove.clear();
    to_remove.resize(new_amb.size(), false);

    gm_trie.clear();
    next_same_ci.clear();
    next_same_ci.resize(new_amb.size(), 0);

    for(size_t j = 0; j < new_amb.size(); j++) {
      auto& b = new_amb[j];
      const monomial b_ai = mons[b.ai()];
      const monomial b_ci = mons[b.ci()];

      bool divisible = false;
      const auto& divs = gm_trie.get_prefixes(b_ci);

      for(const auto& match : divs) {
        mon_id idx = match.first;
        while(idx != 0) {
          size_t i = static_cast<size_t>(idx - 1);
          auto& a = new_amb[i];
          const monomial a_ai = mons[a.ai()];
          if(endswith(b_ai, a_ai)) {
            divisible = true;
            break;
          }
          idx = next_same_ci[i];
        }
        if(divisible)
          break;
      }

      if(divisible) {
        to_remove[j] = true;
      } else {
        mon_id old_id = gm_trie.insert_chain(b_ci, j + 1);
        next_same_ci[j] = old_id;
      }
    }

    for(size_t j = 0; j < new_amb.size(); j++) {
      if(!to_remove[j]) {
        auto& a = new_amb[j];
        amb[a.degree()].push_back(std::move(a));
      }
    }
  }
  //------------------------------------------------------------------------------
  std::vector<std::pair<mon_id, size_t>> overlaps;
  std::vector<std::pair<mon_id, size_t>> inclusions;
  std::vector<ambiguity_> new_amb;
  void compute_ambiguities(mon_id i) {
    monomial m = mons[i];
    monomial ab, b, bc, abc;

    new_amb.clear();
    overlaps.clear();
    inclusions.clear();

    {
      F4NCGB_TIME(overlap);
      prefix_trie.overlaps_and_inclusions(m, overlaps, inclusions);
      // overlaps with m = AB
      ab = m;
      // last k elements of m = AB form overlap B
      for(const auto& [j, k] : overlaps) {
        bc = mons[j];
        I d = ab.size() + bc.size() - k;
        if(d > maxdeg)
          continue;
        I aj = mons.getid(ab.first(ab.size() - k));
        I ci = mons.getid(bc.last(bc.size() - k));
        ambiguity_ a(d, i, j, 0, ci, aj, 0);
        new_amb.push_back(a);
      }
      overlaps.clear();

      // overlaps with m = BC
      suffix_trie.overlaps_rev(m, overlaps);
      bc = m;
      // k determines where B starts in m = BC
      for(const auto& [j, k] : overlaps) {
        ab = mons[j];
        I d = ab.size() + bc.size() - k;
        if(d > maxdeg)
          continue;
        I ai = mons.getid(ab.first(ab.size() - k));
        I cj = mons.getid(bc.last(bc.size() - k));
        ambiguity_ a(d, i, j, ai, 0, 0, cj);
        new_amb.push_back(a);
      }
    }

    {
      F4NCGB_TIME(inclusion);
      // inclusions with m = ABC
      I d = m.size();
      // Every one of these has degree |m|, so the bound is checked once for all
      // of them rather than per match. It can only bite when an input
      // polynomial is itself longer than the bound: everything derived later is
      // bounded by the ambiguities it came from.
      if(d <= maxdeg) {
        // k determines where B starts in m = ABC
        for(const auto& [j, k] : inclusions) {
          if(i == j)
            continue;
          b = mons[j];
          I aj = mons.getid(m.first(k));
          I cj = mons.getid(m.last(d - k - b.size()));
          ambiguity_ a(d, i, j, 0, 0, aj, cj);
          new_amb.push_back(a);
        }
      }
      inclusions.clear();

      // inclusions with m = B
      b = m;
      prefix_trie.inclusions(m, inclusions);
      // last k elements in ABC form C
      for(const auto& [j, k] : inclusions) {
        if(i == j)
          continue;
        abc = mons[j];
        I d = abc.size();
        if(d > maxdeg)
          continue;
        I ai = mons.getid(abc.first(abc.size() - b.size() - k));
        I ci = mons.getid(abc.last(k));
        ambiguity_ a(d, i, j, ai, ci, 0, 0);
        new_amb.push_back(a);
      }
    }

    gebauer_moeller(new_amb);
  }

  //------------------------------------------------------------------------------

  boost::unordered_flat_set<mon_id> todo_seen;
  std::vector<mon_id> todo_vec;

  /// Collects the rows of the next matrix into spolies/reducers, and its
  /// columns into `columns`.
  ///
  /// The columns come for free: every monomial of every row is passed to
  /// push_todo() anyway, so `todo_seen` is exactly the column set. Only the
  /// critical pairs' leading monomials are held back from the queue -- they are
  /// pivots, and looking for a reducer for them would add a redundant row -- so
  /// those go straight into `columns`.
  void symbolic_preprocessing() {
    F4NCGB_TIME(sym_pre);

    size_t todo_pos = 0;
    reducers.clear();
    columns.clear();
    todo_vec.clear();
    todo_seen.clear();
    todo_vec.reserve(1L << 16);
    todo_seen.reserve(1L << 16);

    // rows the caller staged: the whole input during interreduction, the
    // polynomial to be reduced in reduce_mode
    const size_t staged_spolies = spolies.size();

    auto push_todo = [&](mon_id m) {
      // insert into vector only if new
      if(todo_seen.insert(m).second)
        todo_vec.push_back(m);
    };

    auto handle_spoly = [&](poly_id p, auto ai, auto i, auto ci) {
      spolies.push_back(p);
      if(state.proof_level > 0)
        extended_spolies.emplace_back(
          ai, static_cast<int>(poly.get_idx(lm_to_poly[i])), ci);
    };

    auto handle_reducer = [&](poly_id p, auto ai, auto i, auto ci) {
      reducers.push_back(p);
      if(state.proof_level > 0)
        extended_reducers.emplace_back(
          ai, static_cast<int>(poly.get_idx(lm_to_poly[i])), ci);
    };

    // assumes that crit pairs are sorted by lm
    // in increasing order
    for(const auto& a : crit_pairs) {
      poly_id f = a.fi();
      poly_id g = a.fj();
      auto mons_f = poly[f];
      auto mons_g = poly[g];

      mon_id lm = mons_f[0];
      bool first_time = todo_seen.insert(lm).second;

      // a pivot column, deliberately not queued for a reducer search
      if(first_time)
        columns.push_back(lm);

      // lm never seen -> one into reducers
      // lm seen before -> both in spolies
      if(!first_time) {
        handle_spoly(f, a.ai(), a.i(), a.ci());
        handle_spoly(g, a.aj(), a.j(), a.cj());
      } else if(mons_f.size() < mons_g.size()) {
        handle_spoly(f, a.ai(), a.i(), a.ci());
        handle_reducer(g, a.aj(), a.j(), a.cj());
      } else {
        handle_reducer(f, a.ai(), a.i(), a.ci());
        handle_spoly(g, a.aj(), a.j(), a.cj());
      }

      for(const auto m : mons_f)
        push_todo(m);
      for(const auto m : mons_g)
        push_todo(m);
    }
    crit_pairs.clear();

    for(size_t i = 0; i < staged_spolies; i++)
      for(const auto m : poly[spolies[i]])
        push_todo(m);

    while(todo_pos < todo_vec.size()) {
      mon_id m = todo_vec[todo_pos++];
      poly_id reducer = find_reducer(m);
      if(!reducer)
        continue;

      assert(m == poly.get_lm_id(reducer));
      reducers.push_back(reducer);

      for(const mon_id m : poly[reducer])
        push_todo(m);
    }
  }
  //------------------------------------------------------------------------------
  poly_id find_reducer(mon_id m) {
    // Strategy: the smallest leading monomial, and of its occurrences the last
    // one. Stating it here rather than leaning on the order the trie happens to
    // report in keeps the choice -- and with it the whole basis -- independent
    // of how the search works.
    auto& red = prefix_trie.divisors(mons[m]);
    if(red.empty())
      return 0;
    std::pair<mon_id, size_t> match
      = *std::min_element(red.begin(), red.end(), [this](auto a, auto b) {
          if(a.first != b.first)
            return this->mons.template cmp<block_order>(a.first, b.first);
          return a.second > b.second;
        });

    monomial mm = mons[m];
    monomial lm = mons[match.first];
    mon_id a = mons.getid(mm.first(match.second));
    mon_id b = mons.getid(mm.last(mm.size() - match.second - lm.size()));
    poly_id g = lm_to_poly[match.first];

    if(state.proof_level > 0)
      extended_reducers.emplace_back(a, static_cast<int>(poly.get_idx(g)), b);

    poly_id res = poly.multiply_front_and_back(a, g, b);

    return res;
  }

  //------------------------------------------------------------------------------
  void log_cofactors() {

    // compute expanded proofs
    if(state.proof_level > 1 and basis.size() > 1) {
      std::vector<cofactor> expanded;
      for(size_t n = basis.size(); n < cofactors.size(); n++) {
        expanded.clear();
        for(const auto& cofactor : cofactors[n]) {
          const coeff& c = cofactor.c;
          mon_id a = cofactor.a();
          mon_id b = cofactor.b();
          for(const auto& cofactor_i : cofactors[(size_t)cofactor.i()]) {
            coeff cc = c * cofactor_i.c;
            mon_id aa = mons.get_product_id(a, cofactor_i.a());
            mon_id bb = mons.get_product_id(cofactor_i.b(), b);
            expanded.emplace_back(cc, aa, cofactor_i.i(), bb);
          }
        }
        cofactors[n] = std::move(expanded);
      }
      // non-expanded proof
    } else if(state.proof_level == 1) {
      // mark input
      if(basis.size() == 1)
        for(size_t n = basis.size(); n < cofactors.size(); n++)
          for(auto& cofactor : cofactors[n])
            cofactor.t.i = -cofactor.t.i - 1;
      // not input -> shift all indices down by one
      else
        for(size_t n = basis.size(); n < cofactors.size(); n++)
          for(auto& cofactor : cofactors[n])
            cofactor.t.i--;
    }

    // write to file
    bool first = true;
    for(size_t n = basis.size(); n < cofactors.size(); n++) {
      first = true;
      poly_id p = new_elements[n - basis.size()];
      const coeff& lc = *poly.get_coefficients(p).begin();
      for(const auto& cofactor : cofactors[n]) {
        cofactor.log(proof_file, context, mons, lc, first);
        first = false;
      }
      proof_file << std::endl;
      if(poly.get_lm_id(p) == 0)
        return;
    }
  }
  //------------------------------------------------------------------------------
  std::vector<poly_id> res;
  const std::vector<poly_id>& compute_new_polynomials(
    std::vector<std::pair<size_t, size_t>>& idxs,
    fmpz* entries,
    std::vector<mon_id>& columns) {

    std::vector<std::pair<coeff, mon_id>> p;
    std::vector<cofactor> current_cofactors;
    res.clear();
    poly.reset();

    if(idxs.size() == 0) {
      extended_spolies.clear();
      extended_reducers.clear();
      return res;
    }

    size_t n = columns.size();
    size_t k = 0;
    size_t cur_i = idxs[0].first;
    for(const auto& [i, j] : idxs) {
      // a new polynomial starts
      if(i != cur_i) {
        res.push_back(poly.add_polynomial(p));
        p.clear();
        cur_i = i;
        if(state.proof_level > 0) {
          cofactors.emplace_back(std::move(current_cofactors));
          current_cofactors.clear();
        }
      }

      coeff c;
      fmpz_swap(c.value, &entries[k++]);
      assert(!c.is_zero());

      if(j < n) {
        p.emplace_back(std::move(c), columns[j]);
      } else {
        size_t idx = j - n;
        if(idx < extended_reducers.size()) {
          current_cofactors.emplace_back(c,
                                         extended_reducers[idx].a,
                                         extended_reducers[idx].i,
                                         extended_reducers[idx].b);
        } else {
          idx -= extended_reducers.size();
          assert(idx < extended_spolies.size());
          current_cofactors.emplace_back(c,
                                         extended_spolies[idx].a,
                                         extended_spolies[idx].i,
                                         extended_spolies[idx].b);
        }
      }
    }
    // don't forget to add last element
    res.push_back(poly.add_polynomial(p));
    if(state.proof_level > 0)
      cofactors.emplace_back(std::move(current_cofactors));

    extended_spolies.clear();
    extended_reducers.clear();

    return res;
  }
  //------------------------------------------------------------------------------

  template<typename T>
  void apply_perm(std::vector<T>& vec, const std::vector<size_t>& perm) {
    assert(vec.size() == perm.size());
    std::vector<T> tmp;

    tmp.reserve(vec.size());
    for(size_t i : perm)
      tmp.push_back(vec[i]);
    vec.swap(tmp);
  }
  //------------------------------------------------------------------------------

  void sort_rows(std::vector<poly_id>& to_sort,
                 std::vector<triplet>& extended,
                 const boost::unordered_flat_map<mon_id, uint32_t>& col_to_id) {
    // sort rows (+ extended_rows accordingly)
    // first by lm (smaller first), then by support (smaller first)

    size_t m = to_sort.size();

    std::vector<size_t> perm(m);
    std::iota(perm.begin(), perm.end(), 0);

    std::vector<uint32_t> lm_col(m);
    for(size_t i = 0; i < m; i++)
      lm_col[i] = col_to_id.at(poly.get_lm_id(to_sort[i]));

    auto cmp = [&](size_t a, size_t b) {
      if(lm_col[a] != lm_col[b])
        return lm_col[a] > lm_col[b];
      return poly.get_length(to_sort[a]) < poly.get_length(to_sort[b]);
    };
    std::sort(perm.begin(), perm.end(), cmp);

    apply_perm(to_sort, perm);
    if(state.proof_level > 0) {
      apply_perm(extended, perm);
      if(state.interreduce and &to_sort == &spolies)
        apply_perm(state.input_denoms, perm);
    }
  }
  //------------------------------------------------------------------------------

  std::vector<std::span<coeff>> entries_spol;
  csr_indices idxs_spol;
  std::vector<std::span<coeff>> entries_red;
  csr_indices idxs_red;
  /// Length and leading variables of a monomial, packed so that comparing the
  /// two integers agrees with the monomial order. Equal keys decide nothing and
  /// fall back to the full comparison.
  ///
  /// Only valid for the plain degree order: a block order ranks by per-block
  /// degrees, which this does not capture.
  inline uint64_t sort_key(mon_id m) const {
    const monomial v = mons[m];

    // one shared key for monomials too long to encode, so they always fall back
    if(v.size() >= 0xffff)
      return ~uint64_t(0);

    constexpr size_t bits = 8 * sizeof(V);
    constexpr size_t packed = 48 / bits;

    uint64_t key = static_cast<uint64_t>(v.size()) << 48;
    const size_t n = std::min(packed, v.size());
    for(size_t i = 0; i < n; i++)
      key |= static_cast<uint64_t>(v[i]) << (48 - bits * (i + 1));
    return key;
  }

  std::vector<std::pair<uint64_t, mon_id>> keyed_columns;
  /// Puts the columns collected by symbolic_preprocessing() in DESCENDING
  /// order.
  void sort_columns() {
    columns.insert(columns.end(), todo_vec.begin(), todo_vec.end());

    if constexpr(block_order) {
      std::sort(
        columns.begin(), columns.end(), [this](const mon_id a, const mon_id b) {
          return this->mons.template cmp<block_order>(b, a);
        });
      return;
    }

    // Sorting on the packed key keeps the comparator out of the monomial pool:
    // it would otherwise chase two random pointers per comparison, of which
    // there are O(n log n).
    keyed_columns.clear();
    keyed_columns.reserve(columns.size());
    for(const auto m : columns)
      keyed_columns.emplace_back(sort_key(m), m);

    std::sort(keyed_columns.begin(),
              keyed_columns.end(),
              [this](const auto& x, const auto& y) {
                if(x.first != y.first)
                  return x.first > y.first;
                return this->mons.template cmp<block_order>(y.second, x.second);
              });

    columns.clear();
    for(const auto& [key, m] : keyed_columns)
      columns.push_back(m);
  }

  boost::unordered_flat_map<mon_id, uint32_t> col_to_id;
  void prepare_matrix() {

    sort_columns();
    state.nr_cols = columns.size();

    col_to_id.clear();
    col_to_id.reserve(columns.size());
    uint32_t i = 0;
    for(const auto c : columns)
      col_to_id[c] = i++;

    // sort spolies and reducers
    sort_rows(spolies, extended_spolies, col_to_id);
    sort_rows(reducers, extended_reducers, col_to_id);

    // collect entries and indices
    entries_spol.clear();
    idxs_spol.clear();
    entries_red.clear();
    idxs_red.clear();

    entries_spol.reserve(spolies.size());
    entries_red.reserve(reducers.size());

    auto collect_rows = [this](const std::vector<poly_id>& rows,
                               std::vector<std::span<coeff>>& entries,
                               csr_indices& idxs) {
      for(const auto r : rows) {
        entries.push_back(poly.get_coefficients(r));
        for(const auto m : poly[r])
          idxs.push(col_to_id.at(m));
        idxs.end_row();
      }
    };

    collect_rows(spolies, entries_spol, idxs_spol);
    collect_rows(reducers, entries_red, idxs_red);
  }

  //------------------------------------------------------------------------------

  void reduction() {

    {
      F4NCGB_TIME(matrix_setup);
      prepare_matrix();
    }

    // reduction
    F4NCGB_PROFILE(auto timer = gstats.time(gstats.reduction));
    auto [idxs, entries] = linear_algebra(state,
                                          idxs_spol,
                                          idxs_red,
                                          entries_spol,
                                          entries_red,
                                          characteristic,
                                          pool,
                                          tracer);

    // compute new elements
    F4NCGB_PROFILE(auto timer2 = gstats.time(gstats.new_elements));
    new_elements = compute_new_polynomials(idxs, entries, columns);

    fmpz_cleanup(entries, idxs.size());
  }
  //------------------------------------------------------------------------------
  void update_basis_and_amb() {

    // log cofactors
    if(state.proof_level > 0 and !state.reduce_mode) {
      F4NCGB_TIME(other);
      log_cofactors();
    }

    uint32_t n = basis.size();
    for(const poly_id p_id : new_elements) {
      // only do all of this if we don't termiante next iteration
      if(iter < maxiter) {
        // update lm data
        mon_id m_id = poly.get_lm_id(p_id);
        lm_to_poly[m_id] = p_id;

        // update tries
        monomial m = mons[m_id];
        prefix_trie.insert(m, m_id);
        suffix_trie.insert_rev(m, m_id);

        // compute ambiguities
        if(!state.reduce_mode) {
          F4NCGB_TIME(amb);
          compute_ambiguities(m_id);
        }
      }

      // update basis
      poly.set_idx(p_id, n++);
      basis.push_back(p_id);

      // special flag for constant polynomial
      if(poly.get_lm_id(p_id) == 0) {
        constant_flag = true;
        return;
      }
    }
  }
};
}
