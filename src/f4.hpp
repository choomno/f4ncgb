#pragma once

#include <algorithm>
#include <bit>
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
  parser_context& context;
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

    // Variables are numbered 1..num_vars, so that is all the key has to hold.
    key_bits
      = std::max<size_t>(1, static_cast<size_t>(std::bit_width(opts.num_vars)));
    key_packed = 48 / key_bits;

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
  std::vector<coeff> cofactor_denom;
  //------------------------------------------------------------------------------
  inline parse_res read_input(parser_context& context) {
    parse_res res
      = parse_rest_into_polynomial_store(
        context, poly, state.input_denoms, opts.num_vars);
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
    // index 0 shall remain free
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

    // set the parameters for the reduce path
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

  boost::unordered_flat_map<mon_id, uint32_t> todo_seen;
  std::vector<uint32_t> pivot_slots;
  size_t last_columns = 0;

  void symbolic_preprocessing() {
    F4NCGB_TIME(sym_pre);

    reducers.clear();
    columns.clear();
    pivot_slots.clear();
    todo_seen.clear();
    idxs_spol.clear();
    idxs_red.clear();

    const size_t guess = std::max<size_t>(1L << 16, last_columns);
    columns.reserve(guess);
    todo_seen.reserve(guess);

    // rows the caller staged: the whole input during interreduction, the
    // polynomial to be reduced in reduce_mode
    const size_t staged_spolies = spolies.size();

    auto touch = [&](mon_id m) {
      auto [it, inserted]
        = todo_seen.try_emplace(m, static_cast<uint32_t>(columns.size()));
      if(inserted) {
        columns.push_back(m);
        check_fits(columns.size(), UINT32_MAX, "number of matrix columns");
      }
      return it->second;
    };

    auto emit_row = [&](csr_indices& idxs, auto mons) {
      for(const auto m : mons)
        idxs.push(touch(m));
      idxs.end_row();
    };

    auto handle_spoly = [&](poly_id p, auto mons, auto ai, auto i, auto ci) {
      spolies.push_back(p);
      emit_row(idxs_spol, mons);
      if(state.proof_level > 0)
        extended_spolies.emplace_back(
          ai, static_cast<int>(poly.get_idx(lm_to_poly[i])), ci);
    };

    auto handle_reducer = [&](poly_id p, auto mons, auto ai, auto i, auto ci) {
      reducers.push_back(p);
      emit_row(idxs_red, mons);
      if(state.proof_level > 0)
        extended_reducers.emplace_back(
          ai, static_cast<int>(poly.get_idx(lm_to_poly[i])), ci);
    };

    for(size_t i = 0; i < staged_spolies; i++)
      emit_row(idxs_spol, poly[spolies[i]]);

    // assumes that crit pairs are sorted by lm
    // in increasing order
    for(const auto& a : crit_pairs) {
      poly_id f = a.fi();
      poly_id g = a.fj();
      auto mons_f = poly[f];
      auto mons_g = poly[g];

      mon_id lm = mons_f[0];
      bool first_time
        = todo_seen.try_emplace(lm, static_cast<uint32_t>(columns.size()))
            .second;
      if(first_time) {
        pivot_slots.push_back(static_cast<uint32_t>(columns.size()));
        columns.push_back(lm);
      }

      // lm never seen -> one into reducers
      // lm seen before -> both in spolies
      if(!first_time) {
        handle_spoly(f, mons_f, a.ai(), a.i(), a.ci());
        handle_spoly(g, mons_g, a.aj(), a.j(), a.cj());
      } else if(mons_f.size() < mons_g.size()) {
        handle_spoly(f, mons_f, a.ai(), a.i(), a.ci());
        handle_reducer(g, mons_g, a.aj(), a.j(), a.cj());
      } else {
        handle_reducer(f, mons_f, a.ai(), a.i(), a.ci());
        handle_spoly(g, mons_g, a.aj(), a.j(), a.cj());
      }
    }
    crit_pairs.clear();

    size_t next_pivot = 0;
    for(size_t slot = 0; slot < columns.size(); slot++) {
      if(next_pivot < pivot_slots.size() and pivot_slots[next_pivot] == slot) {
        next_pivot++;
        continue;
      }

      mon_id m = columns[slot];
      poly_id reducer = find_reducer(m);
      if(!reducer)
        continue;

      assert(m == poly.get_lm_id(reducer));
      reducers.push_back(reducer);
      emit_row(idxs_red, poly[reducer]);
    }

    last_columns = columns.size();
  }
  //------------------------------------------------------------------------------
  poly_id find_reducer(mon_id m) {
    // Strategy: the reducer with the fewest terms, ties going to the smallest
    // leading monomial and then to the last occurrence.
    
    std::pair<mon_id, size_t> match{ 0, 0 };
    uint32_t best_terms = 0;
    prefix_trie.for_each_shortest_divisor(
      mons[m],
      [this, &match, &best_terms](mon_id id, size_t start, uint32_t terms) {
        // positions grow, so an equal word here is the later occurrence
        if(id == match.first) {
          match.second = start;
          return;
        }
        if(match.first == 0 or terms < best_terms
           or (terms == best_terms
               and this->mons.template cmp<block_order>(id, match.first))) {
          match = { id, start };
          best_terms = terms;
        }
      });
    if(match.first == 0)
      return 0;

    monomial mm = mons[m];
    monomial lm = mons[match.first];
    mon_id a = mons.getid(mm.first(match.second));
    mon_id b = mons.getid(mm.last(mm.size() - match.second - lm.size()));
    poly_id g = lm_to_poly[match.first];

    if(state.proof_level > 0)
      extended_reducers.emplace_back(a, static_cast<int>(poly.get_idx(g)), b);

    poly_id res = poly.multiply_front_and_back(a, g, b, m);

    return res;
  }

  //------------------------------------------------------------------------------
  static void cancel_common_factor(std::vector<cofactor>& cf, coeff& denom) {
    if(cf.empty())
      return;

    coeff g = denom;
    for(const auto& c : cf) {
      g.gcd_inplace(c.c);
      if(g.is_one())
        return;
    }

    for(auto& c : cf)
      c.c /= g;
    denom /= g;
  }
  //------------------------------------------------------------------------------
  void log_cofactors() {

    // The new elements are still stored unnormalized, so their cofactors are
    // too.
    cofactor_denom.resize(cofactors.size());
    for(size_t n = basis.size(); n < cofactors.size(); n++)
      cofactor_denom[n]
        = *poly.get_coefficients(new_elements[n - basis.size()]).begin();

    // compute expanded proofs
    if(state.proof_level > 1 and basis.size() > 1) {
      std::vector<cofactor> expanded;
      for(size_t n = basis.size(); n < cofactors.size(); n++) {
        expanded.clear();

        // Substituting element i multiplies by a coefficient of
        // cofactors[i], which is written over cofactor_denom[i] rather than
        // over 1. Collecting those into one denominator for the whole list
        // keeps every coefficient an integer, as each of them divides it.
        coeff common(1);
        for(const auto& cofactor : cofactors[n]) {
          assert((size_t)cofactor.i() < basis.size());
          common.lcm_inplace(cofactor_denom[(size_t)cofactor.i()]);
        }

        for(const auto& cofactor : cofactors[n]) {
          const coeff& c = cofactor.c;
          mon_id a = cofactor.a();
          mon_id b = cofactor.b();
          const coeff scale = common / cofactor_denom[(size_t)cofactor.i()];
          for(const auto& cofactor_i : cofactors[(size_t)cofactor.i()]) {
            coeff cc = c * cofactor_i.c * scale;
            mon_id aa = mons.get_product_id(a, cofactor_i.a());
            mon_id bb = mons.get_product_id(cofactor_i.b(), b);
            expanded.emplace_back(std::move(cc), aa, cofactor_i.i(), bb);
          }
        }
        cofactors[n] = std::move(expanded);
        cofactor_denom[n] *= common;
        cancel_common_factor(cofactors[n], cofactor_denom[n]);
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
      for(const auto& cofactor : cofactors[n]) {
        cofactor.log(proof_file, context, mons, cofactor_denom[n], first);
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

  void apply_perm(csr_indices& idxs, const std::vector<size_t>& perm) {
    assert(idxs.rows() == perm.size());
    csr_scratch.clear();
    csr_scratch.data.reserve(idxs.data.size());
    csr_scratch.row_start.reserve(perm.size() + 1);
    for(size_t i : perm) {
      const auto row = idxs[i];
      csr_scratch.data.insert(csr_scratch.data.end(), row.begin(), row.end());
      csr_scratch.end_row();
    }
    idxs.data.swap(csr_scratch.data);
    idxs.row_start.swap(csr_scratch.row_start);
  }
  //------------------------------------------------------------------------------

  std::vector<size_t> perm;
  std::vector<size_t> red_order;
  csr_indices csr_scratch;
  std::vector<uint32_t> lm_col;

  /// Sorts the rows by leading column, then by support.
  void sort_rows(std::vector<poly_id>& to_sort,
                 std::vector<triplet>& extended,
                 csr_indices& idxs,
                 std::vector<size_t>* order = nullptr) {

    size_t m = to_sort.size();
    assert(idxs.rows() == m);

    perm.resize(m);
    std::iota(perm.begin(), perm.end(), 0);

    lm_col.resize(m);
    for(size_t i = 0; i < m; i++)
      lm_col[i] = idxs[i][0];

    auto cmp = [&](size_t a, size_t b) {
      if(lm_col[a] != lm_col[b])
        return lm_col[a] > lm_col[b];
      return idxs[a].size() < idxs[b].size();
    };
    std::sort(perm.begin(), perm.end(), cmp);

    apply_perm(to_sort, perm);
    if(state.proof_level > 0) {
      apply_perm(extended, perm);
      if(state.interreduce and &to_sort == &spolies)
        apply_perm(state.input_denoms, perm);
    }

    if(order)
      order->swap(perm);
    else
      apply_perm(idxs, perm);
  }
  //------------------------------------------------------------------------------

  std::vector<std::span<coeff>> entries_spol;
  csr_indices idxs_spol;
  std::vector<std::span<coeff>> entries_red;
  csr_indices idxs_red;
 
  size_t key_bits = 8;
  size_t key_packed = 6;

  // Length and leading variables of a monomial, packed so that comparing the
  // two integers agrees with the monomial order. Equal keys fall back to the 
  // full comparison. Only valid for deglex.
  inline uint64_t sort_key(mon_id m) const {
    const monomial v = mons[m];

    // one shared key for monomials too long to encode, so they always fall back
    if(v.size() >= 0xffff)
      return ~uint64_t(0);

    uint64_t key = static_cast<uint64_t>(v.size()) << 48;
    const size_t n = std::min(key_packed, v.size());
    for(size_t i = 0; i < n; i++)
      key |= static_cast<uint64_t>(v[i]) << (48 - key_bits * (i + 1));
    return key;
  }
  //------------------------------------------------------------------------------

  std::vector<std::pair<uint64_t, uint32_t>> keyed_columns;
  std::vector<uint32_t> slot_to_col;
  std::vector<mon_id> sorted_columns;
  void sort_columns() {
    const size_t n = columns.size();

    keyed_columns.clear();
    keyed_columns.reserve(n);
    for(uint32_t s = 0; s < n; s++) {
      uint64_t key = 0;
      if constexpr(!block_order)
        key = sort_key(columns[s]);
      keyed_columns.emplace_back(key, s);
    }

    std::sort(keyed_columns.begin(),
              keyed_columns.end(),
              [this](const auto& x, const auto& y) {
                if(x.first != y.first)
                  return x.first > y.first;
                return this->mons.template cmp<block_order>(
                  this->columns[y.second], this->columns[x.second]);
              });

    slot_to_col.resize(n);
    sorted_columns.clear();
    sorted_columns.reserve(n);
    for(uint32_t c = 0; c < n; c++) {
      const uint32_t s = keyed_columns[c].second;
      slot_to_col[s] = c;
      sorted_columns.push_back(columns[s]);
    }
    columns.swap(sorted_columns);
  }

  void prepare_matrix() {

    sort_columns();
    state.nr_cols = columns.size();

    for(auto& c : idxs_spol.data)
      c = slot_to_col[c];
    for(auto& c : idxs_red.data)
      c = slot_to_col[c];

    // sort spolies and reducers
    sort_rows(spolies, extended_spolies, idxs_spol);
    sort_rows(reducers, extended_reducers, idxs_red, &red_order);

    // collect entries
    entries_spol.clear();
    entries_red.clear();
    entries_spol.reserve(spolies.size());
    entries_red.reserve(reducers.size());

    for(const auto r : spolies)
      entries_spol.push_back(poly.get_coefficients(r));
    for(const auto r : reducers)
      entries_red.push_back(poly.get_coefficients(r));
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
                                          red_order,
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

        // update tries. The term count is also included so that 
        // find_reducer() can rank a candidate without going back 
        // to the polynomial for it.
        monomial m = mons[m_id];
        prefix_trie.insert(m, m_id, poly.get_length(p_id));
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
