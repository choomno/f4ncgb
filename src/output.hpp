#pragma once

#include <ostream>

#include "coeff.hpp"
#include "options.hpp"
#include "parser.hpp"

namespace f4ncgb {

inline std::ostream&
write_header(std::ostream& o,
             parser_context& names,
             const solve_options& opts) {
  for(auto& b : opts.blocks) {
    for(auto v : b) {
      names.var_to_ostream(o, v);
      if(v < opts.num_vars)
        o << ",";
    }
    o << "\n";
  }
  o << opts.characteristic << "\n";
  return o;
}

template<class MS>
std::ostream&
write_monomial(std::ostream& o,
               parser_context& names,
               const MS& mons,
               typename MS::index_type mon_id,
               bool print_asterisk = false) {

  // Print the monomial with '*' as needed
  for(auto mon : mons[mon_id]) {
    if(print_asterisk)
      o << "*";
    print_asterisk = true;
    names.var_to_ostream(o, mon);
  }
  return o;
}

template<class PS>
std::ostream&
write_polynomial(std::ostream& o,
                 parser_context& names,
                 const PS& p,
                 typename PS::index_type poly_id,
                 bool first_poly,
                 const coeff* reduce_denom = nullptr) {
  if(!first_poly)
    o << ",\n";

  // special case to print zero polynomial
  if(poly_id == 0) {
    o << "0";
    return o;
  }

  const typename PS::monomial_store_& mons = p.get_monomial_store();
  auto coeff_it = p.get_coefficients(poly_id).begin();
  bool first = true;
  const coeff& lc = reduce_denom ? *reduce_denom : *coeff_it;
  rat_coeff rc;

  for(const auto& mon_id : p[poly_id]) {

    bool print_asterisk = false;
    rc.update(*coeff_it, lc);

    // print sign
    if(!first)
      o << (rc.sign() < 0 ? " - " : " + ");
    else if(rc.sign() < 0)
      o << "-";

    // print coefficient if it's not ±1
    if(!rc.is_unit()) {
      o << rc.abs();
      print_asterisk = true;
    }
    // print monomial
    write_monomial(o, names, mons, mon_id, print_asterisk);

    // special case: empty monomial and coefficient ±1
    if(mon_id == 0 && rc.is_unit()) {
      o << "1";
    }

    first = false;
    coeff_it++;
  }

  return o;
}

template<class PS>
std::ostream&
write_all(std::ostream& o,
          parser_context& names,
          const solve_options& opts,
          const PS& p) {
  write_header(o, names, opts);

  bool first_poly = true;
  for(auto poly_it = p.begin() + 1u; poly_it != p.end(); ++poly_it) {
    auto poly_id = *poly_it;
    write_polynomial(o, names, p, poly_id, first_poly);
    first_poly = false;
  }
  o << std::endl;
  return o;
}
}
