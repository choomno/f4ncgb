#include "solve.hpp"

#include <filesystem>

#include <boost/mp11/algorithm.hpp>

#include "diagnostics.hpp"
#include "f4.hpp"
#include "f4ncgb.hpp"
#include "options.hpp"
#include "output.hpp"
#include "parser.hpp"
#include "store.hpp"

namespace f4ncgb {

struct f4_base {
  virtual ~f4_base() = default;
  virtual parse_res read_input(parser_context& context) = 0;
  virtual void compute_basis() = 0;
  virtual void reduced_form() = 0;
  virtual void write_basis(std::ostream& os) = 0;
  virtual void write_basis(void* userdata,
                           f4ncgb_add_cb add_cb,
                           f4ncgb_end_poly_cb end_poly_cb) = 0;
  virtual void print_read_problem() = 0;
  virtual void print_parameters(const std::string& output_name) = 0;

#ifdef F4NCGB_ENABLE_STORE_DUMP
  virtual void set_dump_paths(std::string path) = 0;
#endif
};

template<std::size_t Nblocks, typename V>
struct f4_wrapper : f4_base {
  f4<Nblocks, V> algo;

  f4_wrapper(parser_context& context, const solve_options& opts)
    : algo(context, opts) {}

  parse_res read_input(parser_context& context) override {
    return algo.read_input(context);
  }
  void compute_basis() override { algo.compute_basis(); }
  void reduced_form() override { algo.reduced_form(); }
  void write_basis(std::ostream& os) override { algo.write_basis(os); }
  void write_basis(void* userdata,
                   f4ncgb_add_cb add_cb,
                   f4ncgb_end_poly_cb end_poly_cb) override {
    algo.write_basis(userdata, add_cb, end_poly_cb);
  }
  void print_read_problem() override {
    write_all(std::cout, algo.context, algo.opts, algo.poly);
  }

  void print_parameters(const std::string& output_name) override {
    const solve_options& opts = algo.opts;
    msg("==== Input Parameters ====");
    std::string out_name = "Output file:       ";
    if(output_name == "")
      out_name += "None. Writing output to console.";
    else
      out_name += output_name;

    std::ostringstream mon_order;
    mon_order << "Monomial order:    ";
    bool first = true;
    if(opts.num_blocks() == 0) {
      for(size_t v = 1; v <= opts.num_vars; v++) {
        if(!first)
          mon_order << " < ";
        algo.context.var_to_ostream(mon_order, v);
        first = false;
      }
    } else {
      for(size_t i = 0; i < opts.num_blocks(); i++) {
        first = true;
        for(auto v : opts.block(i)) {
          if(!first)
            mon_order << " < ";
          algo.context.var_to_ostream(mon_order, v);
          first = false;
        }
        if(i < opts.num_blocks() - 1)
          mon_order << " << ";
      }
    }

    std::string proof_str = opts.proof_level > 0 ? "on " : "off";
    if(opts.proof_level > 1)
      proof_str += "(expanded) ";
    if(opts.proof_level > 0)
      proof_str += "(writing to " + opts.proof_file + ")";

    msg("Characteristic:    %lu", opts.characteristic);
    if(!opts.reduce) {
      msg("Max. Iterations:   %lu", opts.maxiter);
      msg("Max. amb. degree:  %lu", opts.maxdeg);
    }
    msg(mon_order.str().c_str());
    msg("Nr. threads:       %lu", opts.threads);
    msg(out_name.c_str());
    if(!opts.reduce)
      msg("Proof logging:     %s", proof_str.c_str());
    msg("Tracer:            %s", opts.tracer ? "on" : "off");
    msg("PID of F4NCGB:     %lu", getpid());
    if(opts.reduce)
      msg("==== Starting Normal Form Computation ====");
    else
      msg("==== Starting Gröbner Basis Computation ====");
  }

#ifdef F4NCGB_ENABLE_STORE_DUMP
  void set_dump_paths(std::string path) override {
    algo.mons.set_binary_dump_path(std::filesystem::path(path + "_mons.bin"));
    algo.poly.set_binary_dump_path(std::filesystem::path(path + "_poly.bin"));
  }
#endif
};

int
f4ncgb_main(parser_context& context,
            const solve_options& opts,
            const run_options& run) {

  return boost::mp11::mp_with_index<F4NCGB_MAX_BLOCKS>(
    opts.num_blocks(), [&context, &opts, &run](auto Nblocks) -> int {
      std::unique_ptr<f4_base> algo_ptr;
      size_t nvars = opts.num_vars;

      if(nvars < std::numeric_limits<uint8_t>::max()) {
        algo_ptr
          = std::make_unique<f4_wrapper<Nblocks, uint8_t>>(context, opts);
      } else if(nvars < std::numeric_limits<uint16_t>::max()) {
        algo_ptr
          = std::make_unique<f4_wrapper<Nblocks, uint16_t>>(context, opts);
      } else {
        die(31, "Too many variables. At most 2^16-1 supported.");
      }

      auto& algo = *algo_ptr;

      {
        F4NCGB_TIME(parse);
        if(auto err = algo.read_input(context)) {
          std::cerr << *err << std::endl;
          die(17, "Error in parsing body of input file.");
        }
      }

      if(run.print_read_problem) {
        algo.print_read_problem();
      }

#ifdef F4NCGB_ENABLE_STORE_DUMP
      algo.set_dump_paths(
        std::filesystem::path(context.get_filename()).filename().string());
#endif

      if(run.stats_print_function) {
        *run.stats_print_function = [&algo]() {
          msg("Current basis:");
          algo.write_basis(std::cerr);
        };
      }

      if(verbose > 0)
        algo.print_parameters(run.output_name);

      if(opts.reduce)
        algo.reduced_form();
      else
        algo.compute_basis();

      if(verbose > 0) {
        if(opts.reduce)
          msg("==== Normal form computation finished ====");
        else
          msg("==== Basis computation finished ====");
      }

      if(run.add_cb && run.end_poly_cb) {
        algo.write_basis(run.userdata, run.add_cb, run.end_poly_cb);
      } else {
        std::ostream* out_file = &std::cout;
        std::ofstream filestream;
        if(run.output_name != "") {
          filestream.open(run.output_name, std::ios_base::trunc);
          if(!filestream)
            die(18, "Failed to open output file.");
          out_file = &filestream;
        }
        algo.write_basis(*out_file);
      }

      if(run.stats_print_function) {
        *run.stats_print_function = []() {};
      }

    // Tearing the stores down costs time and the caller may not need it. Only
    // skipped when nothing is watching: not under AddressSanitizer, not with
    // assertions on, and not while dumping stores.

#if defined(__has_feature) && NDEBUG && !defined(F4NCGB_ENABLE_STORE_DUMP)
#if !__has_feature(address_sanitizer)
      if(run.leak_memory)
        algo_ptr.release();
#endif
#endif
      return 0;
    });
}
}
