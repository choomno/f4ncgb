#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstdlib>
#include <exception>
#include <filesystem>

#include "diagnostics.hpp"
#include "options.hpp"
#include "parser.hpp"
#include "solve.hpp"
#include <flint/ulong_extras.h>

#include <boost/program_options.hpp>
#include <config.hpp>

#ifdef F4NCGB_ENABLE_SIGNAL
#include <csignal>
#endif

namespace po = boost::program_options;

// / Name of the input file
static std::string input_name = "";

// / \brief
// / Name of output files
static std::string output_name = "";
static std::string proof_file = "";

// / Selected prime, maxiter, maxdeg, and number of threads
static size_t maxiter = 0;
static size_t maxdeg = 0;
static size_t threads = 0;

// / Proof logging
static size_t user_proof_level = 0;
static bool expanded_proof = false;
static bool tracer = true;
static bool reduce = false;
/*------------------------------------------------------------------------*/
// ERROR CODES:

static int err_no_file = 10;// no input file given
static int err_prime_too_big = 11;
static int err_no_prime = 12;

static bool print_problem = false;

using namespace f4ncgb;

static std::function<void()> print_statistics_fun;

#ifdef F4NCGB_ENABLE_SIGNAL
static void
handle_signal(int signal) {
  if(signal == SIGUSR1) {
    if(print_statistics_fun) {
      msg("SIGUSR1 received.");
      print_statistics_fun();
      msg("End of SIGUSR1 output.");
    }
  }
}
#endif

/**
    Main Function of f4ncgb.

    Prints statistics to stdout after finishing.
*/
int
main(int argc, char** argv) {
#ifdef F4NCGB_ENABLE_SIGNAL
  signal(SIGUSR1, handle_signal);
#endif

  po::options_description desc("f4ncgb, version " F4NCGB_VERSION "\n"
                               "Copyright(C) 2025 Clemens Hofstadler, Maximilian Heisinger\n"
                               "JKU Linz, Austria\n"
                               "Repository: https://gitlab.sai.jku.at/f4ncgb/f4ncgb\n"
                               "USAGE");
  desc.add_options()
    ("version", "produce version message")
    ("help,h", "produce help message")
    ("expanded-proof,e",  po::value<bool>(&expanded_proof)->default_value(false), "Whether the proofs should be expanded and written in terms of the input. Requires proof logging to be turned on.")
    ("input,i", po::value<std::string>(&input_name)->default_value(""), "Set the input file (either msolve or poly; switched according to content).")
    ("maxiter,m", po::value<size_t>(&maxiter)->default_value(10), "Maximal number of iterations of the F4-algorithm to be performed.")
    ("maxdeg,d", po::value<size_t>(&maxdeg)->default_value(UINT_MAX), "Maximal degree of ambiguities that are considered.")
    ("output,o", po::value<std::string>(&output_name)->default_value(""), "Set the output file.")
    ("print-problem", po::value<bool>(&print_problem)->default_value(false), "Re-print the problem after parsing.")
    ("proof,p",  po::value<std::string>(&proof_file)->default_value(""), "Proof logging.")
    ("reduce,r",  po::value<bool>(&reduce)->default_value(false), "Only compute reduced form of last polynomial w.r.t. the previous ones.")
    ("threads,T", po::value<size_t>(&threads)->default_value(1), "Number of threads to be used.")
    ("tracer,t", po::value<bool>(&tracer)->default_value(true), "Whether computations with the first prime shall be traced. Speeds up the computation, but yields the correct result only with high probability.")
    ("verbosity,v", po::value<int>(&verbose)->default_value(1), "Set the verbosity level.")
  ;

  po::positional_options_description positional_desc;
  positional_desc.add("input", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
              .options(desc)
              .positional(positional_desc)
              .run(),
            vm);
  po::notify(vm);

  if(vm.count("help")) {
    desc.print(std::cout);
    return EXIT_SUCCESS;
  }

  if(vm.count("version")) {
    std::cout << F4NCGB_VERSION << std::endl;
    return EXIT_SUCCESS;
  }

  if(!std::filesystem::exists(input_name))
    die(err_no_file, "no input file given(try '-h')");

  int res = 0;

  std::filesystem::path in = input_name;

  parser_context context;
  solve_options opts;
  parse_res r;
  {
    F4NCGB_TIME(parse);
    context.open(in);
    r = context.parse_header(opts);
  }
  if(r.has_value()) {
    std::cerr << *r << std::endl;
    die(17, "Error in parsing input file.");
  }

  if(opts.num_blocks() > F4NCGB_MAX_BLOCKS)
    die(4, "More blocks than current compilation allows.");

  size_t characteristic = opts.characteristic;
  if(characteristic > 2147483647l)// 2^31 -1
    die(err_prime_too_big,
        "Provided characteristic %lu is too large. Only p < 2^31 supported",
        characteristic);
  if(characteristic != 0 and !n_is_prime(characteristic))
    die(err_no_prime,
        "Provided nonzero characteristic %lu is not prime.",
        characteristic);

  if(proof_file != "")
    user_proof_level = 1;

  if(user_proof_level == 0 and expanded_proof)
    die(78, "Flag for expanded proofs provided but no proof file");
  if(expanded_proof)
    user_proof_level = 2;

  opts.maxiter = maxiter;
  opts.maxdeg = maxdeg;
  opts.proof_level = user_proof_level;
  opts.tracer = tracer;
  opts.threads = threads;
  opts.proof_file = proof_file;
  opts.reduce = reduce;

  try {
    res = f4ncgb::f4ncgb_main(
      context,
      opts,
      { .output_name = output_name,
        .stats_print_function = &print_statistics_fun,
        // the process exits right after, so leaking memory is safe
        .leak_memory = true,
        .print_read_problem = print_problem });
  } catch(const std::exception& e) {
    die(20, "%s", e.what());
  }

  F4NCGB_PROFILE(gstats.print(threads));
  return res;
}
