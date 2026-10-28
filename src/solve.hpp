#pragma once

#include <functional>
#include <string>

#include "f4ncgb.hpp"
#include "options.hpp"

namespace f4ncgb {

class parser_context;

struct run_options {
  // File the basis is written to. Empty writes to stdout, unless the callbacks
  // below are set, in which case the basis is handed to them instead.
  std::string output_name;
  // Optional caller-owned basis destination. C API callbacks take priority.
  std::ostream* output_stream = nullptr;

  // Set to the callbacks by the C API; both or neither.
  void* userdata = nullptr;
  f4ncgb_add_cb add_cb = nullptr;
  f4ncgb_end_poly_cb end_poly_cb = nullptr;

  // Filled in with a function that prints the basis computed so far. The
  // command line calls it from its SIGUSR1 handler.
  std::function<void()>* stats_print_function = nullptr;

  // Skip deleting the stores. Only safe when the process exits right
  // after, which is why the command line sets it and the library does not.
  bool leak_memory = false;

  // Echo the problem back after parsing it.
  bool print_read_problem = false;
};

// Runs one computation: picks the instantiation matching the number of blocks
// and variables, reads the input, computes, and writes the result out.
int
f4ncgb_main(parser_context& context,
            const solve_options& opts,
            const run_options& run);
}
