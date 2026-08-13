#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include <sys/resource.h>
#include <sys/time.h>

#include "diagnostics.hpp"

int verbose = 1;
int msg_printing = 1;

void
msg(const char* fmt, ...) {
  if(!msg_printing)
    return;

  va_list ap;
  fputs("[f4ncgb] ", stderr);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  fflush(stderr);
}

void
die(int error_code, const char* fmt, ...) {
  fflush(stdout);
  va_list ap;
  fprintf(stderr, "*** [f4ncgb] error code %i \n", error_code);
  fputs("*** [f4ncgb] ", stderr);
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fputc('\n', stderr);
  fflush(stderr);
  exit(error_code);
}

// Everything below is the measuring half. This file is always compiled, so it
// has to disappear along with the statistics it prints.
#ifdef F4NCGB_ENABLE_PROFILING

namespace f4ncgb {
statistics gstats;

static size_t
maximum_resident_set_size() {
  struct rusage u;
  if(getrusage(RUSAGE_SELF, &u))
    return 0;
#ifdef __APPLE__
  // ru_maxrss is in bytes here, in kilobytes on Linux.
  return (size_t)u.ru_maxrss;
#else
  return ((size_t)u.ru_maxrss) << 10;
#endif
}

/*------------------------------------------------------------------------*/

static double
process_time() {
  struct rusage u;
  if(getrusage(RUSAGE_SELF, &u))
    return 0;
  double res = u.ru_utime.tv_sec + 1e-6 * u.ru_utime.tv_usec;
  res += u.ru_stime.tv_sec + 1e-6 * u.ru_stime.tv_usec;
  return res;
}

static double
percent(unsigned a, unsigned b) {
  return b ? 100.0 * a / b : 0;
}

static double
percent_d(double a, double b) {
  return b ? 100.0 * a / b : 0;
}
/*------------------------------------------------------------------------*/

/// One "<label> <seconds> (<share of the total> %)" line. Labels are padded to
/// the same width, sub-items are indented in the label itself.
static void
report(const char* label, double seconds, double total_time) {
  msg("%-27s%22.2f (%2.2f %%)", label, seconds, percent_d(seconds, total_time));
}

void
statistics::print(size_t threads) {
  auto total_time = process_time();
  statistics& s = gstats;
  msg("");
  msg("maximum resident set size:  %22.2f MB",
      maximum_resident_set_size() / static_cast<double>((1 << 20)));
  msg("store find calls:            %22zu", size_t(gstats.store_find_calls));
  msg("store find hits:           %22.2f %%",
      percent(s.store_find_hits, s.store_find_calls));

  const double elim_task_cpu = s.elim_task_cpu;

  report("parse input:", s.parse, total_time);
  report("computing ambiguities:", s.amb, total_time);
  report("  computing overlaps:", s.overlap, total_time);
  report("  computing inclusions:", s.inclusion, total_time);
  report("handling critical pairs:", s.crit_pair, total_time);
  report("symbolic preprocessing:", s.sym_pre, total_time);
  report("building the matrix:", s.matrix_setup, total_time);
  report("  sorting columns:", s.col_sort, total_time);
  report("  slot -> column remap:", s.col_remap, total_time);
  report("  sorting rows:", s.row_sort, total_time);
  report("linear algebra:", s.reduction, total_time);
  report("  Gauss elimination:", s.rref, total_time);
  msg("    reduce (CPU-time):     %22.2f (%2.2f %%) /#t: %.2f (%2.2f %%)",
      elim_task_cpu,
      percent_d(elim_task_cpu, total_time),
      elim_task_cpu / static_cast<double>(threads),
      percent_d(elim_task_cpu / static_cast<double>(threads), total_time));
  report("  CRT:", s.crt, total_time);
  report("    support union:", s.crt_support, total_time);
  report("  rat. reconstruction:", s.ratrec, total_time);
  msg("reconstruction rounds:       %22zu", size_t(s.crt_rounds));
  msg("  primes combined per round: %22zu", size_t(s.crt_prime_passes));
  report("construct new elements:", s.new_elements, total_time);
  report("other:", s.other, total_time);
  msg("");
  msg("total process time:        %22.2f seconds", process_time());
}
}
#endif
