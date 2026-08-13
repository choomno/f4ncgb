#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>

/// How the program reports to the user, and how it measures itself.
///
/// The reporting half is unconditional: die() is the error path for argument
/// validation and msg() writes everything the user sees on stderr. The
/// measuring half compiles out entirely when profiling is off, so nothing
/// outside this header should name gstats -- go through the two macros.

/// Level of output verbosity, ranges from 0 to 4
extern int verbose;
extern int msg_printing;

/**
    Prints an error message to stderr and exits the program

    @param char* fmt message
    @param int error_code
*/
void
die(int error_code, const char* fmt, ...);

/**
    Prints a message to stderr

    @param char* fmt message
*/
void
msg(const char* fmt, ...);

#ifdef F4NCGB_ENABLE_PROFILING
#define F4NCGB_PROFILE(EXPR) EXPR
#define F4NCGB_TIME(TGT) \
  auto timer = ::f4ncgb::gstats.time(::f4ncgb::gstats.TGT)
#else
#define F4NCGB_PROFILE(EXPR)
#define F4NCGB_TIME(EXPR)
#endif

namespace f4ncgb {
#ifdef F4NCGB_ENABLE_PROFILING

/// The statistics are global, so they are written by every elimination task and
/// by every concurrent solve. Relaxed atomics keep that race-free; ordering
/// does not matter for a counter nobody reads until the end.
struct counter {
  std::atomic<size_t> value{ 0 };

  void operator++(int) { value.fetch_add(1, std::memory_order_relaxed); }
  operator size_t() const { return value.load(std::memory_order_relaxed); }
};

struct accumulator {
  std::atomic<double> value{ 0 };

  void add(double d) { value.fetch_add(d, std::memory_order_relaxed); }
  operator double() const { return value.load(std::memory_order_relaxed); }
};

struct statistics {
  counter store_find_calls, store_find_hits;

  /// How much of the multi-modular reconstruction is redone. A round that fails
  /// to reconstruct is thrown away and repeated with more primes, so comparing
  /// the prime passes against the primes the result actually needed says how
  /// much of the CRT was spent twice.
  counter crt_rounds, crt_prime_passes;

  // Times
  accumulator init, other, new_elements;
  accumulator amb, crit_pair, sym_pre, matrix_setup, reduction;
  accumulator overlap, inclusion, crt, crt_support, ratrec, rref;
  accumulator col_sort, col_remap, row_sort;
  accumulator elim_task_cpu;
  accumulator parse;

  // Automated timing to add time (in seconds) to some target.
  struct adding_timer {
    adding_timer(accumulator& tgt)
      : target(&tgt) {
      start = std::chrono::high_resolution_clock().now();
    }
    ~adding_timer() {
      if(!target)
        return;

      // The addition is only carried out once! The destructor may be called
      // multiple times.
      std::chrono::time_point<std::chrono::high_resolution_clock> end
        = std::chrono::high_resolution_clock().now();
      std::chrono::duration<double> elapsed = end - start;
      target->add(elapsed.count());
      target = nullptr;
    };

    adding_timer(const adding_timer& o) = delete;
    adding_timer& operator=(const adding_timer& o) = delete;

    private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    accumulator* target;
  };

  adding_timer time(accumulator& tgt) { return adding_timer(tgt); }

  void print(size_t threads = 1);
};
extern statistics gstats;
#endif
}
