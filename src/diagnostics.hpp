#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>

// Level of output verbosity, ranges from 0 to 4
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

  std::chrono::time_point<std::chrono::steady_clock> start_time
    = std::chrono::steady_clock::now();

  // Times
  accumulator init, other, new_elements;
  accumulator amb, crit_pair, sym_pre, matrix_setup, reduction;
  accumulator overlap, inclusion, crt, ratrec, rref;
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

      // The addition is only carried out once! 
      // The destructor may be called multiple times.
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
