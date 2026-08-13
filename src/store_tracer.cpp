#include "store_tracer.hpp"
#include "zip_utils.hpp"
#include <chrono>
#include <cstdio>

namespace f4ncgb {
std::string store_tracer::prefix_ = "trace";

void
store_tracer::set_prefix(const std::string& prefix) {
  prefix_ = prefix;
}

store_tracer::store_tracer(const std::string& name) {
  file_.reset(popen_with_found_bin_write(
    "zstd", "- -qfo \"" + prefix_ + "_" + name + ".zst\""));
}
store_tracer::~store_tracer() {}

void
store_tracer::access(unsigned long i) {
  auto now = std::chrono::steady_clock::now();
  auto stamp
    = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_)
        .count();

  fprintf(file_.get(), "%lu %lld\n", i, static_cast<long long>(stamp));
}
}
