#pragma once

#include <chrono>
#include <memory>
#include <string>

namespace f4ncgb {
class store_tracer {
  static std::string prefix_;
  struct FILE_deleter {
    void operator()(FILE* f) { fclose(f); }
  };
  std::unique_ptr<FILE, FILE_deleter> file_;
  std::chrono::steady_clock::time_point start_
    = std::chrono::steady_clock::now();

  public:
  static void set_prefix(const std::string& prefix);

  store_tracer(const std::string& name);
  ~store_tracer();

  void access(unsigned long i);
};
}
