#pragma once

#include <string>

namespace f4ncgb {

// The database protocol contains algebra input and output bytes only. The caller
// remains responsible for its request identity and mathematical verification.
class sqlite_job {
  std::string database_;
  std::string job_id_;
  std::string input_;

public:
  sqlite_job(std::string database, std::string job_id);
  const std::string& input() const { return input_; }
  void complete(const std::string& basis, const std::string& proof) const;
};

}
