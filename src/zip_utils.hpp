#pragma once

#include <cstdio>
#include <filesystem>
#include <string>

namespace f4ncgb {
bool
file_matches_signature(std::filesystem::path path, int* sig);

std::string
find_in_path(std::string command);

FILE*
popen_with_found_bin(const std::string& command,
                     const std::string& args,
                     std::filesystem::path path);

FILE*
popen_with_found_bin_write(const std::string& command, const std::string& args);

FILE*
open_compressed_or_direct(std::filesystem::path p);
}
