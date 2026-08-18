#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "zip_utils.hpp"

namespace f4ncgb {
static int xzsig[] = { 0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00, EOF };
static int bz2sig[] = { 0x42, 0x5A, 0x68, EOF };
static int gzsig[] = { 0x1F, 0x8B, EOF };
static int sig7z[] = { 0x37, 0x7A, 0xBC, 0xAF, 0x27, 0x1C, EOF };
static int lzmasig[] = { 0x5D, 0x00, 0x00, 0x80, 0x00, EOF };
static int zstdsig[] = { 0x28, 0xb5, 0x2f, 0xfd, EOF };

bool
file_matches_signature(std::filesystem::path path, int* sig) {
  std::ifstream in(path, std::ios::binary);
  for(int* c = sig; *c != EOF; ++c) {
    unsigned char r;
    in >> r;
    if(*c != r)
      return false;
  }
  return true;
}

std::string
find_in_path(std::string command) {
  using namespace std::string_literals;

  std::string path = std::getenv("PATH");
  path += ":";

  size_t pos = 0;
  std::string result;
  while((pos = path.find(":")) != std::string::npos) {
    result = path.substr(0, pos);
    result += "/" + command;
    if(std::filesystem::exists(result)) {
      return result;
    }
    path = path.substr(pos + 1);
  }

  throw std::runtime_error("Could not find "s + command + " in PATH!");

  // This is just to make the compiler happy.
  exit(EXIT_FAILURE);
}

FILE*
popen_with_found_bin_write(const std::string& command,
                           const std::string& args) {
  using namespace std::string_literals;

  std::string abs_command = find_in_path(command);
  std::string cmd = abs_command + " " + args;
  FILE* handle = popen(cmd.c_str(), "w");
  if(!handle) {
    throw std::runtime_error("Coult not popen()! Error: "s + strerror(errno));
  }
  return handle;
}

FILE*
popen_with_found_bin(const std::string& command,
                     const std::string& args,
                     std::filesystem::path path) {
  using namespace std::string_literals;

  std::string abs_command = find_in_path(command);
  std::string cmd = abs_command + " " + args + " \"" + path.string() + "\"";
  FILE* handle = popen(cmd.c_str(), "r");
  if(!handle) {
    throw std::runtime_error("Coult not popen()! Error: "s + strerror(errno));
  }
  return handle;
}

FILE*
open_compressed_or_direct(std::filesystem::path p) {
  FILE* f_ptr = nullptr;
  if(p.extension() == ".xz" && file_matches_signature(p, xzsig))
    f_ptr = popen_with_found_bin("xz", "-c -d", p);
  else if(p.extension() == ".lzma" && file_matches_signature(p, lzmasig))
    f_ptr = popen_with_found_bin("lzma", "-c -d", p);
  else if(p.extension() == ".bz2" && file_matches_signature(p, bz2sig))
    f_ptr = popen_with_found_bin("bzip2", "-c -d", p);
  else if(p.extension() == ".gz" && file_matches_signature(p, gzsig))
    f_ptr = popen_with_found_bin("gzip", "-c -d", p);
  else if(p.extension() == ".zst" && file_matches_signature(p, zstdsig))
    f_ptr = popen_with_found_bin("zstd", "-c -q -d", p);
  else if(p.extension() == ".7z" && file_matches_signature(p, sig7z))
    f_ptr = popen_with_found_bin("7z", "x -so", p);
  else
    f_ptr = std::fopen(p.string().c_str(), "r");

  return f_ptr;
}
}
