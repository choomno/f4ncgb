cmake_minimum_required(VERSION 3.12)
project(f4ncgb VERSION 1.1.2)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

option(ENABLE_PROFILING "Enable tracking statistics" ON)
option(SHARED_LIB "Build the f4ncgb library shared instead of static" OFF)
option(ENABLE_FUZZER "Enable building the fuzzer, if available" OFF)
option(ENABLE_NATIVE "Enable -march=native to target the host CPU (creates non-portable binaries)" ON)
option(ENABLE_TEST "Build f4ncgb_test program" ON)
option(ENABLE_SIGNAL "Enable signal handling for printing statistics" ON)
option(WIDE_MATRIX_OFFSETS "Address matrices with more than 2^32 nonzeros, at the cost of 8 bytes instead of 4 per row for two offset arrays. Off by default; without it those counts are checked and the program stops rather than wrapping." OFF)

set(MAX_BLOCKS 8 CACHE STRING "Maximum number of blocks to compile for. Must be integral. Changes compilation time (duplicates implementation for each block).")

set(F4NCGB_SRCS
  src/api.cpp
  src/solve.cpp
  src/diagnostics.cpp
  src/parser.cpp
  src/monomial_trie.cpp
  src/primes.cpp
  src/linear_algebra.cpp
  src/zip_utils.cpp
)

set(F4NCGB_TEST_SRCS
  src/test.cpp
  src/test_simple.cpp
  src/test_parse.cpp
)

set(F4NCGB_TARGETS
  src/f4ncgb.cpp
)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/config.hpp")

if(isMultiConfig)
    if(NOT "Sanitize" IN_LIST CMAKE_CONFIGURATION_TYPES)
        list(APPEND CMAKE_CONFIGURATION_TYPES Sanitize)
    endif()
else()
    set(allowedBuildTypes Sanitize Debug Release RelWithDebInfo MinSizeRel)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${allowedBuildTypes}")

    if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE IN_LIST allowedBuildTypes)
        message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Sanitize")
  set(SKIP_MIMALLOC TRUE)
else()
  find_package(mimalloc QUIET)
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package( Threads QUIET)

# Do more optimizations in release mode. Also disable assertions.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -DNDEBUG")

if(ENABLE_NATIVE)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()

set(CMAKE_CXX_STANDARD 20)

if(TARGET mimalloc)
  list(APPEND F4NCGB_SRCS src/enable-mimalloc.cpp)
  message(STATUS "Found mimalloc and using it")
elseif(SKIP_MIMALLOC)
  message(STATUS "Sanitize build: not using mimalloc, it conflicts with the sanitizers")
else()
  message(STATUS "Not using mimalloc, but it gives you approx. 10-15% speedup. Consider installing its development package! See https://github.com/microsoft/mimalloc")
endif()

if(SHARED_LIB)
  message(STATUS "libf4ncgb is being built as a shared library")
  set(F4NCGB_LIBTYPE "SHARED")
else()
  message(STATUS "libf4ncgb is being built as a static library")
  set(F4NCGB_LIBTYPE "STATIC")
endif()

add_library(f4ncgb ${F4NCGB_LIBTYPE} ${F4NCGB_SRCS})
set_target_properties(f4ncgb PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/f4ncgb.hpp)

add_executable(f4ncgb_test ${F4NCGB_TEST_SRCS})
add_executable(f4ncgb_bin ${F4NCGB_TARGETS})
target_link_libraries(f4ncgb_test PUBLIC f4ncgb)
target_link_libraries(f4ncgb_bin PUBLIC f4ncgb)
set_target_properties(f4ncgb_bin PROPERTIES OUTPUT_NAME "f4ncgb")

if(SHARED_LIB)
  set_property(TARGET f4ncgb_bin PROPERTY INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()

if (NOT ENABLE_TEST)
   set_target_properties(f4ncgb_test PROPERTIES EXCLUDE_FROM_ALL 1)
endif()

# needed for pthread flag
target_link_libraries(f4ncgb PUBLIC ${CMAKE_THREAD_LIBS_INIT} )

set_property(TARGET f4ncgb PROPERTY CXX_STANDARD 20)
set_property(TARGET f4ncgb_test PROPERTY CXX_STANDARD 20)
set_property(TARGET f4ncgb_bin PROPERTY CXX_STANDARD 20)

if(TARGET mimalloc)
  target_link_libraries(f4ncgb PUBLIC mimalloc)
endif()

target_compile_definitions(f4ncgb PUBLIC F4NCGB_MAX_BLOCKS=${MAX_BLOCKS})
message(STATUS "Compiling with " ${MAX_BLOCKS} " maximum blocks")

if(WIDE_MATRIX_OFFSETS)
  message(STATUS "Matrix offsets are 64 bit")
  target_compile_definitions(f4ncgb PUBLIC F4NCGB_WIDE_MATRIX_OFFSETS)
else()
  message(STATUS "Matrix offsets are 32 bit (see WIDE_MATRIX_OFFSETS)")
endif()

if(ENABLE_SIGNAL)
   message(STATUS "Signal handling is enabled")
   target_compile_definitions(f4ncgb_bin PUBLIC F4NCGB_ENABLE_SIGNAL)
else()
   message(STATUS "Signal handling is disabled")
endif()

if(CMAKE_BUILD_TYPE MATCHES "Sanitize")
  target_compile_options(f4ncgb PRIVATE -fsanitize=address -fsanitize=undefined -g)
  target_link_options(f4ncgb PRIVATE -fsanitize=address -fsanitize=undefined)
  target_compile_options(f4ncgb_bin PRIVATE -fsanitize=address -fsanitize=undefined -g)
  target_link_options(f4ncgb_bin PRIVATE -fsanitize=address -fsanitize=undefined)
  target_compile_options(f4ncgb_test PRIVATE -fsanitize=address -fsanitize=undefined -g)
  target_link_options(f4ncgb_test PRIVATE -fsanitize=address -fsanitize=undefined)

  target_link_libraries(f4ncgb_test PRIVATE -fsanitize=address -fsanitize=undefined)
  target_link_libraries(f4ncgb_bin PRIVATE -fsanitize=address -fsanitize=undefined)
endif()

find_package(Boost COMPONENTS unit_test_framework program_options REQUIRED)
find_package(GMP REQUIRED)
find_package(Flint REQUIRED)

target_compile_options(f4ncgb PRIVATE
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall -Wextra -Wsign-conversion>
     $<$<CXX_COMPILER_ID:MSVC>:
          /W4>)

target_compile_options(f4ncgb_test PRIVATE
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall -Wextra -Wsign-conversion>
     $<$<CXX_COMPILER_ID:MSVC>:
          /W4>)

target_compile_options(f4ncgb_bin PRIVATE
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
          -Wall -Wextra -Wsign-conversion>
     $<$<CXX_COMPILER_ID:MSVC>:
          /W4>)

target_link_libraries(f4ncgb PUBLIC Boost::boost GMP::GMP)
target_link_libraries(f4ncgb_test PUBLIC Boost::unit_test_framework)
target_link_libraries(f4ncgb PUBLIC Flint::Flint Boost::boost GMP::GMP)
target_link_libraries(f4ncgb_bin PUBLIC Boost::boost Boost::program_options)

target_include_directories(f4ncgb PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(f4ncgb_bin PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(ENABLE_PROFILING)
  target_compile_definitions(f4ncgb PUBLIC F4NCGB_ENABLE_PROFILING)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

if(ipo_supported AND CMAKE_BUILD_TYPE MATCHES "Release")
  message(STATUS "IPO / LTO enabled (Release build and compiler supports it)")
  # The library holds the hot code and f4ncgb_bin is what users run, so both
  # need this at least as much as the test binary does.
  set_property(TARGET f4ncgb f4ncgb_bin f4ncgb_test
               PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
  if(CMAKE_BUILD_TYPE MATCHES "Release")
    message(STATUS "IPO / LTO not supported: <${ipo_error}>")
  else()
    message(STATUS "IPO / LTO not activated, build in Release mode to activate")
  endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fuzztest/CMakeLists.txt" AND ENABLE_FUZZER)
  # Create the fuzzing target only if the fuzztest subdirectory
  # exists.
  message(STATUS "Fuzzer enabled")
  set(FUZZTEST_FUZZING_MODE ON)
  add_subdirectory(fuzztest)
  fuzztest_setup_fuzzing_flags()

  add_executable(
    f4ncgb_fuzz
    src/fuzzer.cpp
  )

  link_fuzztest(f4ncgb_fuzz)
  target_link_libraries(f4ncgb_fuzz PUBLIC f4ncgb)
  target_compile_options(f4ncgb_fuzz PRIVATE -g)

  if(CMAKE_BUILD_TYPE MATCHES "Sanitize")
    target_link_options(f4ncgb_fuzz PRIVATE -fsanitize=address -fsanitize=undefined)
    target_link_libraries(f4ncgb_fuzz PRIVATE -fsanitize=address -fsanitize=undefined)
  endif()
else()
  message(STATUS "Fuzzer disabled")
endif()

include(GNUInstallDirs)

install(TARGETS f4ncgb_bin)
install(TARGETS f4ncgb
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/f4ncgb)
