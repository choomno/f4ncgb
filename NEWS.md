v1.1.1
==========

## Improvements

  - Improvements to cofactor expansion for `-e` flag.

## Fixes

  - Fix a crash when computing a normal form against no reducers, i.e. when the
    input to `--reduce` holds a single polynomial.

v1.1.0
======

## Improvements

  - Mask for nonzero entries in Gaussian elimination
  - Improvements to matrix setup
  - Improvements to the symbolic preprocessing
  - Improvements to the monomial trie data structure
  - More fine-grained parallelisation of linear algebra

## New Features

  - The input is validated: a variable index outside `1..num_vars`, and a block
    specification that does not cover the variables in order, are reported as
    errors.
  - Add CMake-option `WIDE_MATRIX_OFFSETS` for matrices with more than 2^32
    nonzero entries. Off by default, in which case the counts that could overflow are
    checked and the computation stops.

## Fixes

  - Fix for expanded cofactor representations
  - Block orders were ignored for the `poly` format
  - Degree bound was ignored for inclusion ambiguities 

v1.0.1
======

## Fixes
  - Fix bug from not resetting global variables

v1.0.0
======

## New Features
  - Proper version of reduced_form
  
## Improvements
  - Better internal coefficient handling (FLINT instead of GMP)
  - More efficient matrix data structures
  - refactored several bottlenecks 
  - several bugfixes

v0.3.4
======

## Fixes
  - Fix bug in cofactor representations
  
v0.3.3
======

## Fixes
  - Fix bug in reduced_form

v0.3.2
======

## New Features

  - Rudimentary version of reduced_form.

## Fixes
  - Bugfix: stop computation when 1 is in the Gröbner basis

v0.3.1
======

## Fixes

  - Fix an allocation issue on some memory-constrained systems by iteratively
    decreasing the requested memory space.

v0.3
====

## New Features

  - Introduce an API (C-based) in `f4ncgb.hpp` that can start basis
    calculations and return the computed basis.
  - Add a handler for SIGUSR1 that prints the basis that has already
    been computed up to this point.
  - Add support for shared library builds.
  - Add support for dumping the stores at destruction (enable with
    CMake-option `-DENABLE_STORE_DUMP=ON`).
  - Add CMake-options `ENABLE_TEST` and `ENABLE_SIGNAL` for toggling
    the generation of the test binary and inclusion of signal-handler
    code in the f4ncgb binary. (thanks to Benjamin Lorenz)
  - Fix compilation issues on MacOS and with pthreads (thanks to
    Benjamin Lorenz)

## Changes

  - Both `proof_level` and `tracer` are no longer global symbols,
    instead they are given as parameters to a solve call.

v0.2.1
======

## New Features

  - Add an install target for installing the future API header, the
    binary, and the library.

v0.2
====

## Fixes

  - Fix a test case that was broken because of a file rename.
