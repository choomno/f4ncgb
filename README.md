# f4ncgb

`f4ncgb` ("fancy-gb") is an open-source C++ library for computing noncommutative Gröbner bases
in free algebras (with rational coefficients or coefficients in a prime field). 
It transfers recent advancements in commutative Gröbner basis software to the
noncommutative setting.

Below is a runtime comparison (in seconds) of `f4ncgb` (version 1.1.2) with the state-of-the-art tool [Singular](https://www.singular.uni-kl.de/index.php.html).
All benchmarks are part of the [SymbolicData project](https://symbolicdata.github.io).

| Example | Singular | f4ncgb (1 core) | f4ncgb (2 cores) | f4ncgb (4 cores) | f4ncgb (8 cores) |
| --- | --- | --- | --- | --- | --- |
| 4nilp5s-10 | 1 217.19 | 29.51 | 20.56 | 17.00 | 13.61 |
| braid3-16 | 18 314.52 | 9.89 | 6.69 | 4.85 | 3.90 |
| braidX-18 | >43 200 | 139.04 | 78.61 | 53.42 | 37.23 |
| braidXY-12 | 1 608.16 | 10.52 | 8.19 | 6.98 | 6.56 |
| holt_G3562h-17 | >43 200 | 43.26 | 39.52 | 38.48 | 37.26 |
| lascala_neuh-13 | 166.95 | 0.99 | 0.97 | 1.02 | 0.93 |
| lv2d10-100 | >43 200 | 9.56 | 10.40 | 9.55 | 9.87 |
| malle_G12h-100 | 3 934.29 | 35.66 | 36.33 | 35.06 | 35.89 |
| serre-e6-17 | 107.46 | 40.51 | 41.36 | 41.23 | 41.64 |
| serre-ha11-17 | 94.85 | 9.04 | 8.18 | 7.70 | 8.28 |


# Table of Contents

[[_TOC_]]

News about current updates and new features may be found in
[NEWS.md](NEWS.md).

# Compilation

## Dependencies

Please install the following required dependencies:

-   [Boost](https://www.boost.org/) (tested with 1.83, at least 1.80, Apt Package `libboost-all-dev`)
-   [Flint](https://flintlib.org/) (tested with 3.0.1, Apt Package `libflint-dev`)
-   [GMP](https://gmplib.org/) (tested with 6.3.0, Apt Package `libgmp-dev`)

And recommended:

-   [Mimalloc](https://github.com/microsoft/mimalloc) (for more efficient memory allocation, tested with 2.1.2,
    Apt Package `libmimalloc-dev`)

The fuzzer can be used by cloning [FuzzTest](https://github.com/google/fuzztest) into the project's root
directory.

## Compilation using CMake

Please use [CMake](https://cmake.org/) to compile this project. All dependencies are
automatically found:

    git clone https://gitlab.sai.jku.at/f4ncgb/f4ncgb.git
    mkdir build-release
    cd build-release
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j

Then, `f4ncgb` and `f4ncgb_test` are produced.

The release build automatically optimizes using `-march=native`, so
the resulting binary may not run on a machine that does not support
the same instructions as the builder.

# Basic usage of `f4ncgb`

    ./f4ncgb <input.ms>

These are the provided options (also visible from `--help`:

-   **`--version`:** Produce version message
-   **`--help`:** Produce help message
-   **`--expanded-proof`, `-e`:** Whether the proofs should be expanded
    and written in terms of the input. Requires proof logging to be
    turned on
-   **`--input`, `-i`:** Set the input file (either msolve or poly)
-   **`--maxiter`, `-m`:** Maximal number of iterations of the
    F4-algorithm to be performed (default: 10)
-   **`--maxdeg`, `-d`:** Maximal degree of ambiguities that are
    considered (default: INT<sub>MAX</sub>)
-   **`--output`, `-o`:** Set the output file (default: std::cout)
-   **`--print-problem`:** Re-print the problem after parsing.
-   **`--proof`, `-p`:** Proof logging (default: disabled)
-   **`--threads`, `-T`:** Number of threads to be used (default: 1)
-   **`--tracer`, `-t`:** Whether computations with the first prime
    shall be traced. Speeds up the computation, but yields the correct
    result only with high probability (default: enabled)
-   **`--verbosity`, `-v`:** Set the verbosity level


## `msolve` Input Format

This format is compatible to the input format of [msolve](https://msolve.lip6.fr/). We provide
a simplified EBNF below. See also the files in `test_inputs` for examples.

    input          := var_list EOL
                      [ weights EOL ]
                      characteristic EOL
                      poly_list EOF .
    
    var_list       := ident { "," ident } .
    
    weights        := "w" nat { nat } .
    
    characteristic := int .
    
    poly_list      := poly { "," EOL poly } .
    
    poly           := ["+","-"] [ factor ] mon { ["+","-"] factor mon } .
    
    mon            := ident [ exp ] { "*" ident [ exp ] } .
    
    factor         := int [ "/" int ] .
    
    exp            := "^" int .
    
    ident          := letter { letter | digit } .
    
    int            := digit { digit } .
    
    letter         := "A"..."Z" | "a"..."z" | ... .
    
    digit          := "0"..."9" .

## `poly` Input Format

Poly is a simplified format made for machine generation, inspired by the 
DIMACS format that SAT-solvers use. We provide a simplified EBNF below.

    input          := "p poly " nr_of_vars nr_of_blocks characteristic EOL
                      block { EOL block } EOL
                      [ weights EOL ]
                      poly { EOL poly } EOF .
    
    nr_of_vars     := nat .
    
    nr_of_blocks   := nat .
    
    characteristic := nat .
    
    block          := nat { nat } "0" .
    
    weights        := "w" nat { nat } .
    
    poly           := mon { "+" mon } 0 .
    
    mon            := numerator denominator var { var } .
    
    numerator      := int .
    
    denominator    := nat .
    
    var            := nat .
    
    int            := ["+","-"] nat .
    
    nat            := digit { digit } .
    
    digit          := "0"..."9" .

## Weighted orders

Both formats accept an optional `w` line carrying one positive integer weight per variable, in the order the variables are numbered. 

    x,y,z
    w 1 1 3
    0
    ...

(Per default, all weights are 1.)

# Tests

    ./f4ncgb_test

This should produce the following output:

    Running 30 test cases...
     
    *** No errors detected

Internally, the tests also create our store structure multiple times
and test the memory allocation to fit our requirements.

# Additional features

## Proof logging

With `-p` the solver produces a cofactor representation `sum of c*a*[k]*b` for each basis element.
A bracketed index is either

-   `[k]`, meaning basis element `k`, or
-   `[ik]`, meaning input polynomial `k`.

Indexing starts at 0.    

A plain `-p` proof uses both forms, `-e` expands the whole proof back to the input, so only `[ik]` appears.

## Optimization for Mersenne Primes

We provide a specialized implementation for computing modular images modulo Mersenne primes,
used for the first prime `2^31-1` of the multi-modular Gaussian elimination.

Some examples of the Mersenne Prime optimization speedups with
different compilers and processors, each tuned to the native
instruction set level using `-march=native` can be seen in the table
below. We sorted the table according to the achieved speedup. While
the optimized variant is slower for Clang, GCC produces binaries that
overall are faster than the ones from Clang.

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">

<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-right" />

<col  class="org-right" />

<col  class="org-right" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">CPU</th>
<th scope="col" class="org-left">Compiler</th>
<th scope="col" class="org-right">Standard [s]</th>
<th scope="col" class="org-right">Optimized [s]</th>
<th scope="col" class="org-right">Speedup [Standard/Optimized]</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left">Intel i7-10700</td>
<td class="org-left">GCC 15.1.1 20250425 (Red Hat 15.1.1-1)</td>
<td class="org-right">11.685667</td>
<td class="org-right">1.311982</td>
<td class="org-right">8.9068806</td>
</tr>

<tr>
<td class="org-left">Intel Xeon w5-2455X</td>
<td class="org-left">GCC Ubuntu 13.3.0</td>
<td class="org-right">4.709925</td>
<td class="org-right">1.177109</td>
<td class="org-right">4.0012650</td>
</tr>

<tr>
<td class="org-left">AMD EPYC 7443P</td>
<td class="org-left">GCC Ubuntu 13.3.0</td>
<td class="org-right">3.757219</td>
<td class="org-right">1.072234</td>
<td class="org-right">3.5041036</td>
</tr>

<tr>
<td class="org-left">AMD EPYC 7443P</td>
<td class="org-left">Clang 18.1.3 (1ubuntu1)</td>
<td class="org-right">3.754652</td>
<td class="org-right">1.072636</td>
<td class="org-right">3.5003972</td>
</tr>

<tr>
<td class="org-left">AMD EPYC 7313</td>
<td class="org-left">Clang 18.1.3 (1ubuntu1)</td>
<td class="org-right">4.057575</td>
<td class="org-right">1.159551</td>
<td class="org-right">3.4992639</td>
</tr>

<tr>
<td class="org-left">AMD EPYC 7313</td>
<td class="org-left">GCC Ubuntu 13.3.0</td>
<td class="org-right">4.057734</td>
<td class="org-right">1.159754</td>
<td class="org-right">3.4987885</td>
</tr>

<tr>
<td class="org-left">Intel i7-10700</td>
<td class="org-left">Clang 20.1.3 (Fedora 20.1.3-1.fc42)</td>
<td class="org-right">11.623475</td>
<td class="org-right">8.451767</td>
<td class="org-right">1.3752716</td>
</tr>

<tr>
<td class="org-left">Intel Xeon w5-2455X</td>
<td class="org-left">Clang 18.1.3 (1ubuntu1)</td>
<td class="org-right">4.708247</td>
<td class="org-right">3.460993</td>
<td class="org-right">1.3603746</td>
</tr>

<tr>
<td class="org-left">MacBookAir M2 (ARM64)</td>
<td class="org-left">AppleClang 16</td>
<td class="org-right">1.263310</td>
<td class="org-right">1.491293</td>
<td class="org-right">0.84712394</td>
</tr>
</tbody>
</table>

# Citing `f4ncgb`

If you have used `f4ncgb` for a paper, we would kindly ask you to cite it as:

    Maximilian Heisinger and Clemens Hofstadler.    
    f4ncgb: High Performance Gröbner Basis Computations in Free Algebras. 
    In: Computer Algebra in Scientific Computing. CASC 2025. 
    Lecture Notes in Computer Science, vol 16235. Springer, 2025.

Or, if you are using BibTeX:

    @InProceedings{f4ncgb,
    author="Heisinger, Maximilian and Hofstadler, Clemens",
    editor="Boulier, Fran{\c{c}}ois and Mou, Chenqi and Sadykov, Timur M. and Vorozhtsov, Evgenii V.",
    title="f4ncgb: High Performance Gr{\"o}bner Basis Computations in Free Algebras",
    booktitle="Computer Algebra in Scientific Computing (CASC) 2025",
    year="2025",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="79--97"}


The paper can be found [here](https://doi.org/10.1007/978-3-032-09645-6_5).

