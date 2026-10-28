# Optional SQLite input and output

This fork can read an algebra presentation from a SQLite BLOB and write its
basis and proof to the same database. The parser and mathematical computation
are shared with the file interface. No algebra-input, basis or proof files are
created in this mode.

## Build

SQLite I/O is disabled by default. To enable it, install the SQLite development
package (`libsqlite3-dev` on Debian/Ubuntu or `sqlite-devel` on Fedora), and use
CMake 3.14 or newer:

```sh
cmake -S . -B build-sqlite -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_NATIVE=OFF -DENABLE_SQLITE=ON
cmake --build build-sqlite --parallel 2
./build-sqlite/f4ncgb --features
```

The final command prints `{"sqlite":true}`. A normal build prints
`{"sqlite":false}` and needs no SQLite headers or library. The existing
`--version` output is unchanged.

## Database protocol

The caller creates this table and inserts a ready job:

```sql
CREATE TABLE f4ncgb_io (
    job_id TEXT PRIMARY KEY,
    input BLOB NOT NULL,
    basis BLOB,
    proof BLOB,
    state TEXT NOT NULL CHECK(state IN('ready','complete'))
);
```

The database may contain other application tables. `f4ncgb_io` must use exactly
the declared table definition and have no triggers. ASCII case and inter-token
whitespace may differ; SQL string literals must remain exact. Extra constraints, generated columns,
missing checks and other table-definition changes are rejected. Store the exact input bytes as a BLOB in
`input`, set `state='ready'`, and leave `basis` and `proof` NULL. Use a distinct
job id for each attempt. For example, with Python's standard library:

```python
import sqlite3

with sqlite3.connect("jobs.sqlite3") as connection:
    connection.execute("""CREATE TABLE f4ncgb_io (
        job_id TEXT PRIMARY KEY,input BLOB NOT NULL,basis BLOB,proof BLOB,
        state TEXT NOT NULL CHECK(state IN('ready','complete')))""")
    connection.execute(
        "INSERT INTO f4ncgb_io VALUES (?,?,NULL,NULL,'ready')",
        ("example", b"x,y\n0\nx*x-x,\ny*y-y,\nx*y-y*x\n"),
    )
```

Then run:

```sh
./build-sqlite/f4ncgb --sqlite jobs.sqlite3 --sqlite-job example -m 100
```

The input BLOB accepts the same `msolve` and `poly` syntax as the file parser.
Database mode records nonexpanded proofs by default. `--expanded-proof 1`
selects expanded proofs. Computation flags such as iteration limits, degree
limits, threading, reduction and tracing retain their usual meanings.
`--input`, `--output` and `--proof` filenames cannot be combined with SQLite
mode. Logs and statistics remain on the standard output/error streams.

On success, the row has `state='complete'` and BLOBs in `basis` and `proof`:

```python
with sqlite3.connect("jobs.sqlite3") as connection:
    basis, proof = connection.execute(
        "SELECT basis,proof FROM f4ncgb_io WHERE job_id=? AND state='complete'",
        ("example",),
    ).fetchone()
```

The database contains the ordinary basis/proof serialization bytes. A caller
can inspect, export or independently verify them without a custom binary codec.
The protocol contains no application-specific presentation names, sampling
policy or provenance assertions.

## Publication and interpretation

The solver opens the existing database for reading and writing, so SQLite can
recover a hot rollback journal after an interrupted writer. It enables
connection-level query-only mode for the input snapshot, reads the input
in a short transaction, closes that connection and computes without holding a
database lock. It publishes basis,
proof and state together in one write transaction. Completion is conditional
on the same job still being ready, retaining the original input bytes and
having no output values. If another writer changes that job or completes it
first, publication fails without overwriting the other result. Independent jobs
can run concurrently; SQLite coordinates their short publication transactions.

The solver's connections explicitly set `PRAGMA foreign_keys=OFF`. The protocol
has no foreign-key semantics, and native completion does not validate application
foreign keys or invoke their cascades. Application tables must not rely on
foreign-key actions involving `f4ncgb_io`; update dependent application metadata
in the caller. This also prevents a cascade from invoking a trigger on another
table that changes solver outputs or deletes another job. Triggers directly on
the protocol table remain unsupported.

Basis and proof text are buffered in memory until publication. File mode remains
available for outputs that exceed the available memory or SQLite's BLOB limits.

Parsing errors, computation failures and publication errors leave the ready row
without partial outputs. A completed job is not overwritten on a repeated call.
The caller owns cleanup of abandoned ready jobs.

`complete` means that this invocation finished and wrote its outputs. It does
not assert that an iteration- or degree-limited basis is complete, or that a
separate target belongs to its ideal. Those mathematical conclusions require
the corresponding checks. Source provenance, executable identification and
application-level proof validation remain the caller's responsibility.

## Tests

```sh
./build-sqlite/f4ncgb_test --log_level=test_suite
python3 tests/sqlite_io.py --binary build-sqlite/f4ncgb
```

The native tests cover input changes, competing writers and rollback. The
process tests compare file and database basis/proof bytes over the rationals
and a finite field, including expanded proofs, and check failed inputs, option
conflicts, independent concurrent jobs and read-only databases. The same script
checks a disabled build with `--expect-disabled`.
