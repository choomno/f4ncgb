"""Small native file/SQLite parity and publication regressions."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest


DDL = ("CREATE TABLE f4ncgb_io (job_id TEXT PRIMARY KEY,input BLOB NOT NULL,"
       "basis BLOB,proof BLOB,state TEXT NOT NULL CHECK(state IN('ready','complete')))")
BINARY = None


class NativeSQLite(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="f4ncgb-sqlite-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / "jobs.sqlite3"
        self.work = self.root / "work"
        self.work.mkdir()
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(DDL)
            connection.commit()

    def invoke(self, *arguments):
        return subprocess.run([str(BINARY), *arguments], cwd=self.work,
                              capture_output=True, timeout=30)

    def enqueue(self, key, source):
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("INSERT INTO f4ncgb_io VALUES (?,?,NULL,NULL,'ready')", (key, source))
            connection.commit()

    def record(self, key):
        with closing(sqlite3.connect(self.database)) as connection:
            return connection.execute("SELECT input,basis,proof,state FROM f4ncgb_io WHERE job_id=?",
                                      (key,)).fetchone()

    def solve(self, key, *flags):
        return self.invoke("--sqlite", str(self.database), "--sqlite-job", key,
                           "-m", "100", "-v", "0", *flags)

    def test_features_and_existing_version_output(self):
        result = self.invoke("--features")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"sqlite": True})
        self.assertRegex(self.invoke("--version").stdout.decode(), r"^\d+\.\d+\.\d+\s*$")

    def test_file_and_database_basis_and_proof_bytes_match(self):
        for characteristic in (0, 101):
            for expanded in (0, 1):
                with self.subTest(characteristic=characteristic, expanded=expanded):
                    source = (f"x,y\n{characteristic}\n1/2*x*x-x,\ny*y-y,\nx*y-y*x\n").encode()
                    key = f"field-{characteristic}-proof-{expanded}"
                    self.enqueue(key, source)
                    folder = self.root / key
                    folder.mkdir()
                    input_file, basis, proof = (folder / name for name in ("input.ms", "basis.txt", "proof.txt"))
                    input_file.write_bytes(source)
                    file_result = self.invoke("-i", str(input_file), "-o", str(basis), "-p", str(proof),
                                              "-e", str(expanded), "-m", "100", "-v", "0")
                    self.assertEqual(file_result.returncode, 0, file_result.stderr)
                    result = self.solve(key, "-e", str(expanded))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    original, database_basis, database_proof, state = self.record(key)
                    self.assertEqual(original, source)
                    self.assertEqual(database_basis, basis.read_bytes())
                    self.assertEqual(database_proof, proof.read_bytes())
                    self.assertEqual(state, "complete")
                    self.assertEqual(list(self.work.iterdir()), [])

    def test_single_variable_and_crlf_database_inputs(self):
        for index, source in enumerate((b"x\n0\nx*x-x\n", b"x,y\r\n7\r\nx*y-y\r\n")):
            key = f"format-{index}"
            self.enqueue(key, source)
            result = self.solve(key)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.record(key)[0], source)
            self.assertEqual(self.record(key)[3], "complete")

    def test_reduction_mode_preserves_file_normal_form_and_proof_bytes(self):
        source = b"x,y\n0\nx*x-x,\ny*y-y,\nx*x*x\n"
        self.enqueue("reduce", source)
        input_file, basis, proof = (self.root / name for name in ("input.ms", "basis.txt", "proof.txt"))
        input_file.write_bytes(source)
        file_result = self.invoke("-i", str(input_file), "-o", str(basis), "-p", str(proof),
                                  "-r", "1", "-v", "0")
        self.assertEqual(file_result.returncode, 0, file_result.stderr)
        result = self.solve("reduce", "-r", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.record("reduce")[1:3], (basis.read_bytes(), proof.read_bytes()))

    def test_malformed_input_does_not_publish_partial_outputs(self):
        for index, source in enumerate((b"invalid input", b"x\n0\nx*undeclared\n", b"x\n4\nx\n")):
            key = f"bad-{index}"
            self.enqueue(key, source)
            before = self.record(key)
            result = self.solve(key)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(self.record(key), before)

    def test_completed_job_cannot_be_overwritten(self):
        self.enqueue("once", b"x\n0\nx*x-x\n")
        self.assertEqual(self.solve("once").returncode, 0)
        before = self.record("once")
        result = self.solve("once")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"not an unused ready job", result.stderr)
        self.assertEqual(self.record("once"), before)

    def test_concurrent_independent_jobs_publish_without_losing_rows(self):
        sources = {"first": b"x\n0\nx*x-x\n", "second": b"x\n101\nx*x-2*x\n"}
        for key, source in sources.items():
            self.enqueue(key, source)
        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(self.solve, sources))
        self.assertTrue(all(result.returncode == 0 for result in results), results)
        for key, source in sources.items():
            self.assertEqual(self.record(key)[0], source)
            self.assertEqual(self.record(key)[3], "complete")

    def test_hot_journal_is_recovered_before_retrying_a_ready_job(self):
        source = b"x\n0\nx*x-x\n"
        self.enqueue("crashed", source)
        child_code = '''import sqlite3,sys,time
connection=sqlite3.connect(sys.argv[1])
connection.execute("PRAGMA journal_mode=DELETE")
connection.execute("PRAGMA cache_size=1")
connection.execute("PRAGMA cache_spill=ON")
connection.execute("BEGIN IMMEDIATE")
connection.execute("UPDATE f4ncgb_io SET basis=?,proof=?,state='complete' WHERE job_id='crashed'",
                   (b'b'*12000000,b'p'*12000000))
print("spilled",flush=True)
time.sleep(60)
'''
        child = subprocess.Popen([sys.executable, "-c", child_code, str(self.database)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "spilled")
        finally:
            child.kill()
            child.communicate(timeout=10)
        journal = Path(str(self.database) + "-journal")
        self.assertTrue(journal.is_file())
        self.assertGreater(journal.stat().st_size, 0)
        # Opening the database with Python here would recover it and conceal
        # whether the native retry itself can handle the interrupted writer.
        result = self.solve("crashed")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(journal.exists())
        original, basis, proof, state = self.record("crashed")
        self.assertEqual(original, source)
        self.assertEqual(state, "complete")
        self.assertTrue(basis)
        self.assertTrue(proof)

    def test_file_options_and_incomplete_database_options_are_rejected(self):
        self.enqueue("conflict", b"x\n0\nx*x-x\n")
        for flag in ("-i", "-o", "-p"):
            result = self.solve("conflict", flag, str(self.root / "unwanted"))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"cannot be combined", result.stderr)
        for args in (("--sqlite", str(self.database)), ("--sqlite-job", "conflict"),
                     ("--sqlite", "", "--sqlite-job", "")):
            result = self.invoke(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"Both --sqlite", result.stderr)
        self.assertEqual(self.record("conflict")[3], "ready")
        self.assertFalse((self.root / "unwanted").exists())

    def test_missing_database_missing_job_and_wrong_schema_are_rejected(self):
        missing = self.root / "absent.sqlite3"
        result = self.invoke("--sqlite", str(missing), "--sqlite-job", "absent")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(missing.exists())
        self.assertNotEqual(self.solve("absent").returncode, 0)
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("DROP TABLE f4ncgb_io")
            connection.execute("CREATE TABLE f4ncgb_io (job_id TEXT PRIMARY KEY,input TEXT)")
            connection.commit()
        self.assertNotEqual(self.solve("wrong-schema").returncode, 0)

    def test_text_input_and_triggered_tables_are_rejected(self):
        self.enqueue("typed", "x\n0\nx*x-x\n")
        self.assertNotEqual(self.solve("typed").returncode, 0)
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("CREATE TRIGGER alter_outputs AFTER UPDATE ON f4ncgb_io "
                               "BEGIN UPDATE f4ncgb_io SET proof=X'' WHERE job_id=NEW.job_id; END")
            connection.commit()
        self.enqueue("trigger", b"x\n0\nx*x-x\n")
        result = self.solve("trigger")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"must not have triggers", result.stderr)
        self.assertEqual(self.record("trigger")[3], "ready")

    def test_uppercase_and_quoted_trigger_tables_are_rejected(self):
        self.enqueue("protected", b"x\n0\nx*x-x\n")
        for table in ("F4NCGB_IO", '"F4ncgb_Io"'):
            with self.subTest(table=table):
                with closing(sqlite3.connect(self.database)) as connection:
                    connection.execute(f"CREATE TRIGGER case_trigger AFTER UPDATE ON {table} "
                                       "BEGIN UPDATE f4ncgb_io SET proof=X'' WHERE job_id=NEW.job_id; END")
                    connection.commit()
                result = self.solve("protected")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"must not have triggers", result.stderr)
                self.assertEqual(self.record("protected")[1:], (None, None, "ready"))
                with closing(sqlite3.connect(self.database)) as connection:
                    connection.execute("DROP TRIGGER case_trigger")
                    connection.commit()

    def test_replace_constraint_cannot_delete_a_previous_completed_job(self):
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("DROP TABLE f4ncgb_io")
            connection.execute(DDL[:-1] + ",UNIQUE(basis) ON CONFLICT REPLACE)")
            connection.execute("INSERT INTO f4ncgb_io VALUES (?,?,?,?,?)",
                               ("previous", b"x\n0\nx*x-x\n", b"x*x - x\n", b"prior proof", "complete"))
            connection.commit()
        self.enqueue("new", b"x\n0\nx*x-x\n")
        previous = self.record("previous")
        result = self.solve("new")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"unsupported table definition", result.stderr)
        self.assertEqual(self.record("previous"), previous)
        self.assertEqual(self.record("new")[1:], (None, None, "ready"))

    def test_same_named_unrelated_trigger_does_not_shadow_protocol_table(self):
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("DROP TABLE f4ncgb_io")
            connection.execute("CREATE TABLE other (value INTEGER)")
            connection.execute("CREATE TRIGGER f4ncgb_io AFTER INSERT ON other BEGIN SELECT 1; END")
            connection.execute(DDL)
            connection.commit()
        self.enqueue("ordinary", b"x\n0\nx*x-x\n")
        result = self.solve("ordinary")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.record("ordinary")[3], "complete")

    def test_nonprotocol_definitions_are_rejected_and_harmless_case_is_accepted(self):
        definitions = [DDL.replace(" CHECK(state IN('ready','complete'))", ""),
                       DDL.replace("'ready'", "'READY'"),
                       DDL.replace("job_id TEXT", "job_idTEXT")]
        if sqlite3.sqlite_version_info >= (3, 31):
            definitions.append(DDL[:-1] + ",derived INTEGER GENERATED ALWAYS AS (length(input)))")
        for definition in definitions:
            with self.subTest(definition=definition):
                with closing(sqlite3.connect(self.database)) as connection:
                    connection.execute("DROP TABLE f4ncgb_io")
                    connection.execute(definition)
                    connection.commit()
                result = self.solve("absent")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"unsupported table definition", result.stderr)
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute("DROP TABLE f4ncgb_io")
            connection.execute("""create table F4ncgb_Io (
                JOB_ID text primary key, INPUT blob not null, BASIS blob,
                PROOF blob, STATE text not null check (STATE in ('ready', 'complete')))""")
            connection.commit()
        self.enqueue("normalized", b"x\n0\nx*x-x\n")
        result = self.solve("normalized")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.record("normalized")[3], "complete")

    def test_readonly_database_does_not_publish_a_partial_result(self):
        self.enqueue("readonly", b"x\n0\nx*x-x\n")
        before = self.record("readonly")
        self.database.chmod(0o444)
        self.addCleanup(self.database.chmod, 0o644)
        if os.access(self.database, os.W_OK):
            self.skipTest("process can write read-only files")
        result = self.solve("readonly")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.record("readonly"), before)


class DisabledSQLite(unittest.TestCase):
    def test_capability_and_request_report_disabled_backend(self):
        features = subprocess.run([str(BINARY), "--features"], capture_output=True, check=True)
        self.assertEqual(json.loads(features.stdout), {"sqlite": False})
        with tempfile.TemporaryDirectory(prefix="f4ncgb-disabled-io-") as temporary:
            database = Path(temporary) / "unopened.sqlite3"
            result = subprocess.run([str(BINARY), "--sqlite", str(database), "--sqlite-job", "job"],
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"build with -DENABLE_SQLITE=ON", result.stderr)
            self.assertFalse(database.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--expect-disabled", action="store_true")
    arguments = parser.parse_args()
    BINARY = arguments.binary.resolve()
    case = DisabledSQLite if arguments.expect_disabled else NativeSQLite
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
    raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
