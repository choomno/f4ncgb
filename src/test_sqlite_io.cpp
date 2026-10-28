#include <boost/test/unit_test.hpp>

#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>

#include <sqlite3.h>

#include "sqlite_io.hpp"

namespace {

class temporary_jobs {
  std::filesystem::path directory_;
public:
  sqlite3* database = nullptr;
  std::string path;

  temporary_jobs() {
    directory_ = std::filesystem::temp_directory_path()
                 / ("f4ncgb-native-jobs-" + std::to_string(std::random_device{}())
                    + "-" + std::to_string(std::random_device{}()));
    std::filesystem::create_directory(directory_);
    path = (directory_ / "jobs.sqlite3").string();
    BOOST_REQUIRE_EQUAL(sqlite3_open(path.c_str(), &database), SQLITE_OK);
    execute("CREATE TABLE f4ncgb_io (job_id TEXT PRIMARY KEY,input BLOB NOT NULL,"
            "basis BLOB,proof BLOB,state TEXT NOT NULL CHECK(state IN('ready','complete')))");
    execute("INSERT INTO f4ncgb_io VALUES ('job',X'780a300a780a',NULL,NULL,'ready')");
  }

  ~temporary_jobs() {
    sqlite3_close(database);
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
  }

  void execute(const char* sql) {
    BOOST_REQUIRE_EQUAL(sqlite3_exec(database, sql, nullptr, nullptr, nullptr), SQLITE_OK);
  }

  std::string value(const char* sql) {
    sqlite3_stmt* statement = nullptr;
    BOOST_REQUIRE_EQUAL(sqlite3_prepare_v2(database, sql, -1, &statement, nullptr), SQLITE_OK);
    BOOST_REQUIRE_EQUAL(sqlite3_step(statement), SQLITE_ROW);
    const auto* text = sqlite3_column_text(statement, 0);
    const std::string result = text == nullptr ? "NULL" : reinterpret_cast<const char*>(text);
    sqlite3_finalize(statement);
    return result;
  }
};

}

BOOST_AUTO_TEST_CASE(sqlite_completion_rejects_input_changed_during_computation) {
  temporary_jobs jobs;
  f4ncgb::sqlite_job computation(jobs.path, "job");
  BOOST_TEST(computation.input() == "x\n0\nx\n");
  jobs.execute("UPDATE f4ncgb_io SET input=X'6368616e676564' WHERE job_id='job'");
  BOOST_CHECK_THROW(computation.complete("basis", "proof"), std::runtime_error);
  BOOST_TEST(jobs.value("SELECT state FROM f4ncgb_io WHERE job_id='job'") == "ready");
  BOOST_TEST(jobs.value("SELECT basis FROM f4ncgb_io WHERE job_id='job'") == "NULL");
  BOOST_TEST(jobs.value("SELECT proof FROM f4ncgb_io WHERE job_id='job'") == "NULL");
}

BOOST_AUTO_TEST_CASE(sqlite_first_completed_writer_preserves_its_outputs) {
  temporary_jobs jobs;
  f4ncgb::sqlite_job first(jobs.path, "job"), second(jobs.path, "job");
  first.complete("first basis", "first proof");
  BOOST_CHECK_THROW(second.complete("second basis", "second proof"), std::runtime_error);
  BOOST_TEST(jobs.value("SELECT basis FROM f4ncgb_io") == "first basis");
  BOOST_TEST(jobs.value("SELECT proof FROM f4ncgb_io") == "first proof");
  BOOST_TEST(jobs.value("SELECT state FROM f4ncgb_io") == "complete");
}

BOOST_AUTO_TEST_CASE(sqlite_completion_failure_rolls_back_all_output_columns) {
  temporary_jobs jobs;
  f4ncgb::sqlite_job computation(jobs.path, "job");
  jobs.execute("INSERT INTO f4ncgb_io VALUES ('other',X'78',X'6261736973',X'70726f6f66','complete')");
  jobs.execute("CREATE UNIQUE INDEX unique_basis ON f4ncgb_io(basis)");
  BOOST_CHECK_THROW(computation.complete("basis", "new proof"), std::runtime_error);
  BOOST_TEST(jobs.value("SELECT state FROM f4ncgb_io WHERE job_id='job'") == "ready");
  BOOST_TEST(jobs.value("SELECT basis FROM f4ncgb_io WHERE job_id='job'") == "NULL");
  BOOST_TEST(jobs.value("SELECT proof FROM f4ncgb_io WHERE job_id='job'") == "NULL");
  BOOST_TEST(jobs.value("SELECT proof FROM f4ncgb_io WHERE job_id='other'") == "proof");
}

BOOST_AUTO_TEST_CASE(sqlite_schema_change_cannot_mutate_completion_outputs) {
  temporary_jobs jobs;
  f4ncgb::sqlite_job computation(jobs.path, "job");
  jobs.execute("CREATE TRIGGER output_mutation AFTER UPDATE ON f4ncgb_io "
               "BEGIN UPDATE f4ncgb_io SET proof=X'' WHERE job_id=NEW.job_id; END");
  BOOST_CHECK_THROW(computation.complete("basis", "proof"), std::runtime_error);
  BOOST_TEST(jobs.value("SELECT state FROM f4ncgb_io WHERE job_id='job'") == "ready");
}

BOOST_AUTO_TEST_CASE(sqlite_empty_outputs_remain_blob_values) {
  temporary_jobs jobs;
  f4ncgb::sqlite_job computation(jobs.path, "job");
  computation.complete("", "");
  BOOST_TEST(jobs.value("SELECT typeof(basis) FROM f4ncgb_io") == "blob");
  BOOST_TEST(jobs.value("SELECT typeof(proof) FROM f4ncgb_io") == "blob");
  BOOST_TEST(jobs.value("SELECT length(basis) FROM f4ncgb_io") == "0");
}

BOOST_AUTO_TEST_CASE(sqlite_trigger_table_case_cannot_bypass_publication_guard) {
  for(const std::string table : {"F4NCGB_IO", "\"F4ncgb_Io\""}) {
    temporary_jobs jobs;
    f4ncgb::sqlite_job computation(jobs.path, "job");
    const std::string trigger = "CREATE TRIGGER output_mutation AFTER UPDATE ON " + table
      + " BEGIN UPDATE f4ncgb_io SET proof=X'' WHERE job_id=NEW.job_id; END";
    jobs.execute(trigger.c_str());
    BOOST_CHECK_THROW(f4ncgb::sqlite_job(jobs.path, "job"), std::runtime_error);
    BOOST_CHECK_THROW(computation.complete("basis", "proof"), std::runtime_error);
    BOOST_TEST(jobs.value("SELECT state FROM f4ncgb_io WHERE job_id='job'") == "ready");
    BOOST_TEST(jobs.value("SELECT proof FROM f4ncgb_io WHERE job_id='job'") == "NULL");
  }
}

BOOST_AUTO_TEST_CASE(sqlite_replace_constraint_cannot_delete_another_completed_job) {
  temporary_jobs jobs;
  f4ncgb::sqlite_job computation(jobs.path, "job");
  jobs.execute("DROP TABLE f4ncgb_io");
  jobs.execute("CREATE TABLE f4ncgb_io (job_id TEXT PRIMARY KEY,input BLOB NOT NULL,"
               "basis BLOB,proof BLOB,state TEXT NOT NULL CHECK(state IN('ready','complete')),"
               "UNIQUE(basis) ON CONFLICT REPLACE)");
  jobs.execute("INSERT INTO f4ncgb_io VALUES ('job',X'780a300a780a',NULL,NULL,'ready')");
  jobs.execute("INSERT INTO f4ncgb_io VALUES ('other',X'78',X'6261736973',X'7072696f722070726f6f66','complete')");
  BOOST_CHECK_THROW(f4ncgb::sqlite_job(jobs.path, "job"), std::runtime_error);
  BOOST_CHECK_THROW(computation.complete("basis", "new proof"), std::runtime_error);
  BOOST_TEST(jobs.value("SELECT count(*) FROM f4ncgb_io") == "2");
  BOOST_TEST(jobs.value("SELECT proof FROM f4ncgb_io WHERE job_id='other'") == "prior proof");
  BOOST_TEST(jobs.value("SELECT state FROM f4ncgb_io WHERE job_id='job'") == "ready");
  BOOST_TEST(jobs.value("SELECT basis FROM f4ncgb_io WHERE job_id='job'") == "NULL");
}

BOOST_AUTO_TEST_CASE(sqlite_unrelated_same_named_trigger_does_not_shadow_the_table) {
  temporary_jobs jobs;
  jobs.execute("DROP TABLE f4ncgb_io");
  jobs.execute("CREATE TABLE other (value INTEGER)");
  jobs.execute("CREATE TRIGGER f4ncgb_io AFTER INSERT ON other BEGIN SELECT 1; END");
  jobs.execute("CREATE TABLE f4ncgb_io (job_id TEXT PRIMARY KEY,input BLOB NOT NULL,"
               "basis BLOB,proof BLOB,state TEXT NOT NULL CHECK(state IN('ready','complete')))");
  jobs.execute("INSERT INTO f4ncgb_io VALUES ('job',X'780a300a780a',NULL,NULL,'ready')");
  f4ncgb::sqlite_job computation(jobs.path, "job");
  computation.complete("basis", "proof");
  BOOST_TEST(jobs.value("SELECT proof FROM f4ncgb_io WHERE job_id='job'") == "proof");
}

BOOST_AUTO_TEST_CASE(sqlite_exact_protocol_definition_preserves_string_literal_case) {
  temporary_jobs jobs;
  jobs.execute("DROP TABLE f4ncgb_io");
  jobs.execute("CREATE TABLE f4ncgb_io (job_id TEXT PRIMARY KEY,input BLOB NOT NULL,"
               "basis BLOB,proof BLOB,state TEXT NOT NULL CHECK(state IN('READY','complete')))");
  BOOST_CHECK_THROW(f4ncgb::sqlite_job(jobs.path, "job"), std::runtime_error);
}
