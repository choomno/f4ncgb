#include "sqlite_io.hpp"

#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <sqlite3.h>

namespace f4ncgb {
namespace {

struct database_deleter {
  void operator()(sqlite3* database) const { sqlite3_close(database); }
};
using database_handle = std::unique_ptr<sqlite3, database_deleter>;

struct statement_deleter {
  void operator()(sqlite3_stmt* statement) const { sqlite3_finalize(statement); }
};
using statement_handle = std::unique_ptr<sqlite3_stmt, statement_deleter>;

[[noreturn]] void fail(sqlite3* database, std::string_view operation) {
  throw std::runtime_error(std::string("SQLite ") + std::string(operation) + ": "
                           + sqlite3_errmsg(database));
}

void execute(sqlite3* database, const char* sql) {
  if(sqlite3_exec(database, sql, nullptr, nullptr, nullptr) != SQLITE_OK)
    fail(database, "operation failed");
}

database_handle open_database(const std::string& path) {
  sqlite3* raw = nullptr;
  const int status = sqlite3_open_v2(
    path.c_str(), &raw, SQLITE_OPEN_READWRITE, nullptr);
  database_handle database(raw);
  if(status != SQLITE_OK)
    fail(database.get(), "open failed");
  sqlite3_busy_timeout(database.get(), 30000);
  // The protocol owns no foreign-key relationships. Application cascades could
  // invoke triggers on other tables and mutate these outputs indirectly.
  execute(database.get(), "PRAGMA foreign_keys=OFF");
  execute(database.get(), "PRAGMA trusted_schema=OFF");
  return database;
}

statement_handle prepare(sqlite3* database, const char* sql) {
  sqlite3_stmt* raw = nullptr;
  if(sqlite3_prepare_v2(database, sql, -1, &raw, nullptr) != SQLITE_OK)
    fail(database, "statement preparation failed");
  return statement_handle(raw);
}

void bind_text(sqlite3* database, sqlite3_stmt* statement, int index,
               const std::string& text) {
  if(sqlite3_bind_text64(statement, index, text.data(), text.size(),
                         SQLITE_TRANSIENT, SQLITE_UTF8) != SQLITE_OK)
    fail(database, "text binding failed");
}

void bind_blob(sqlite3* database, sqlite3_stmt* statement, int index,
               const std::string& bytes) {
  if(sqlite3_bind_blob64(statement, index, bytes.data(), bytes.size(),
                         SQLITE_TRANSIENT) != SQLITE_OK)
    fail(database, "blob binding failed");
}

std::string column_string(sqlite3_stmt* statement, int column, int required_type) {
  if(sqlite3_column_type(statement, column) != required_type)
    throw std::runtime_error("SQLite job has an invalid column type");
  const int length = sqlite3_column_bytes(statement, column);
  if(length == 0)
    return {};
  const void* bytes = sqlite3_column_blob(statement, column);
  if(bytes == nullptr)
    throw std::runtime_error("SQLite could not read job bytes");
  return std::string(static_cast<const char*>(bytes), static_cast<size_t>(length));
}

std::vector<std::string> normalized_ddl(std::string_view sql) {
  std::vector<std::string> result;
  std::string token;
  bool literal = false;
  for(char c : sql) {
    if(literal) {
      token += c;
      if(c == '\'') {
        literal = false;
        result.push_back(std::move(token));
        token.clear();
      }
    } else if((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
              || (c >= '0' && c <= '9') || c == '_' || c == '$') {
      if(c >= 'a' && c <= 'z')
        c = static_cast<char>(c - 'a' + 'A');
      token += c;
    } else {
      if(!token.empty()) {
        result.push_back(std::move(token));
        token.clear();
      }
      if(c == '\'') {
        literal = true;
        token += c;
      } else if(c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != '\f' && c != '\v') {
        result.emplace_back(1, c);
      }
    }
  }
  if(!token.empty())
    result.push_back(std::move(token));
  return result;
}

void check_schema(sqlite3* database) {
  // Column listings omit generated columns and do not describe conflict
  // policies. Accept only the protocol's full table definition, with harmless
  // ASCII case/whitespace differences outside SQL string literals.
  constexpr auto definition =
    "CREATE TABLE f4ncgb_io (job_id TEXT PRIMARY KEY,input BLOB NOT NULL,"
    "basis BLOB,proof BLOB,state TEXT NOT NULL CHECK(state IN('ready','complete')))";
  auto table = prepare(database,
    "SELECT sql FROM sqlite_master "
    "WHERE name='f4ncgb_io' COLLATE NOCASE AND type='table'");
  if(sqlite3_step(table.get()) != SQLITE_ROW
     || normalized_ddl(column_string(table.get(), 0, SQLITE_TEXT)) != normalized_ddl(definition))
    throw std::runtime_error("SQLite f4ncgb_io has an unsupported table definition");
  auto triggers = prepare(database,
    "SELECT 1 FROM sqlite_master "
    "WHERE type='trigger' AND tbl_name='f4ncgb_io' COLLATE NOCASE");
  if(sqlite3_step(triggers.get()) != SQLITE_DONE)
    throw std::runtime_error("SQLite f4ncgb_io table must not have triggers");
}

class transaction {
  sqlite3* database_;
  bool active_ = true;
public:
  explicit transaction(sqlite3* database, bool write) : database_(database) {
    execute(database_, write ? "BEGIN IMMEDIATE" : "BEGIN");
  }
  ~transaction() {
    if(active_)
      sqlite3_exec(database_, "ROLLBACK", nullptr, nullptr, nullptr);
  }
  void commit() {
    execute(database_, "COMMIT");
    active_ = false;
  }
};

std::string read_ready_input(sqlite3* database, const std::string& id) {
  auto statement = prepare(database,
    "SELECT input,basis,proof,state FROM f4ncgb_io WHERE job_id=?");
  bind_text(database, statement.get(), 1, id);
  const int status = sqlite3_step(statement.get());
  if(status != SQLITE_ROW) {
    if(status != SQLITE_DONE)
      fail(database, "job read failed");
    throw std::runtime_error("SQLite job does not exist");
  }
  std::string input = column_string(statement.get(), 0, SQLITE_BLOB);
  if(column_string(statement.get(), 3, SQLITE_TEXT) != "ready"
     || sqlite3_column_type(statement.get(), 1) != SQLITE_NULL
     || sqlite3_column_type(statement.get(), 2) != SQLITE_NULL)
    throw std::runtime_error("SQLite job is not an unused ready job");
  if(sqlite3_step(statement.get()) != SQLITE_DONE)
    fail(database, "job read did not finish");
  return input;
}

}

sqlite_job::sqlite_job(std::string database, std::string job_id)
  : database_(std::move(database)), job_id_(std::move(job_id)) {
  if(database_.empty() || job_id_.empty())
    throw std::runtime_error("SQLite database and job id must be nonempty");
  // Even the initial snapshot needs a writable connection: SQLite may have to
  // recover a hot rollback journal left by an interrupted earlier invocation.
  // No transaction remains open while the algebra is computed.
  auto database_handle = open_database(database_);
  execute(database_handle.get(), "PRAGMA query_only=ON");
  transaction snapshot(database_handle.get(), false);
  check_schema(database_handle.get());
  input_ = read_ready_input(database_handle.get(), job_id_);
  snapshot.commit();
}

void sqlite_job::complete(const std::string& basis, const std::string& proof) const {
  auto database_handle = open_database(database_);
  transaction update(database_handle.get(), true);
  check_schema(database_handle.get());
  auto statement = prepare(database_handle.get(),
    "UPDATE f4ncgb_io SET basis=?,proof=?,state='complete' "
    "WHERE job_id=? AND input=? AND state='ready' AND basis IS NULL AND proof IS NULL");
  bind_blob(database_handle.get(), statement.get(), 1, basis);
  bind_blob(database_handle.get(), statement.get(), 2, proof);
  bind_text(database_handle.get(), statement.get(), 3, job_id_);
  bind_blob(database_handle.get(), statement.get(), 4, input_);
  if(sqlite3_step(statement.get()) != SQLITE_DONE)
    fail(database_handle.get(), "job completion failed");
  if(sqlite3_changes(database_handle.get()) != 1)
    throw std::runtime_error("SQLite job changed while the computation ran");
  update.commit();
}

}
