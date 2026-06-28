#include "stonks/datafeed/parquetmeta.h"

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/compute/api.h>
#include <parquet/arrow/reader.h>

namespace {

template <class T>
T unwrap_or_throw(arrow::Result<T> result, const char* what)
{
    if (!result.ok()) {
        throw std::runtime_error{ std::string{ what } + ": " + result.status().ToString() };
    }
    return std::move(result).ValueOrDie();
}

void check_status(const arrow::Status& status, const char* what)
{
    if (!status.ok()) {
        throw std::runtime_error{ std::string{ what } + ": " + status.ToString() };
    }
}

std::shared_ptr<arrow::ChunkedArray> cast_to(
    const std::shared_ptr<arrow::ChunkedArray>& column,
    const std::shared_ptr<arrow::DataType>& target_type,
    const char* what)
{
    if (column->type()->Equals(*target_type)) {
        return column;
    }
    auto datum = unwrap_or_throw(arrow::compute::Cast(column, target_type), what);
    return datum.chunked_array();
}

} // namespace

namespace stonks::datafeed {

ParquetMeta peek_parquet(const std::filesystem::path& parquet_path)
{
    auto infile = unwrap_or_throw(
        arrow::io::ReadableFile::Open(parquet_path.string()),
        "open parquet file");

    auto reader = unwrap_or_throw(
        parquet::arrow::OpenFile(infile, arrow::default_memory_pool()),
        "open parquet reader");

    std::shared_ptr<arrow::Schema> schema;
    check_status(reader->GetSchema(&schema), "read parquet schema");

    const int sym_idx = schema->GetFieldIndex("symbol");
    const int ts_idx = schema->GetFieldIndex("timestamp");
    if (sym_idx < 0) { throw std::runtime_error{ "missing parquet column: symbol" }; }
    if (ts_idx < 0) { throw std::runtime_error{ "missing parquet column: timestamp" }; }

    // Project only the two columns we need, so big OHLCV files stay cheap.
    std::shared_ptr<arrow::Table> table;
    check_status(reader->ReadTable({ ts_idx, sym_idx }, &table), "read parquet columns");

    ParquetMeta meta;
    meta.rows = table->num_rows();

    auto sym_col = table->GetColumnByName("symbol");
    auto ts_col = table->GetColumnByName("timestamp");
    if (!sym_col || !ts_col) { throw std::runtime_error{ "missing projected parquet column" }; }

    sym_col = cast_to(sym_col, arrow::large_utf8(), "cast symbol column");
    // Normalize to epoch-ms then int64 so MinMax yields integral bounds.
    const auto ts_ms = cast_to(ts_col, arrow::timestamp(arrow::TimeUnit::MILLI, "UTC"),
                               "cast timestamp column");
    const auto ts_int = cast_to(ts_ms, arrow::int64(), "cast timestamp to int64");

    // Distinct symbols. Done by hand rather than via the 'unique'/'min_max'
    // compute kernels, which are not registered in every Arrow build (only the
    // cast kernels are linked here). A std::set yields sorted, de-duplicated keys.
    std::set<std::string> distinct;
    for (const auto& chunk : sym_col->chunks()) {
        const auto& arr = static_cast<const arrow::LargeStringArray&>(*chunk);
        for (std::int64_t i = 0; i < arr.length(); ++i) {
            if (!arr.IsNull(i)) { distinct.emplace(arr.GetView(i)); }
        }
    }
    meta.symbols.assign(distinct.begin(), distinct.end());

    // Timestamp bounds.
    bool seen = false;
    std::int64_t mn = 0, mx = 0;
    for (const auto& chunk : ts_int->chunks()) {
        const auto& arr = static_cast<const arrow::Int64Array&>(*chunk);
        for (std::int64_t i = 0; i < arr.length(); ++i) {
            if (arr.IsNull(i)) { continue; }
            const std::int64_t v = arr.Value(i);
            if (!seen) { mn = mx = v; seen = true; }
            else { mn = std::min(mn, v); mx = std::max(mx, v); }
        }
    }
    if (seen) {
        meta.min_ts_ms = mn;
        meta.max_ts_ms = mx;
    }

    return meta;
}

} // namespace stonks::datafeed
