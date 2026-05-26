#include "stonks/datafeed/klinefeed.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

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

std::shared_ptr<arrow::ChunkedArray> column_as(
    const arrow::Table& table,
    const std::string& name,
    const std::shared_ptr<arrow::DataType>& target_type)
{
    const auto column = table.GetColumnByName(name);
    if (!column) {
        throw std::runtime_error{ "missing parquet column: " + name };
    }
    if (column->type()->Equals(*target_type)) {
        return column;
    }
    auto datum = unwrap_or_throw(
        arrow::compute::Cast(column, target_type),
        ("cast column " + name).c_str());
    return datum.chunked_array();
}

} // namespace

namespace stonks::datafeed {

KLineFeed::KLineFeed(std::filesystem::path parquet_path,
                     core::Timestamp::duration resolution)
: m_resolution{ resolution }
{
    auto infile = unwrap_or_throw(
        arrow::io::ReadableFile::Open(parquet_path.string()),
        "open parquet file");

    auto reader = unwrap_or_throw(
        parquet::arrow::OpenFile(infile, arrow::default_memory_pool()),
        "open parquet reader");

    auto table = unwrap_or_throw(reader->ReadTable(), "read parquet table");

    const auto ts_col = column_as(*table, "timestamp", arrow::timestamp(arrow::TimeUnit::MILLI, "UTC"));
    const auto symbol_col = column_as(*table, "symbol", arrow::large_utf8());
    const auto open_col = column_as(*table, "open", arrow::float64());
    const auto high_col = column_as(*table, "high", arrow::float64());
    const auto low_col = column_as(*table, "low", arrow::float64());
    const auto close_col = column_as(*table, "close", arrow::float64());
    const auto volume_col = column_as(*table, "volume", arrow::float64());

    m_klines.reserve(static_cast<std::size_t>(table->num_rows()));

    const int num_chunks = ts_col->num_chunks();
    for (int c = 0; c < num_chunks; ++c) {
        const auto& ts = static_cast<const arrow::TimestampArray&>(*ts_col->chunk(c));
        const auto& symbol = static_cast<const arrow::LargeStringArray&>(*symbol_col->chunk(c));
        const auto& open = static_cast<const arrow::DoubleArray&>(*open_col->chunk(c));
        const auto& high = static_cast<const arrow::DoubleArray&>(*high_col->chunk(c));
        const auto& low = static_cast<const arrow::DoubleArray&>(*low_col->chunk(c));
        const auto& close = static_cast<const arrow::DoubleArray&>(*close_col->chunk(c));
        const auto& volume = static_cast<const arrow::DoubleArray&>(*volume_col->chunk(c));

        const int64_t rows = ts.length();
        for (int64_t i = 0; i < rows; ++i) {
            m_klines.push_back(core::KLine{
                core::Timestamp::from_millis(ts.Value(i)),
                std::string{ symbol.GetView(i) },
                open.Value(i),
                high.Value(i),
                low.Value(i),
                close.Value(i),
                volume.Value(i),
            });
        }
    }
}

std::optional<core::Timestamp> KLineFeed::next_timestamp() const
{
    if (m_cursor >= m_klines.size()) { return std::nullopt; }
    return m_klines[m_cursor].timestamp;
}

void KLineFeed::advance()
{
    if (m_cursor < m_klines.size()) { ++m_cursor; }
}

std::vector<core::KLine> KLineFeed::klines(
    core::Timestamp start,
    core::Timestamp end) const
{
    if (end < start) { return {}; }
    auto lo = std::lower_bound(m_klines.begin(), m_klines.end(), start,
        [](const core::KLine& k, core::Timestamp t) { return k.timestamp < t; });
    auto hi = std::upper_bound(m_klines.begin(), m_klines.end(), end,
        [](core::Timestamp t, const core::KLine& k) { return t < k.timestamp; });
    return std::vector<core::KLine>(lo, hi);
}

} // namespace stonks::datafeed
