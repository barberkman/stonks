#include "stonks/datafeed/klinefeed.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/compute/api.h>
#include <parquet/arrow/reader.h>

#include "stonks/core/log.h"

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

    std::vector<Row> rows;
    rows.reserve(static_cast<std::size_t>(table->num_rows()));

    const int num_chunks = ts_col->num_chunks();
    for (int c = 0; c < num_chunks; ++c) {
        const auto& ts = static_cast<const arrow::TimestampArray&>(*ts_col->chunk(c));
        const auto& symbol = static_cast<const arrow::LargeStringArray&>(*symbol_col->chunk(c));
        const auto& open = static_cast<const arrow::DoubleArray&>(*open_col->chunk(c));
        const auto& high = static_cast<const arrow::DoubleArray&>(*high_col->chunk(c));
        const auto& low = static_cast<const arrow::DoubleArray&>(*low_col->chunk(c));
        const auto& close = static_cast<const arrow::DoubleArray&>(*close_col->chunk(c));
        const auto& volume = static_cast<const arrow::DoubleArray&>(*volume_col->chunk(c));

        const int64_t chunk_rows = ts.length();
        for (int64_t i = 0; i < chunk_rows; ++i) {
            rows.push_back(Row{
                ts.Value(i),
                std::string{ symbol.GetView(i) },
                open.Value(i),
                high.Value(i),
                low.Value(i),
                close.Value(i),
                volume.Value(i),
            });
        }
    }

    build(std::move(rows));
}

KLineFeed::KLineFeed(std::vector<Row> rows, core::Timestamp::duration resolution)
: m_resolution{ resolution }
{
    build(std::move(rows));
}

void KLineFeed::build(std::vector<Row> rows)
{
    const auto n = static_cast<std::uint32_t>(rows.size());

    // 1. Intern symbols by first appearance (file order) -> SymbolID.
    std::unordered_map<std::string_view, core::SymbolID> intern;
    std::vector<core::SymbolID> file_symbol(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        const auto next_id = static_cast<core::SymbolID>(m_id_to_ticker.size());
        const auto [it, inserted] = intern.try_emplace(rows[i].symbol, next_id);
        if (inserted) { m_id_to_ticker.push_back(rows[i].symbol); }
        file_symbol[i] = it->second;
    }
    const auto num_symbols = static_cast<core::SymbolID>(m_id_to_ticker.size());

    // 2. Physical layout: group rows by symbol (contiguous), chronological
    //    within a symbol, file order as the stable tiebreak. perm[r] is the
    //    file index of physical row r.
    std::vector<std::uint32_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0u);
    std::stable_sort(perm.begin(), perm.end(),
        [&](std::uint32_t a, std::uint32_t b) {
            if (file_symbol[a] != file_symbol[b]) { return file_symbol[a] < file_symbol[b]; }
            return rows[a].timestamp_ms < rows[b].timestamp_ms;
        });

    // 3. Fill the columns from the permutation; derive per-symbol slices and
    //    local indices as we go (rows of one symbol are now consecutive).
    m_ts.resize(n);
    m_open.resize(n);
    m_high.resize(n);
    m_low.resize(n);
    m_close.resize(n);
    m_volume.resize(n);
    m_row_symbol.resize(n);
    m_row_local.resize(n);
    std::vector<std::uint32_t> running_local(num_symbols, 0u);

    for (std::uint32_t r = 0; r < n; ++r) {
        const std::uint32_t i = perm[r];
        const core::SymbolID s = file_symbol[i];
        m_ts[r] = rows[i].timestamp_ms;
        m_open[r] = rows[i].open;
        m_high[r] = rows[i].high;
        m_low[r] = rows[i].low;
        m_close[r] = rows[i].close;
        m_volume[r] = rows[i].volume;
        m_row_symbol[r] = s;
        m_row_local[r] = running_local[s]++;
    }

    // 4. Global iteration order: physical rows by (timestamp, original file
    //    position), so the engine visits bars in chronological order with file
    //    order breaking ties — deterministic and identical to the legacy
    //    flat-vector walk for time-sorted input.
    m_order.resize(n);
    std::iota(m_order.begin(), m_order.end(), 0u);
    std::stable_sort(m_order.begin(), m_order.end(),
        [&](std::uint32_t a, std::uint32_t b) {
            if (m_ts[a] != m_ts[b]) { return m_ts[a] < m_ts[b]; }
            return perm[a] < perm[b];
        });

    // 5. Timestamp-group boundaries: m_group_start[g] is the m_order index where
    //    group g begins; the trailing sentinel is n. Consecutive m_order entries
    //    with equal timestamp form one group (one strategy tick).
    m_group_start.clear();
    for (std::uint32_t k = 0; k < n; ++k) {
        if (k == 0 || m_ts[m_order[k]] != m_ts[m_order[k - 1]]) {
            m_group_start.push_back(k);
        }
    }
    m_group_start.push_back(n);
}

std::optional<core::Timestamp> KLineFeed::next_timestamp() const
{
    if (m_group + 1 >= m_group_start.size()) { return std::nullopt; }
    return core::Timestamp::from_millis(m_ts[m_order[m_group_start[m_group]]]);
}

void KLineFeed::advance()
{
    if (m_group + 1 < m_group_start.size()) {
        ++m_group;
    }
}

std::vector<core::KLine> KLineFeed::current_bars() const
{
    std::vector<core::KLine> bars;
    const std::uint32_t begin = m_group_start[m_group];
    const std::uint32_t end = m_group_start[m_group + 1];
    bars.reserve(end - begin);
    for (std::uint32_t k = begin; k < end; ++k) {
        const std::uint32_t r = m_order[k];
        bars.push_back(core::KLine{
            core::Timestamp::from_millis(m_ts[r]),
            m_id_to_ticker[m_row_symbol[r]],
            m_open[r],
            m_high[r],
            m_low[r],
            m_close[r],
            m_volume[r],
        });
    }
    return bars;
}

core::SeriesView KLineFeed::series_for(std::uint32_t r, int count) const
{
    const std::uint32_t available = m_row_local[r] + 1;   // bars up to & incl. row r
    const std::uint32_t cnt = (count <= 0)
        ? 0u
        : std::min(static_cast<std::uint32_t>(count), available);
    const std::uint32_t first = (cnt == 0) ? r : r - (cnt - 1);

    return core::SeriesView{
        std::span<const std::int64_t>{ m_ts.data() + first, cnt },
        std::span<const double>{ m_open.data() + first, cnt },
        std::span<const double>{ m_high.data() + first, cnt },
        std::span<const double>{ m_low.data() + first, cnt },
        std::span<const double>{ m_close.data() + first, cnt },
        std::span<const double>{ m_volume.data() + first, cnt },
    };
}

core::MarketWindow KLineFeed::window(int count) const
{
    core::MarketWindow w;
    const std::uint32_t begin = m_group_start[m_group];
    const std::uint32_t end = m_group_start[m_group + 1];
    w.series.reserve(end - begin);
    for (std::uint32_t k = begin; k < end; ++k) {
        const std::uint32_t r = m_order[k];
        w.series.push_back(core::SymbolSeries{
            std::string_view{ m_id_to_ticker[m_row_symbol[r]] },
            series_for(r, count),
        });
    }
    return w;
}

} // namespace stonks::datafeed
