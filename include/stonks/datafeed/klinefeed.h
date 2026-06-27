#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::datafeed {

// Optional row filter applied at KLineFeed construction. Bounds are UTC date
// strings ("YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS"), half-open [start, end). An
// empty member is no constraint: "" bounds are unbounded, an empty symbol list
// admits all symbols. The default member initializers make partial designated
// initialization (e.g. { .start = "2024-01-01" }) warning-free.
struct Filter
{
    std::string start{};                  // inclusive lower bound, "" = unbounded
    std::string end{};                    // exclusive upper bound, "" = unbounded
    std::vector<core::Symbol> symbols{};  // allowlist; empty = all symbols
};

// Columnar OHLCV feed. Bars are stored struct-of-arrays with each symbol's rows
// laid out contiguously and chronologically, so per-symbol lookback is a O(1)
// contiguous slice and the columns can be handed to Python as zero-copy numpy
// arrays. Iteration walks all symbols in global time order.
class KLineFeed
{
public:
    // One input bar in file/insertion order. Used by the in-memory constructor
    // and internally by the Parquet loader.
    struct Row
    {
        std::int64_t timestamp_ms;
        core::Symbol symbol;
        core::Price open, high, low, close;
        core::Volume volume;
    };

    explicit KLineFeed(std::filesystem::path parquet_path,
                       Filter filter = {},
                       core::Timestamp::duration resolution = std::chrono::days{ 1 });

    // In-memory construction (testing): rows in file order, run through the same
    // interning, per-symbol grouping, and global time ordering as the Parquet
    // path.
    explicit KLineFeed(std::vector<Row> rows,
                       Filter filter = {},
                       core::Timestamp::duration resolution = std::chrono::days{ 1 });

    // Iteration is per timestamp: next_timestamp() reports the current timestamp,
    // advance() moves to the next distinct one.
    std::optional<core::Timestamp> next_timestamp() const;
    void advance();

    // The bars at the current timestamp (one per symbol that printed), fed to the
    // broker before the strategy runs.
    std::vector<core::KLine> current_bars() const;

    // Each symbol printing at the current timestamp, with its last `count` bars
    // (including today's) as zero-copy column views. No-lookahead by
    // construction: a symbol's slice can only reach its own bars up to now.
    // count <= 0 yields empty slices; count beyond a symbol's seen bars clamps.
    core::MarketWindow window(int count) const;

    core::Timestamp::duration resolution() const { return m_resolution; }

    std::size_t size() const { return m_order.size(); }

private:
    void build(std::vector<Row> rows, const Filter& filter);

    // Zero-copy view of physical row r's symbol, last `count` bars ending at r.
    core::SeriesView series_for(std::uint32_t r, int count) const;

    // Columnar SoA, indexed by physical row. Each symbol's rows occupy one
    // contiguous, chronologically-ordered block.
    std::vector<std::int64_t> m_ts;               // ms since epoch
    std::vector<core::Price> m_open, m_high, m_low, m_close;
    std::vector<core::Volume> m_volume;

    std::vector<core::Symbol> m_id_to_ticker;     // SymbolID -> ticker

    std::vector<core::SymbolID> m_row_symbol;     // physical row -> owning symbol
    std::vector<std::uint32_t> m_row_local;       // physical row -> index within its block

    std::vector<std::uint32_t> m_order;           // physical rows in global time order
    std::vector<std::uint32_t> m_group_start;     // m_order index where each timestamp begins (+ end sentinel)
    std::size_t m_group{ 0 };                      // current timestamp-group index

    core::Timestamp::duration m_resolution;
};

} // namespace stonks::datafeed
