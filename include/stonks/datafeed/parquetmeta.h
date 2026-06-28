#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::datafeed {

// Lightweight metadata read from a parquet OHLCV file without building a full
// KLineFeed: the distinct symbols and the timestamp span. Powers GUI pickers
// (the symbol allowlist and a default date range) and dataset listings.
struct ParquetMeta
{
    std::vector<core::Symbol> symbols;          // distinct, sorted ascending
    std::optional<std::int64_t> min_ts_ms;      // earliest timestamp (epoch ms)
    std::optional<std::int64_t> max_ts_ms;      // latest timestamp (epoch ms)
    std::int64_t rows{ 0 };
};

// Reads only the `symbol` and `timestamp` columns (column projection, so big
// OHLCV files stay cheap to peek). Throws std::runtime_error on open/parse
// failure or a missing column — the same failure mode as KLineFeed.
ParquetMeta peek_parquet(const std::filesystem::path& parquet_path);

} // namespace stonks::datafeed
