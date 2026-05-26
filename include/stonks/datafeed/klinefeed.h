#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::datafeed {

class KLineFeed
{
public:
    explicit KLineFeed(std::filesystem::path parquet_path,
                       core::Timestamp::duration resolution = std::chrono::days{ 1 });

    std::optional<core::Timestamp> next_timestamp() const;
    void advance();

    std::vector<core::KLine> klines(
        core::Timestamp start,
        core::Timestamp end) const;

    core::Timestamp::duration resolution() const { return m_resolution; }

private:
    std::vector<core::KLine> m_klines;
    std::size_t m_cursor{ 0 };
    core::Timestamp::duration m_resolution;
};

} // namespace stonks::datafeed
