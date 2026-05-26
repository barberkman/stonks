#pragma once

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <vector>

#include "stonks/core/types.h"

class KLineFeed
{
public:
    explicit KLineFeed(std::filesystem::path parquet_path,
                       stonks::core::Timestamp::duration resolution = std::chrono::days{ 1 });

    std::optional<stonks::core::Timestamp> next_timestamp() const;
    void advance();

    std::vector<stonks::core::KLine> klines(
        stonks::core::Timestamp start,
        stonks::core::Timestamp end) const;

    stonks::core::Timestamp::duration resolution() const { return m_resolution; }

private:
    std::vector<stonks::core::KLine> m_klines;
    std::size_t m_cursor{ 0 };
    stonks::core::Timestamp::duration m_resolution;
};
