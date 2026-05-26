#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "stonks/core/types.h"

class KLineFeed
{
public:
    explicit KLineFeed(std::filesystem::path parquet_path);

    std::optional<stonks::core::Timestamp> peek(stonks::core::Timestamp current) const;

    std::vector<stonks::core::KLine> klines(int count) const;

private:
    std::vector<stonks::core::KLine> m_klines;
};
