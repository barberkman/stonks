#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::core::test {

struct StubBroker
{
    std::vector<Order>* placed{ nullptr };

    Balance cash() const { return {}; }
    Balance equity() const { return {}; }
    bool place_order(const Order& o)
    {
        if (placed) { placed->push_back(o); }
        return true;
    }
};

struct StubFeed
{
    std::vector<KLine> bars;
    std::size_t cursor{ 0 };
    Timestamp::duration res{ std::chrono::milliseconds{ 1000 } };

    std::optional<Timestamp> next_timestamp() const
    {
        if (cursor >= bars.size()) { return std::nullopt; }
        return bars[cursor].timestamp;
    }

    void advance() { if (cursor < bars.size()) { ++cursor; } }

    std::vector<KLine> klines(Timestamp start, Timestamp end) const
    {
        if (end < start) { return {}; }
        std::vector<KLine> out;
        for (const auto& b : bars) {
            if (b.timestamp >= start && b.timestamp <= end) { out.push_back(b); }
        }
        return out;
    }

    Timestamp::duration resolution() const { return res; }
};

inline KLine make_bar(std::int64_t ms, double close)
{
    return KLine{
        Timestamp::from_millis(ms),
        Symbol{ "X" },
        Price{ close },
        Price{ close },
        Price{ close },
        Price{ close },
        Volume{ 1.0 },
    };
}

} // namespace stonks::core::test
