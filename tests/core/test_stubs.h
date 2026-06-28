#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::core::test {

struct StubBroker
{
    std::vector<Order>* placed{ nullptr };
    std::vector<Symbol> closed;                                       // symbols passed to close()
    std::vector<std::tuple<Symbol, std::optional<Price>, std::optional<Price>>> exit_updates;
    std::unordered_map<TradeID, Trade> m_trades;
    std::unordered_map<OrderID, Order> m_orders;
    OrderID next_id{ 1 };

    Balance cash() const { return {}; }
    Balance equity() const { return {}; }
    const std::unordered_map<TradeID, Trade>& trades() const { return m_trades; }
    const std::unordered_map<OrderID, Order>& orders() const { return m_orders; }

    OrderID place_order(const OrderParams& p)
    {
        const OrderID id = next_id;
        Order o{ .id = id, .timestamp = Timestamp{}, .symbol = p.symbol, .side = p.side,
                 .type = p.type, .status = OrderStatus::Open, .price = p.price,
                 .stop_loss = p.stop_loss, .take_profit = p.take_profit,
                 .quantity = p.quantity, .ttl = p.ttl };
        if (placed) { placed->push_back(o); }
        m_orders.try_emplace(id, std::move(o));
        ++next_id;
        return id;
    }
    bool close(const Symbol& symbol) { closed.push_back(symbol); return true; }
    bool update_exits(const Symbol& symbol, std::optional<Price> stop_loss, std::optional<Price> take_profit)
    {
        exit_updates.emplace_back(symbol, stop_loss, take_profit);
        return true;
    }
    std::optional<Position> position(const Symbol&) const { return std::nullopt; }
    void on_tick(const KLine&) {}
};

// Columnar test feed mirroring KLineFeed's per-timestamp model: tests assign
// `bars` (in time order), and the feed lazily builds the same
// per-symbol-contiguous, global-time-order layout, then iterates one timestamp
// group per tick — so current_bars()/window() semantics match the real feed.
struct StubFeed
{
    std::vector<KLine> bars;          // input; set by tests before use
    Timestamp::duration res{ std::chrono::milliseconds{ 1000 } };

    std::optional<Timestamp> next_timestamp() const
    {
        ensure_built();
        if (m_group + 1 >= m_group_start.size()) { return std::nullopt; }
        return Timestamp::from_millis(m_ts[m_order[m_group_start[m_group]]]);
    }

    void advance()
    {
        ensure_built();
        if (m_group + 1 < m_group_start.size()) { ++m_group; }
    }

    std::vector<KLine> current_bars() const
    {
        ensure_built();
        std::vector<KLine> out;
        const std::uint32_t begin = m_group_start[m_group];
        const std::uint32_t end = m_group_start[m_group + 1];
        out.reserve(end - begin);
        for (std::uint32_t k = begin; k < end; ++k) {
            const std::uint32_t r = m_order[k];
            out.push_back(KLine{
                Timestamp::from_millis(m_ts[r]),
                m_id_to_ticker[m_row_symbol[r]],
                m_open[r], m_high[r], m_low[r], m_close[r], m_volume[r],
            });
        }
        return out;
    }

    MarketWindow window(int count) const
    {
        ensure_built();
        MarketWindow w;
        const std::uint32_t begin = m_group_start[m_group];
        const std::uint32_t end = m_group_start[m_group + 1];
        w.series.reserve(end - begin);
        for (std::uint32_t k = begin; k < end; ++k) {
            const std::uint32_t r = m_order[k];
            w.series.push_back(SymbolSeries{
                std::string_view{ m_id_to_ticker[m_row_symbol[r]] },
                series_for(r, count),
            });
        }
        return w;
    }

    Timestamp::duration resolution() const { return res; }

    std::size_t size() const { return bars.size(); }

private:
    // Built once from `bars`; mutable so the const accessors can lazily prepare.
    mutable bool m_built{ false };
    mutable std::vector<std::int64_t> m_ts;
    mutable std::vector<double> m_open, m_high, m_low, m_close, m_volume;
    mutable std::vector<Symbol> m_id_to_ticker;
    mutable std::vector<SymbolID> m_row_symbol;
    mutable std::vector<std::uint32_t> m_row_local;
    mutable std::vector<std::uint32_t> m_order;
    mutable std::vector<std::uint32_t> m_group_start;
    std::size_t m_group{ 0 };

    SeriesView series_for(std::uint32_t r, int count) const
    {
        const std::uint32_t available = m_row_local[r] + 1;
        const std::uint32_t cnt = (count <= 0)
            ? 0u
            : std::min(static_cast<std::uint32_t>(count), available);
        const std::uint32_t first = (cnt == 0) ? r : r - (cnt - 1);
        return SeriesView{
            std::span<const std::int64_t>{ m_ts.data() + first, cnt },
            std::span<const double>{ m_open.data() + first, cnt },
            std::span<const double>{ m_high.data() + first, cnt },
            std::span<const double>{ m_low.data() + first, cnt },
            std::span<const double>{ m_close.data() + first, cnt },
            std::span<const double>{ m_volume.data() + first, cnt },
        };
    }

    void ensure_built() const
    {
        if (m_built) { return; }
        m_built = true;
        const auto n = static_cast<std::uint32_t>(bars.size());

        std::unordered_map<std::string_view, SymbolID> intern;
        std::vector<SymbolID> file_symbol(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const auto next_id = static_cast<SymbolID>(m_id_to_ticker.size());
            const auto [it, inserted] = intern.try_emplace(bars[i].symbol, next_id);
            if (inserted) { m_id_to_ticker.push_back(bars[i].symbol); }
            file_symbol[i] = it->second;
        }
        const auto num_symbols = static_cast<SymbolID>(m_id_to_ticker.size());

        std::vector<std::uint32_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0u);
        std::stable_sort(perm.begin(), perm.end(),
            [&](std::uint32_t a, std::uint32_t b) {
                if (file_symbol[a] != file_symbol[b]) { return file_symbol[a] < file_symbol[b]; }
                return bars[a].timestamp < bars[b].timestamp;
            });

        m_ts.resize(n); m_open.resize(n); m_high.resize(n);
        m_low.resize(n); m_close.resize(n); m_volume.resize(n);
        m_row_symbol.resize(n); m_row_local.resize(n);
        std::vector<std::uint32_t> running_local(num_symbols, 0u);

        for (std::uint32_t r = 0; r < n; ++r) {
            const std::uint32_t i = perm[r];
            const SymbolID s = file_symbol[i];
            const auto& b = bars[i];
            m_ts[r] = b.timestamp.value.time_since_epoch().count();
            m_open[r] = b.open; m_high[r] = b.high; m_low[r] = b.low;
            m_close[r] = b.close; m_volume[r] = b.volume;
            m_row_symbol[r] = s;
            m_row_local[r] = running_local[s]++;
        }

        m_order.resize(n);
        std::iota(m_order.begin(), m_order.end(), 0u);
        std::stable_sort(m_order.begin(), m_order.end(),
            [&](std::uint32_t a, std::uint32_t b) {
                if (m_ts[a] != m_ts[b]) { return m_ts[a] < m_ts[b]; }
                return perm[a] < perm[b];
            });

        m_group_start.clear();
        for (std::uint32_t k = 0; k < n; ++k) {
            if (k == 0 || m_ts[m_order[k]] != m_ts[m_order[k - 1]]) {
                m_group_start.push_back(k);
            }
        }
        m_group_start.push_back(n);
    }
};

inline KLine make_bar(std::int64_t ms, const Symbol& symbol, double close)
{
    return KLine{
        Timestamp::from_millis(ms),
        symbol,
        Price{ close },
        Price{ close },
        Price{ close },
        Price{ close },
        Volume{ 1.0 },
    };
}

inline KLine make_bar(std::int64_t ms, double close)
{
    return make_bar(ms, Symbol{ "X" }, close);
}

} // namespace stonks::core::test
