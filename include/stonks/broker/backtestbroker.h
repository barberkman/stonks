#pragma once

#include <optional>
#include <unordered_map>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::broker {

// Cash-secured, one-position-per-symbol broker.
//
// Orders come in two roles, chosen by which method you call:
//   place_order -> an ENTRY; opens a position, only valid on a clear symbol.
//   place_exit  -> a reduce-only EXIT (stop-loss / take-profit); only reduces an
//                  existing opposite position, never opens or flips it.
//
// Exits attach to a position implicitly, by symbol: at most one context (a live
// position or a working entry) exists per symbol, so "the exits on symbol S"
// unambiguously belong to S's position. When the position goes flat, its exits
// are cancelled (OCO).
class BacktestBroker
{
public:
    explicit BacktestBroker(core::Balance initial_cash);

    core::Balance cash() const;
    core::Balance equity() const;
    const std::unordered_map<core::TradeID, core::Trade>& trades() const;
    const std::unordered_map<core::OrderID, core::Order>& orders() const;
    std::optional<core::Position> position(const core::Symbol& symbol) const;

    core::OrderID place_order(const core::OrderParams& parameters);   // entry (opens)
    core::OrderID place_exit (const core::OrderParams& parameters);   // reduce-only exit (SL/TP)

    void on_tick(const core::KLine& bar);

private:
    core::OrderID submit(const core::OrderParams& parameters, bool reduce_only);
    void cancel_exits(const core::Symbol& symbol, std::vector<core::OrderID>& to_remove, core::OrderID keep);
    void record_fill(core::Order& order, const core::KLine& bar, core::Quantity quantity, core::Price price);

    core::Balance m_cash;
    core::Timestamp m_now;
    std::unordered_map<core::Symbol, core::Position> m_positions;
    std::unordered_map<core::Symbol, core::Price> m_last_prices;
    std::unordered_map<core::OrderID, core::Order> m_orders;
    std::unordered_map<core::TradeID, core::Trade> m_trades;
    std::vector<core::OrderID> m_open_orders;

    core::OrderID m_next_order_id;
    core::TradeID m_next_trade_id;
};

} // namespace stonks::broker
