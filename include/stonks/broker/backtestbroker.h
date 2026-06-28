#pragma once

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::broker {

// Cash-secured, one-position-per-symbol broker.
//
// A single order role: place_order opens a position. The order may carry
// stop_loss and/or take_profit price levels; when it fills, the position
// inherits them and the broker closes the whole position when a later bar
// touches either level (the stop wins ties). close() flattens a position at
// market on the next bar; update_exits() retargets the levels on a still-resting
// entry or a live position.
class BacktestBroker
{
public:
    explicit BacktestBroker(core::Balance initial_cash);

    core::Balance cash() const;
    core::Balance equity() const;
    const std::unordered_map<core::TradeID, core::Trade>& trades() const;
    const std::unordered_map<core::OrderID, core::Order>& orders() const;
    std::optional<core::Position> position(const core::Symbol& symbol) const;

    core::OrderID place_order(const core::OrderParams& parameters);   // entry (opens), may carry SL/TP
    bool close(const core::Symbol& symbol);                          // market-close at the next bar's open
    bool update_exits(const core::Symbol& symbol,                    // retarget SL/TP on the order or position
                      std::optional<core::Price> stop_loss,
                      std::optional<core::Price> take_profit);

    void on_tick(const core::KLine& bar);

private:
    std::optional<core::Price> trigger_price(const core::Order& order, const core::KLine& bar) const;
    void close_position(const core::KLine& bar, core::Price price, const char* reason);
    void record_fill(core::Order& order, const core::KLine& bar, core::Quantity quantity, core::Price price);

    core::Balance m_cash;
    core::Timestamp m_now;
    std::unordered_map<core::Symbol, core::Position> m_positions;
    std::unordered_map<core::Symbol, core::Price> m_last_prices;
    std::unordered_map<core::OrderID, core::Order> m_orders;
    std::unordered_map<core::TradeID, core::Trade> m_trades;
    std::vector<core::OrderID> m_open_orders;
    std::unordered_set<core::Symbol> m_pending_close;

    core::OrderID m_next_order_id;
    core::TradeID m_next_trade_id;
};

} // namespace stonks::broker
