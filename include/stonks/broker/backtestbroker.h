#pragma once

#include <unordered_map>
#include <vector>
#include <optional>

#include "stonks/core/types.h"

namespace stonks::broker {

class BacktestBroker
{
public:
    explicit BacktestBroker(core::Balance initial_cash);

    core::Balance cash() const;
    core::Balance equity() const;
    bool bankrupt() const;
    const std::unordered_map<core::TradeID, core::Trade>& trades() const;
    const std::unordered_map<core::OrderID, core::Order>& orders() const;

    core::OrderID place_order(const core::MarketOrderParams& parameters,
                              std::optional<core::OrderID> parent = std::nullopt);
    core::OrderID place_order(const core::LimitOrderParams& parameters,
                              std::optional<core::OrderID> parent = std::nullopt);

    void on_tick(const core::KLine& bar);

private:
    core::OrderID register_order(core::Order order);

    // Cancel every still-open order in `parent`'s subtree (children, grandchildren, ...),
    // skipping `keep` (the order currently being filled, which lives in the same subtree).
    void cancel_subtree(core::OrderID parent, std::vector<core::OrderID>& to_remove,
                        core::OrderID keep = 0);

    // Force-close the whole position at `fill_price`: realize the P&L, return the
    // posted margin, record a liquidation Trade backed by a synthetic Filled order,
    // and cancel the position's bracket subtree.
    void liquidate_position(const core::Symbol& symbol, core::Price fill_price,
                            std::vector<core::OrderID>& to_remove);

    core::Balance m_cash;
    core::Timestamp m_now;
    std::unordered_map<core::Symbol, core::Position> m_positions;
    std::unordered_map<core::Symbol, core::Price> m_last_prices;
    std::unordered_map<core::OrderID, core::Order> m_orders;
    std::unordered_map<core::TradeID, core::Trade> m_trades;
    std::vector<core::OrderID> m_open_orders;
    
    core::OrderID m_next_order_id;
    core::TradeID m_next_trade_id;
    bool m_bankrupt{ false };
};

} // namespace stonks::broker
