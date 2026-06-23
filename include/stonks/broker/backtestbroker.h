#pragma once

#include <unordered_map>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::broker {

class BacktestBroker
{
public:
    explicit BacktestBroker(core::Balance initial_cash);

    core::Balance cash() const;
    core::Balance equity() const;
    const std::unordered_map<core::TradeID, core::Trade>& trades() const;
    const std::unordered_map<core::OrderID, core::Order>& orders() const;

    core::OrderID place_order(const core::MarketOrderParams& parameters);
    core::OrderID place_order(const core::LimitOrderParams& parameters);

    void on_tick(const core::KLine& bar);

private:
    core::OrderID register_order(core::Order order);

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
