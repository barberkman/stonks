#include "stonks/broker/backtestbroker.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <utility>

#include "stonks/core/types.h"

namespace stonks::broker {

BacktestBroker::BacktestBroker(core::Balance initial_cash)
: m_cash{ initial_cash },
  m_now{},
  m_next_order_id{ 1 },
  m_next_trade_id{ 1 }
{}

core::Balance BacktestBroker::cash() const { return m_cash; }

core::Balance BacktestBroker::equity() const
{
    core::Balance e = m_cash;
    for (const auto& [symbol, pos] : m_positions) {
        const core::Balance reserved = std::abs(pos.quantity) * pos.price;
        core::Balance upnl = 0.0;
        const auto it = m_last_prices.find(symbol);
        if (it != m_last_prices.end()) {
            const core::Price mark = it->second;
            upnl = (pos.quantity > 0.0 ? mark - pos.price
                                       : pos.price - mark) * std::abs(pos.quantity);
        }
        e += reserved + upnl;
    }
    return e;
}

const std::unordered_map<core::TradeID, core::Trade>& BacktestBroker::trades() const { return m_trades; }

const std::unordered_map<core::OrderID, core::Order>& BacktestBroker::orders() const { return m_orders; }

core::OrderID BacktestBroker::place_order(const core::MarketOrderParams& parameters)
{
    return register_order(core::Order
    {
        core::OrderID{ m_next_order_id++ },
        m_now,
        parameters.symbol,
        parameters.side,
        core::OrderType::Market,
        core::OrderStatus::Open,
        std::nullopt,
        parameters.quantity,
        parameters.time_in_force
    });
}

core::OrderID BacktestBroker::place_order(const core::LimitOrderParams& parameters)
{
    return register_order(core::Order
    {
        core::OrderID{ m_next_order_id++ },
        m_now,
        parameters.symbol,
        parameters.side,
        core::OrderType::Limit,
        core::OrderStatus::Open,
        parameters.price,
        parameters.quantity,
        parameters.time_in_force
    });
}

void BacktestBroker::on_tick(const core::KLine& bar)
{
    // Save current time
    m_now = bar.timestamp;

    // Update symbol last price
    m_last_prices[bar.symbol] = bar.close;

    // Process the open orders, remove if can not filled
    const auto fill_end = std::remove_if(
        m_open_orders.begin(), m_open_orders.end(),
        [&](core::OrderID id) { 
            // Fetch order
            core::Order& order = m_orders.at(id);
            if (order.symbol != bar.symbol) { return false; } // Only this symbol's order react to this bar

            // Lookahead gate
            if (order.timestamp >= bar.timestamp) { return false; }

            // Calculate the fill price
            core::Price price;
            switch (order.type) 
            {
            case core::OrderType::Market:
                price = bar.open;
                break;
            case core::OrderType::Limit:
            {
                core::Price limit = *order.price;
                if (order.side == core::OrderSide::Buy) {
                    if (bar.low > limit) { return false; } // price never reached limit
                    price = std::min(limit, bar.open);
                } else {
                    if (bar.high < limit) { return false; } // price never reached limit
                    price = std::max(limit, bar.open);
                }
                break;
            }
            }

            const auto position_it = m_positions.find(order.symbol);
            core::Quantity filled_qty;

            if (position_it == m_positions.end()) {
                // No position
                const core::Balance cost = order.quantity * price;
                if (cost > m_cash) {
                    order.status = core::OrderStatus::Rejected; // Not enough cash
                    return true;
                }
                m_cash -= cost;
                const core::Quantity qty = (order.side == core::OrderSide::Buy ? order.quantity
                                                                               : -order.quantity);
                m_positions[order.symbol] = core::Position{ qty, price };
                filled_qty = order.quantity;
            } else {
                // Position exist
                core::Position& position = position_it->second;
                const bool position_is_long = position.quantity > 0.0;
                const bool order_is_buy = (order.side == core::OrderSide::Buy);

                // Same direction as the position: we never add to a position.
                if (position_is_long == order_is_buy) {
                    order.status = core::OrderStatus::Rejected;
                    return true;
                }

                // Opposite direction: close part or all of it.
                filled_qty = std::min(order.quantity, std::abs(position.quantity));
                const core::Balance pnl = (position_is_long ? price - position.price
                                                            : position.price - price) * filled_qty;
                m_cash += filled_qty * position.price + pnl;
                position.quantity += (position_is_long ? -filled_qty : filled_qty);
                if (position.quantity == 0.0) { m_positions.erase(position_it); }
            }

            // Recort the fill
            order.status = core::OrderStatus::Filled;
            m_trades.try_emplace(m_next_trade_id, core::Trade{
                m_next_trade_id, order.id, bar.timestamp, order.symbol, order.side, filled_qty, price
            });
            ++m_next_trade_id;
            return true;
    });
    m_open_orders.erase(fill_end, m_open_orders.end());
}

core::OrderID BacktestBroker::register_order(core::Order order)
{
    const core::OrderID id = order.id;
    const bool valid = order.quantity > 0.0 && 
        ( order.type != core::OrderType::Limit || (order.price.has_value() && *order.price > 0.0) );
    order.status = valid ? core::OrderStatus::Open : core::OrderStatus::Rejected;
    m_orders.try_emplace(id, std::move(order));
    if (valid) { m_open_orders.push_back(id); }
    return id;
}

} // namespace stonks::broker
