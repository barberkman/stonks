#include "stonks/broker/backtestbroker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::broker {

namespace {
// STONKS_LOG uses std::format, which can't print scoped enums or Timestamp
// directly; these turn them into format-friendly values. [[maybe_unused]]
// because they are unreferenced when logging is compiled out.
[[maybe_unused]] const char* side_str(core::OrderSide s) { return s == core::OrderSide::Buy ? "Buy" : "Sell"; }
[[maybe_unused]] const char* type_str(core::OrderType t) { return t == core::OrderType::Market ? "Market" : "Limit"; }
[[maybe_unused]] std::int64_t ts_ms(core::Timestamp t) { return t.value.time_since_epoch().count(); }
} // namespace

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

core::OrderID BacktestBroker::place_order(const core::MarketOrderParams& parameters,
                                          std::optional<core::OrderID> parent)
{
    return register_order(core::Order
    {
        core::OrderID{ m_next_order_id++ },
        parent,
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

core::OrderID BacktestBroker::place_order(const core::LimitOrderParams& parameters,
                                          std::optional<core::OrderID> parent)
{
    return register_order(core::Order
    {
        core::OrderID{ m_next_order_id++ },
        parent,
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

    // Process the open orders; collect the ids that reached a terminal state
    // (filled / rejected) and drop them from the working set afterwards.
    std::vector<core::OrderID> to_remove;

    for (core::OrderID id : m_open_orders) {
        // Fetch order
        core::Order& order = m_orders.at(id);
        if (order.status != core::OrderStatus::Open) { continue; }
        if (order.symbol != bar.symbol) { continue; } // Only this symbol's order react to this bar

        // Child order stays dormant until parent has filled
        if (order.parent_id.has_value()) {
            const auto order_it = m_orders.find(order.parent_id.value());
            if (order_it == m_orders.end() || order_it->second.status != core::OrderStatus::Filled) {
                continue;   // parent missing or not yet filled -> stay dormant
            }
        }

        // Lookahead gate
        if (order.timestamp >= bar.timestamp) {
            continue;
        }

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
                if (bar.low > limit) {
                    continue; // price never reached limit
                }
                price = std::min(limit, bar.open);
            } else {
                if (bar.high < limit) {
                    continue; // price never reached limit
                }
                price = std::max(limit, bar.open);
            }
            break;
        }
        }

        // Postiion logic
        const auto position_it = m_positions.find(order.symbol);
        core::Quantity filled_qty;
        if (position_it == m_positions.end()) {
            // No position
            const core::Balance cost = order.quantity * price;
            if (cost > m_cash) {
                order.status = core::OrderStatus::Rejected; // Not enough cash
                cancel_subtree(id, to_remove);              // entry died -> kill its dormant children
                to_remove.push_back(id);
                continue;
            }
            m_cash -= cost;
            const core::Quantity qty = (order.side == core::OrderSide::Buy ? order.quantity
                                                                           : -order.quantity);
            m_positions[order.symbol] = core::Position{ qty, price, order.id };
            filled_qty = order.quantity;
        } else {
            // Position exist
            core::Position& position = position_it->second;
            const bool position_is_long = position.quantity > 0.0;
            const bool order_is_buy = (order.side == core::OrderSide::Buy);

            // Same direction as the position: we never add to a position.
            if (position_is_long == order_is_buy) {
                order.status = core::OrderStatus::Rejected;
                cancel_subtree(id, to_remove);              // order died -> kill its dormant children
                to_remove.push_back(id);
                continue;
            }

            // Opposite direction: close part or all of it.
            filled_qty = std::min(order.quantity, std::abs(position.quantity));
            const core::Balance pnl = (position_is_long ? price - position.price
                                                        : position.price - price) * filled_qty;
            m_cash += filled_qty * position.price + pnl;
            position.quantity += (position_is_long ? -filled_qty : filled_qty);
            if (position.quantity == 0.0) {
                // flat -> cancel the whole bracket (all levels), keep the leg filling now.
                cancel_subtree(position.entry_id, to_remove, id);
                m_positions.erase(position_it);
            }
        }

        // Recort the fill
        order.status = core::OrderStatus::Filled;
        m_trades.try_emplace(m_next_trade_id, core::Trade{
            m_next_trade_id, order.id, bar.timestamp, order.symbol, order.side, filled_qty, price
        });
        ++m_next_trade_id;

        // Update timestamp orders chained to this order
        for (auto& [c_id, c_order] : m_orders) {
            if (c_order.parent_id == order.id && c_order.status == core::OrderStatus::Open) {
                c_order.timestamp = bar.timestamp;
            }
        }

        to_remove.push_back(id);
    }

    // Drop the terminal orders from the working set
    std::erase_if(m_open_orders, [&](core::OrderID id) {
        return std::ranges::find(to_remove, id) != to_remove.end();
    });
}

core::OrderID BacktestBroker::register_order(core::Order order)
{
    const core::OrderID id = order.id;
    bool valid = order.quantity > 0.0 &&
        ( order.type != core::OrderType::Limit || (order.price.has_value() && *order.price > 0.0) );

    // Find the parent if parent_id exist
    if (order.parent_id.has_value()) {
        auto order_it = m_orders.find(order.parent_id.value());
        if (order_it == m_orders.end() || order_it->second.status != core::OrderStatus::Open) {
            valid = false;
        }
    }

    order.status = valid ? core::OrderStatus::Open : core::OrderStatus::Rejected;

    m_orders.try_emplace(id, std::move(order));
    if (valid) { m_open_orders.push_back(id); }
    return id;
}

void BacktestBroker::cancel_subtree(core::OrderID parent, std::vector<core::OrderID>& to_remove,
                                    core::OrderID keep)
{
    for (auto& [oid, o] : m_orders) {
        if (o.parent_id == parent && o.status == core::OrderStatus::Open) {
            if (oid != keep) {
                o.status = core::OrderStatus::Cancelled;
                to_remove.push_back(oid);
            }
            cancel_subtree(oid, to_remove, keep);   // recurse to reach grandchildren
        }
    }
}

} // namespace stonks::broker
