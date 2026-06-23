#include "stonks/broker/backtestbroker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include "stonks/core/log.h"
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

    STONKS_LOG("broker", "mark sym={} open={:.4f} close={:.4f} ts={} open_orders={}",
        bar.symbol, bar.open, bar.close, ts_ms(bar.timestamp), m_open_orders.size());

    // Process the open orders, remove if can not filled
    const auto fill_end = std::remove_if(
        m_open_orders.begin(), m_open_orders.end(),
        [&](core::OrderID id) {
            // Fetch order
            core::Order& order = m_orders.at(id);
            if (order.symbol != bar.symbol) { return false; } // Only this symbol's order react to this bar

            // Lookahead gate
            if (order.timestamp >= bar.timestamp) {
                STONKS_LOG("broker", "no-fill GATE id={} sym={} order_ts={} >= bar_ts={}",
                    order.id, order.symbol, ts_ms(order.timestamp), ts_ms(bar.timestamp));
                return false;
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
                        STONKS_LOG("broker", "no-fill LIMIT id={} sym={} buy limit={:.4f} < bar_low={:.4f} (lingers)",
                            order.id, order.symbol, limit, bar.low);
                        return false; // price never reached limit
                    }
                    price = std::min(limit, bar.open);
                } else {
                    if (bar.high < limit) {
                        STONKS_LOG("broker", "no-fill LIMIT id={} sym={} sell limit={:.4f} > bar_high={:.4f} (lingers)",
                            order.id, order.symbol, limit, bar.high);
                        return false; // price never reached limit
                    }
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
                    STONKS_LOG("broker", "no-fill CASH id={} sym={} need={:.4f} have={:.4f} -> REJECT",
                        order.id, order.symbol, cost, m_cash);
                    order.status = core::OrderStatus::Rejected; // Not enough cash
                    return true;
                }
                m_cash -= cost;
                const core::Quantity qty = (order.side == core::OrderSide::Buy ? order.quantity
                                                                               : -order.quantity);
                m_positions[order.symbol] = core::Position{ qty, price };
                filled_qty = order.quantity;
                STONKS_LOG("broker", "open id={} sym={} side={} qty={:.6f} entry={:.4f} cost={:.4f} cash={:.4f}",
                    order.id, order.symbol, side_str(order.side), qty, price, cost, m_cash);
            } else {
                // Position exist
                core::Position& position = position_it->second;
                const bool position_is_long = position.quantity > 0.0;
                const bool order_is_buy = (order.side == core::OrderSide::Buy);

                // Same direction as the position: we never add to a position.
                if (position_is_long == order_is_buy) {
                    STONKS_LOG("broker", "no-fill ADD id={} sym={} side={} pos_qty={:.6f} (one position per symbol) -> REJECT",
                        order.id, order.symbol, side_str(order.side), position.quantity);
                    order.status = core::OrderStatus::Rejected;
                    return true;
                }

                // Opposite direction: close part or all of it.
                filled_qty = std::min(order.quantity, std::abs(position.quantity));
                const core::Balance pnl = (position_is_long ? price - position.price
                                                            : position.price - price) * filled_qty;
                m_cash += filled_qty * position.price + pnl;
                position.quantity += (position_is_long ? -filled_qty : filled_qty);
                STONKS_LOG("broker", "close id={} sym={} side={} filled={:.6f} entry={:.4f} exit={:.4f} pnl={:.4f} cash={:.4f} remaining={:.6f}",
                    order.id, order.symbol, side_str(order.side), filled_qty,
                    position.price, price, pnl, m_cash, position.quantity);
                if (position.quantity == 0.0) { m_positions.erase(position_it); }
            }

            // Recort the fill
            order.status = core::OrderStatus::Filled;
            STONKS_LOG("broker", "FILL trade_id={} order_id={} sym={} side={} qty={:.6f} price={:.4f} bar_ts={} cash={:.4f}",
                m_next_trade_id, order.id, order.symbol, side_str(order.side),
                filled_qty, price, ts_ms(bar.timestamp), m_cash);
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

    if (valid) {
        STONKS_LOG("broker", "queue order id={} sym={} side={} type={} qty={:.6f} price={:.4f} ts={} open_orders={}",
            id, order.symbol, side_str(order.side), type_str(order.type),
            order.quantity, order.price.value_or(0.0), ts_ms(order.timestamp), m_open_orders.size());
    } else {
        STONKS_LOG("broker", "reject placement id={} sym={} reason={} qty={:.6f} price={:.4f}",
            id, order.symbol,
            (order.quantity > 0.0 ? "non-positive-limit-price" : "non-positive-quantity"),
            order.quantity, order.price.value_or(0.0));
    }

    m_orders.try_emplace(id, std::move(order));
    if (valid) { m_open_orders.push_back(id); }
    return id;
}

} // namespace stonks::broker
