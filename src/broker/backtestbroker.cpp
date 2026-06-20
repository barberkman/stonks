#include "stonks/broker/backtestbroker.h"

#include <algorithm>

#include "stonks/core/log.h"

namespace stonks::broker {

namespace {
[[maybe_unused]] const char* side_str(core::OrderSide s) { return s == core::OrderSide::Buy ? "Buy" : "Sell"; }
[[maybe_unused]] const char* type_str(core::OrderType t) { return t == core::OrderType::Market ? "Market" : "Limit"; }
} // namespace

BacktestBroker::BacktestBroker(core::Balance initial_cash)
: m_cash{ initial_cash }
{}

core::Balance BacktestBroker::cash() const { return m_cash; }

core::Balance BacktestBroker::equity() const
{
    core::Balance e = m_cash;
    for (const auto& [symbol, qty] : m_positions) {
        const auto it = m_last_price.find(symbol);
        if (it != m_last_price.end()) {
            e += qty * it->second;
        }
    }
    return e;
}

const std::vector<core::Trade>& BacktestBroker::trades() const { return m_trades; }

const std::vector<core::Order>& BacktestBroker::orders() const { return m_order_log; }

bool BacktestBroker::place_order(const core::Order& order)
{
    m_order_log.push_back(order);
    m_open_orders.push_back(order);
    STONKS_LOG("broker", "queue order id={} sym={} side={} type={} qty={} price={} ts={} open_orders={}",
        order.id, order.symbol, side_str(order.side), type_str(order.type),
        order.quantity, order.price.value_or(0.0),
        order.timestamp.value.time_since_epoch().count(), m_open_orders.size());
    return true;
}

void BacktestBroker::on_tick(const core::KLine& bar)
{
    m_last_price[bar.symbol] = bar.close;
    STONKS_LOG("broker", "mark sym={} close={} open={} ts={} open_orders={}",
        bar.symbol, bar.close, bar.open, bar.timestamp.value.time_since_epoch().count(),
        m_open_orders.size());

    const auto fill_end = std::remove_if(
        m_open_orders.begin(), m_open_orders.end(),
        [&](const core::Order& order) { return try_fill(order, bar); });
    m_open_orders.erase(fill_end, m_open_orders.end());
}

bool BacktestBroker::try_fill(const core::Order& order, const core::KLine& bar)
{
    if (order.symbol != bar.symbol) { return false; }

    // No-lookahead gate: an order cannot fill against a bar at or before the
    // timestamp it was placed on.
    if (order.timestamp >= bar.timestamp) {
        STONKS_LOG("broker", "no-fill GATE id={} sym={} order_ts={} >= bar_ts={}",
            order.id, order.symbol,
            order.timestamp.value.time_since_epoch().count(),
            bar.timestamp.value.time_since_epoch().count());
        return false;
    }

    core::Price fill_price{};
    switch (order.type) {
    case core::OrderType::Market:
        fill_price = bar.open;
        break;
    case core::OrderType::Limit: {
        const core::Price limit = *order.price;
        if (order.side == core::OrderSide::Buy) {
            if (bar.low > limit) {
                STONKS_LOG("broker", "no-fill LIMIT id={} sym={} buy limit={:.4f} < bar_low={:.4f} (lingers)",
                    order.id, order.symbol, limit, bar.low);
                return false;
            }
            fill_price = std::min(limit, bar.open);
        } else {
            if (bar.high < limit) {
                STONKS_LOG("broker", "no-fill LIMIT id={} sym={} sell limit={:.4f} > bar_high={:.4f} (lingers)",
                    order.id, order.symbol, limit, bar.high);
                return false;
            }
            fill_price = std::max(limit, bar.open);
        }
        break;
    }
    }

    const core::Quantity qty = order.quantity;
    if (order.side == core::OrderSide::Buy) {
        const core::Balance cost = fill_price * qty;
        if (cost > m_cash) {
            STONKS_LOG("broker", "no-fill CASH id={} sym={} need={:.4f} have={:.4f} (lingers)",
                order.id, order.symbol, cost, m_cash);
            return false;
        }
        m_cash -= cost;
        m_positions[order.symbol] += qty;
    } else {
        m_cash += fill_price * qty;
        m_positions[order.symbol] -= qty;
    }

    STONKS_LOG("broker",
        "FILL trade_id={} order_id={} sym={} side={} qty={} price={} bar_ts={} cash_after={} pos_after={}",
        static_cast<std::uint64_t>(m_next_trade_id), order.id, order.symbol,
        side_str(order.side), qty, fill_price,
        bar.timestamp.value.time_since_epoch().count(),
        m_cash, m_positions[order.symbol]);

    m_trades.push_back(core::Trade{
        m_next_trade_id++,
        order.id,
        bar.timestamp,
        order.symbol,
        order.side,
        qty,
        fill_price,
    });
    return true;
}

} // namespace stonks::broker
