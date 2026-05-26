#include "stonks/broker/backtestbroker.h"

#include <algorithm>

namespace stonks::broker {

namespace {

bool try_fill(const core::Order& order,
              const core::KLine& bar,
              core::Balance& cash,
              std::unordered_map<core::Symbol, core::Quantity>& positions)
{
    if (order.symbol != bar.symbol) { return false; }
    if (order.timestamp >= bar.timestamp) { return false; }

    core::Price fill_price{};
    switch (order.type) {
    case core::OrderType::Market:
        fill_price = bar.open;
        break;
    case core::OrderType::Limit: {
        const core::Price limit = *order.price;
        if (order.side == core::OrderSide::Buy) {
            if (bar.low > limit) { return false; }
            fill_price = std::min(limit, bar.open);
        } else {
            if (bar.high < limit) { return false; }
            fill_price = std::max(limit, bar.open);
        }
        break;
    }
    }

    const core::Quantity qty = order.quantity;
    if (order.side == core::OrderSide::Buy) {
        const core::Balance cost = fill_price * qty;
        if (cost > cash) { return false; }
        cash -= cost;
        positions[order.symbol] += qty;
    } else {
        cash += fill_price * qty;
        positions[order.symbol] -= qty;
    }
    return true;
}

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

bool BacktestBroker::place_order(const core::Order& order)
{
    m_open_orders.push_back(order);
    return true;
}

void BacktestBroker::on_tick(const core::KLine& bar)
{
    m_last_price[bar.symbol] = bar.close;

    const auto fill_end = std::remove_if(
        m_open_orders.begin(), m_open_orders.end(),
        [&](const core::Order& order) {
            return try_fill(order, bar, m_cash, m_positions);
        });
    m_open_orders.erase(fill_end, m_open_orders.end());
}

} // namespace stonks::broker
