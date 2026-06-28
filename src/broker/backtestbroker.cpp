#include "stonks/broker/backtestbroker.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

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
            upnl = (pos.quantity > 0.0 ? it->second - pos.price
                                       : pos.price - it->second) * std::abs(pos.quantity);
        }
        e += reserved + upnl;
    }
    return e;
}

const std::unordered_map<core::TradeID, core::Trade>& BacktestBroker::trades() const { return m_trades; }
const std::unordered_map<core::OrderID, core::Order>& BacktestBroker::orders() const { return m_orders; }

std::optional<core::Position> BacktestBroker::position(const core::Symbol& symbol) const
{
    const auto it = m_positions.find(symbol);
    return it == m_positions.end() ? std::nullopt : std::optional{ it->second };
}

core::OrderID BacktestBroker::place_order(const core::OrderParams& p) { return submit(p, false); }
core::OrderID BacktestBroker::place_exit (const core::OrderParams& p) { return submit(p, true ); }

core::OrderID BacktestBroker::submit(const core::OrderParams& p, bool reduce_only)
{
    const core::OrderID id = m_next_order_id++;

    // The symbol's committed direction: a live position, else a working entry,
    // else nullopt (the symbol is clear).
    std::optional<bool> ctx_long;
    if (const auto it = m_positions.find(p.symbol); it != m_positions.end()) {
        ctx_long = it->second.quantity > 0.0;
    } else {
        for (const auto& [oid, o] : m_orders) {
            if (!o.reduce_only && o.symbol == p.symbol && o.status == core::OrderStatus::Open) {
                ctx_long = (o.side == core::OrderSide::Buy);
                break;
            }
        }
    }

    const bool structural = p.quantity > 0.0
        && (p.type == core::OrderType::Market || (p.price.has_value() && *p.price > 0.0));
    const bool role_ok = reduce_only
        ? (ctx_long.has_value() && *ctx_long != (p.side == core::OrderSide::Buy))   // exit must oppose the context
        : !ctx_long.has_value();                                                    // entry needs a clear symbol
    const bool ok = structural && role_ok;

    m_orders.try_emplace(id, core::Order{
        id, m_now, p.symbol, p.side, p.type,
        ok ? core::OrderStatus::Open : core::OrderStatus::Rejected,
        p.price, p.quantity, reduce_only, p.time_in_force });
    if (ok) { m_open_orders.push_back(id); }
    return id;
}

void BacktestBroker::on_tick(const core::KLine& bar)
{
    m_now = bar.timestamp;
    m_last_prices[bar.symbol] = bar.close;

    std::vector<core::OrderID> to_remove;
    for (core::OrderID id : m_open_orders) {
        core::Order& order = m_orders.at(id);
        if (order.status != core::OrderStatus::Open) { continue; }
        if (order.symbol != bar.symbol)              { continue; }
        if (order.timestamp >= bar.timestamp)        { continue; }   // no same-bar fill / no lookahead

        // Trigger + fill price. Limit = passive (favorable touch); Stop = active
        // (adverse cross). nullopt means it did not trigger on this bar.
        std::optional<core::Price> price;
        switch (order.type) {
        case core::OrderType::Market:
            price = bar.open;
            break;
        case core::OrderType::Limit:
            if (order.side == core::OrderSide::Buy) { if (bar.low  <= *order.price) { price = std::min(*order.price, bar.open); } }
            else                                    { if (bar.high >= *order.price) { price = std::max(*order.price, bar.open); } }
            break;
        case core::OrderType::Stop:
            if (order.side == core::OrderSide::Buy) { if (bar.high >= *order.price) { price = std::max(*order.price, bar.open); } }
            else                                    { if (bar.low  <= *order.price) { price = std::min(*order.price, bar.open); } }
            break;
        }
        if (!price) { continue; }

        const auto pos_it = m_positions.find(order.symbol);
        const bool have_pos = (pos_it != m_positions.end());

        if (order.reduce_only) {
            // EXIT: reduce the opposing position; clamp to held size; never flips.
            if (!have_pos) { continue; }
            core::Position& pos = pos_it->second;
            const bool pos_long = pos.quantity > 0.0;
            if (pos_long == (order.side == core::OrderSide::Buy)) { continue; }   // not an opposing reducer

            const core::Quantity fill_qty = std::min(order.quantity, std::abs(pos.quantity));
            m_cash += fill_qty * pos.price
                    + (pos_long ? *price - pos.price : pos.price - *price) * fill_qty;   // collateral + pnl
            pos.quantity += (pos_long ? -fill_qty : fill_qty);
            record_fill(order, bar, fill_qty, *price);
            if (pos.quantity == 0.0) {
                m_positions.erase(pos_it);
                cancel_exits(order.symbol, to_remove, /*keep=*/id);   // flat -> OCO
            }
        } else if (have_pos) {
            // Defensive: submit() keeps one context per symbol, so an entry should
            // never fill into an existing position; reject rather than overwrite it.
            order.status = core::OrderStatus::Rejected;
        } else {
            // ENTRY: open a fresh position on a clear symbol.
            const core::Balance cost = order.quantity * *price;
            if (cost > m_cash) {
                order.status = core::OrderStatus::Rejected;
                cancel_exits(order.symbol, to_remove, /*keep=*/0);   // its pre-placed exits are orphaned
            } else {
                m_cash -= cost;
                const core::Quantity q = (order.side == core::OrderSide::Buy ? order.quantity : -order.quantity);
                m_positions[order.symbol] = core::Position{ q, *price };
                record_fill(order, bar, order.quantity, *price);
                // Pre-placed exits go live next bar: stamp them so they cannot fire on this one.
                for (auto& [oid, o] : m_orders) {
                    if (o.reduce_only && o.symbol == order.symbol && o.status == core::OrderStatus::Open) {
                        o.timestamp = bar.timestamp;
                    }
                }
            }
        }
        to_remove.push_back(id);
    }

    std::erase_if(m_open_orders, [&](core::OrderID id) {
        return std::ranges::find(to_remove, id) != to_remove.end();
    });
}

void BacktestBroker::cancel_exits(const core::Symbol& sym, std::vector<core::OrderID>& gone, core::OrderID keep)
{
    for (auto& [oid, o] : m_orders) {
        if (o.reduce_only && o.symbol == sym && o.status == core::OrderStatus::Open && oid != keep) {
            o.status = core::OrderStatus::Cancelled;
            gone.push_back(oid);
        }
    }
}

void BacktestBroker::record_fill(core::Order& order, const core::KLine& bar, core::Quantity qty, core::Price price)
{
    order.status = core::OrderStatus::Filled;
    m_trades.try_emplace(m_next_trade_id,
        core::Trade{ m_next_trade_id, order.id, bar.timestamp, order.symbol, order.side, qty, price });
    ++m_next_trade_id;
}

} // namespace stonks::broker
