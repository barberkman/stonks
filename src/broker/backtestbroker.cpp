#include "stonks/broker/backtestbroker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "stonks/core/log.h"
#include "stonks/core/logfmt.h"
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
        STONKS_LOG("broker", "ev=epos sym={} qty={:.6f} entry={:.4f} mark={:.4f} reserved={:.4f} upnl={:.4f} marked={}",
                   symbol, pos.quantity, pos.price,
                   (it != m_last_prices.end() ? it->second : pos.price),
                   reserved, upnl, int(it != m_last_prices.end()));
    }
    STONKS_LOG("broker", "ev=equity cash={:.4f} equity={:.4f} npos={}", m_cash, e, m_positions.size());
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

    STONKS_LOG("broker", "ev=bar ts={} sym={} o={:.4f} h={:.4f} l={:.4f} c={:.4f} open_orders={}",
               log::ts_ms(bar.timestamp), bar.symbol, bar.open, bar.high, bar.low, bar.close, m_open_orders.size());

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
            STONKS_LOG("broker", "ev=gate id={} why=lookahead order_ts={} bar_ts={}",
                       id, log::ts_ms(order.timestamp), log::ts_ms(bar.timestamp));
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
                    STONKS_LOG("broker", "ev=gate id={} why=limit_buy_no_trigger limit={:.4f} bar_low={:.4f}",
                               id, limit, bar.low);
                    continue; // price never reached limit
                }
                price = std::min(limit, bar.open);
            } else {
                if (bar.high < limit) {
                    STONKS_LOG("broker", "ev=gate id={} why=limit_sell_no_trigger limit={:.4f} bar_high={:.4f}",
                               id, limit, bar.high);
                    continue; // price never reached limit
                }
                price = std::max(limit, bar.open);
            }
            break;
        }
        }

        // Observational state for the ev=fill log; set in the branches below.
        [[maybe_unused]] const core::Balance cash_before = m_cash;
        [[maybe_unused]] const char* fill_kind = "";
        [[maybe_unused]] core::Price fill_entry = 0.0;
        [[maybe_unused]] core::Balance fill_cost = 0.0;
        [[maybe_unused]] core::Balance fill_collateral = 0.0;
        [[maybe_unused]] core::Balance fill_pnl = 0.0;
        [[maybe_unused]] core::Quantity pos_qty_before = 0.0;
        [[maybe_unused]] core::Quantity pos_qty_after = 0.0;

        // Postiion logic
        const auto position_it = m_positions.find(order.symbol);
        core::Quantity filled_qty;
        if (position_it == m_positions.end()) {
            // No position
            const core::Balance cost = order.quantity * price;
            if (cost > m_cash) {
                STONKS_LOG("broker", "ev=reject id={} why=insufficient_cash cost={:.4f} cash={:.4f}",
                           id, cost, m_cash);
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
            fill_kind = (order.side == core::OrderSide::Buy ? "open_long" : "open_short");
            fill_entry = price;
            fill_cost = cost;
            pos_qty_before = 0.0;
            pos_qty_after = qty;
        } else {
            // Position exist
            core::Position& position = position_it->second;
            const bool position_is_long = position.quantity > 0.0;
            const bool order_is_buy = (order.side == core::OrderSide::Buy);

            // Same direction as the position: we never add to a position.
            if (position_is_long == order_is_buy) {
                STONKS_LOG("broker", "ev=reject id={} why=same_side_add pos_qty={:.6f} order_side={}",
                           id, position.quantity, log::side_str(order.side));
                order.status = core::OrderStatus::Rejected;
                cancel_subtree(id, to_remove);              // order died -> kill its dormant children
                to_remove.push_back(id);
                continue;
            }

            // Opposite direction: close part or all of it.
            const core::Price entry_at_close = position.price;
            const core::Quantity pos_before_close = position.quantity;
            filled_qty = std::min(order.quantity, std::abs(position.quantity));
            const core::Balance pnl = (position_is_long ? price - position.price
                                                        : position.price - price) * filled_qty;
            m_cash += filled_qty * position.price + pnl;
            position.quantity += (position_is_long ? -filled_qty : filled_qty);

            // Observational state for the ev=fill log; read before any erase below.
            fill_kind = (position.quantity == 0.0 ? "close_full" : "close_partial");
            fill_entry = entry_at_close;
            fill_collateral = filled_qty * entry_at_close;
            fill_pnl = pnl;
            pos_qty_before = pos_before_close;
            pos_qty_after = position.quantity;

            if (position.quantity == 0.0) {
                // flat -> cancel the whole bracket (all levels), keep the leg filling now.
                cancel_subtree(position.entry_id, to_remove, id);
                m_positions.erase(position_it);
            }
        }

        // Recort the fill
        order.status = core::OrderStatus::Filled;
        STONKS_LOG("broker",
            "ev=fill trade={} id={} ts={} sym={} side={} type={} kind={} req_qty={:.6f} fill_qty={:.6f} "
            "price={:.4f} bar_open={:.4f} limit={:.4f} entry={:.4f} cost={:.4f} collateral={:.4f} pnl={:.4f} "
            "cash_before={:.4f} cash_after={:.4f} pos_before={:.6f} pos_after={:.6f}",
            m_next_trade_id, order.id, log::ts_ms(bar.timestamp), order.symbol,
            log::side_str(order.side), log::type_str(order.type), fill_kind,
            order.quantity, filled_qty, price, bar.open, order.price.value_or(0.0),
            fill_entry, fill_cost, fill_collateral, fill_pnl, cash_before, m_cash,
            pos_qty_before, pos_qty_after);
        m_trades.try_emplace(m_next_trade_id, core::Trade{
            m_next_trade_id, order.id, bar.timestamp, order.symbol, order.side, filled_qty, price
        });
        ++m_next_trade_id;

        // Update timestamp orders chained to this order
        for (auto& [c_id, c_order] : m_orders) {
            if (c_order.parent_id == order.id && c_order.status == core::OrderStatus::Open) {
                c_order.timestamp = bar.timestamp;
                STONKS_LOG("broker", "ev=arm parent={} child={} new_ts={}",
                           order.id, c_id, log::ts_ms(bar.timestamp));
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

#ifdef STONKS_LOG_ENABLED
    // Break the combined `valid` flag back into its reasons for the audit log.
    const bool qty_ok = order.quantity > 0.0;
    const bool price_ok = order.type != core::OrderType::Limit
                          || (order.price.has_value() && *order.price > 0.0);
    bool parent_ok = true;
    if (order.parent_id.has_value()) {
        const auto pit = m_orders.find(*order.parent_id);
        parent_ok = (pit != m_orders.end() && pit->second.status == core::OrderStatus::Open);
    }
    STONKS_LOG("broker",
        "ev=place id={} parent={} ts={} sym={} side={} type={} qty={:.6f} price={:.4f} status={} "
        "qty_ok={} price_ok={} parent_ok={}",
        order.id, order.parent_id.value_or(0), log::ts_ms(order.timestamp), order.symbol,
        log::side_str(order.side), log::type_str(order.type), order.quantity,
        order.price.value_or(0.0), (valid ? "Open" : "Rejected"),
        int(qty_ok), int(price_ok), int(parent_ok));
#endif

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
                STONKS_LOG("broker", "ev=cancel id={} parent={} keep={}", oid, parent, keep);
            }
            cancel_subtree(oid, to_remove, keep);   // recurse to reach grandchildren
        }
    }
}

} // namespace stonks::broker
