#include "stonks/broker/backtestbroker.h"

#include <algorithm>
#include <cmath>
#include <optional>
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
{
    STONKS_LOG("broker", "ev=init cash={:.4f}", m_cash);
}

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
    // else nullopt (the symbol is clear). When the context is a working entry,
    // ctx_entry_ts records the tick it was placed on, to tell a fresh bracket's
    // exits (same tick as their entry) from a superseded signal's (earlier tick).
    std::optional<bool> ctx_long;
    std::optional<core::Timestamp> ctx_entry_ts;
    if (const auto it = m_positions.find(p.symbol); it != m_positions.end()) {
        ctx_long = it->second.quantity > 0.0;
    } else {
        for (const auto& [oid, o] : m_orders) {
            if (!o.reduce_only && o.symbol == p.symbol && o.status == core::OrderStatus::Open) {
                ctx_long = (o.side == core::OrderSide::Buy);
                ctx_entry_ts = o.timestamp;
                break;
            }
        }
    }

    const bool structural = p.quantity > 0.0
        && (p.type == core::OrderType::Market || (p.price.has_value() && *p.price > 0.0));
    const bool role_ok = reduce_only
        ? (ctx_long.has_value() && *ctx_long != (p.side == core::OrderSide::Buy)   // exit must oppose the context
            && (!ctx_entry_ts.has_value() || *ctx_entry_ts == m_now))              // and belong to the working entry's own bracket, not a superseded signal's
        : (!ctx_long.has_value() || ctx_entry_ts.has_value());                      // entry: clear symbol, or supersede a resting (unfilled) entry
    const bool ok = structural && role_ok;

    STONKS_LOG("broker",
        "ev=submit id={} ts={} sym={} role={} side={} type={} qty={:.8f} price={:.4f} "
        "ctx={} structural={} role_ok={} status={}",
        id, log::ts_ms(m_now), p.symbol, reduce_only ? "exit" : "entry",
        log::side_str(p.side), log::type_str(p.type), p.quantity, p.price.value_or(-1.0),
        (ctx_long ? (*ctx_long ? "long" : "short") : "clear"),
        int(structural), int(role_ok), ok ? "Open" : "Rejected");

    // Cancel-and-replace: a fresh entry supersedes a resting (unfilled) entry's whole
    // bracket, so only the latest signal's bracket is ever live on a symbol.
    if (ok && !reduce_only && ctx_entry_ts.has_value()) {
        std::vector<core::OrderID> gone;
        for (auto& [oid, o] : m_orders) {
            if (!o.reduce_only && o.symbol == p.symbol && o.status == core::OrderStatus::Open) {
                o.status = core::OrderStatus::Cancelled;
                STONKS_LOG("broker", "ev=entry_supersede id={} sym={} by={}", oid, p.symbol, id);
                gone.push_back(oid);
            }
        }
        cancel_exits(p.symbol, gone, /*keep=*/0);   // the superseded entry's pre-placed exits go too
        std::erase_if(m_open_orders, [&](core::OrderID oid) {
            return std::ranges::find(gone, oid) != gone.end();
        });
    }

    m_orders.try_emplace(id, core::Order{
        id, m_now, p.symbol, p.side, p.type,
        ok ? core::OrderStatus::Open : core::OrderStatus::Rejected,
        p.price, p.quantity, reduce_only, p.time_in_force, p.ttl });
    if (ok) { m_open_orders.push_back(id); }
    return id;
}

void BacktestBroker::on_tick(const core::KLine& bar)
{
    m_now = bar.timestamp;
    m_last_prices[bar.symbol] = bar.close;

    if (!m_open_orders.empty()) {
        STONKS_LOG("broker", "ev=bar ts={} sym={} o={:.4f} h={:.4f} l={:.4f} c={:.4f} open_orders={}",
                   log::ts_ms(bar.timestamp), bar.symbol, bar.open, bar.high, bar.low, bar.close,
                   m_open_orders.size());
    }

    std::vector<core::OrderID> to_remove;
    for (core::OrderID id : m_open_orders) {
        core::Order& order = m_orders.at(id);
        if (order.status != core::OrderStatus::Open) { continue; }
        if (order.symbol != bar.symbol)              { continue; }

        // TTL expiry: an unfilled order past its lifetime is cancelled. For an entry, its
        // pre-placed exits go too (freeing the symbol for new signals). A filled entry is
        // Filled (not Open), so this only ever hits unfilled entries.
        if (order.ttl && (bar.timestamp - order.timestamp) >= *order.ttl) {
            order.status = core::OrderStatus::Cancelled;
            STONKS_LOG("broker", "ev=expire id={} sym={} reduce_only={} age_ms={} ttl_ms={}",
                       id, order.symbol, int(order.reduce_only),
                       (bar.timestamp - order.timestamp).count(), order.ttl->count());
            if (!order.reduce_only) { cancel_exits(order.symbol, to_remove, /*keep=*/0); }
            to_remove.push_back(id);
            continue;
        }

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

        STONKS_LOG("broker",
            "ev=trigger id={} ts={} sym={} role={} type={} side={} order_px={:.4f} "
            "bar_o={:.4f} bar_h={:.4f} bar_l={:.4f} fill_px={:.4f}",
            id, log::ts_ms(bar.timestamp), order.symbol, order.reduce_only ? "exit" : "entry",
            log::type_str(order.type), log::side_str(order.side), order.price.value_or(-1.0),
            bar.open, bar.high, bar.low, *price);

        const auto pos_it = m_positions.find(order.symbol);
        const bool have_pos = (pos_it != m_positions.end());

        if (order.reduce_only) {
            // EXIT: reduce the opposing position; clamp to held size; never flips.
            if (!have_pos) {
                STONKS_LOG("broker", "ev=exit_skip id={} sym={} reason=no_position", id, order.symbol);
                continue;
            }
            core::Position& pos = pos_it->second;
            const bool pos_long = pos.quantity > 0.0;
            if (pos_long == (order.side == core::OrderSide::Buy)) {
                STONKS_LOG("broker", "ev=exit_skip id={} sym={} reason=same_direction pos_long={}",
                           id, order.symbol, int(pos_long));
                continue;   // not an opposing reducer
            }

            const core::Quantity fill_qty = std::min(order.quantity, std::abs(pos.quantity));
            [[maybe_unused]] const core::Balance cash_before = m_cash;
            [[maybe_unused]] const core::Quantity pos_before = pos.quantity;
            m_cash += fill_qty * pos.price
                    + (pos_long ? *price - pos.price : pos.price - *price) * fill_qty;   // collateral + pnl
            pos.quantity += (pos_long ? -fill_qty : fill_qty);
            record_fill(order, bar, fill_qty, *price);
            STONKS_LOG("broker",
                "ev=exit_fill id={} sym={} side={} fill_qty={:.8f} fill_px={:.4f} entry_px={:.4f} pnl={:.4f} "
                "pos_before={:.8f} pos_after={:.8f} cash_before={:.4f} cash_after={:.4f}",
                id, order.symbol, log::side_str(order.side), fill_qty, *price, pos.price,
                (pos_long ? *price - pos.price : pos.price - *price) * fill_qty,
                pos_before, pos.quantity, cash_before, m_cash);
            if (pos.quantity == 0.0) {
                m_positions.erase(pos_it);
                STONKS_LOG("broker", "ev=pos_flat sym={} by_order={}", order.symbol, id);
                cancel_exits(order.symbol, to_remove, /*keep=*/id);   // flat -> OCO
            }
        } else if (have_pos) {
            // Defensive: submit() keeps one context per symbol, so an entry should
            // never fill into an existing position; reject rather than overwrite it.
            STONKS_LOG("broker", "ev=entry_reject id={} sym={} reason=position_exists", id, order.symbol);
            order.status = core::OrderStatus::Rejected;
        } else {
            // ENTRY: open a fresh position on a clear symbol.
            const core::Balance cost = order.quantity * *price;
            if (cost > m_cash) {
                STONKS_LOG("broker", "ev=entry_reject id={} sym={} reason=insufficient_cash cost={:.4f} cash={:.4f}",
                           id, order.symbol, cost, m_cash);
                order.status = core::OrderStatus::Rejected;
                cancel_exits(order.symbol, to_remove, /*keep=*/0);   // its pre-placed exits are orphaned
            } else {
                [[maybe_unused]] const core::Balance cash_before = m_cash;
                m_cash -= cost;
                const core::Quantity q = (order.side == core::OrderSide::Buy ? order.quantity : -order.quantity);
                m_positions[order.symbol] = core::Position{ q, *price };
                record_fill(order, bar, order.quantity, *price);
                STONKS_LOG("broker",
                    "ev=entry_fill id={} sym={} side={} qty={:.8f} fill_px={:.4f} cost={:.4f} "
                    "cash_before={:.4f} cash_after={:.4f} pos_qty={:.8f}",
                    id, order.symbol, log::side_str(order.side), order.quantity, *price, cost,
                    cash_before, m_cash, q);
                // Pre-placed exits go live next bar: stamp them so they cannot fire on this one.
                for (auto& [oid, o] : m_orders) {
                    if (o.reduce_only && o.symbol == order.symbol && o.status == core::OrderStatus::Open) {
                        o.timestamp = bar.timestamp;
                        STONKS_LOG("broker", "ev=arm_exit entry_id={} exit_id={} sym={} stamped_ts={}",
                                   id, oid, order.symbol, log::ts_ms(bar.timestamp));
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
            STONKS_LOG("broker", "ev=oco_cancel id={} sym={} keep={}", oid, sym, keep);
            gone.push_back(oid);
        }
    }
}

void BacktestBroker::record_fill(core::Order& order, const core::KLine& bar, core::Quantity qty, core::Price price)
{
    order.status = core::OrderStatus::Filled;
    m_trades.try_emplace(m_next_trade_id,
        core::Trade{ m_next_trade_id, order.id, bar.timestamp, order.symbol, order.side, qty, price });
    STONKS_LOG("broker", "ev=trade tid={} order_id={} ts={} sym={} side={} qty={:.8f} px={:.4f}",
               m_next_trade_id, order.id, log::ts_ms(bar.timestamp), order.symbol,
               log::side_str(order.side), qty, price);
    ++m_next_trade_id;
}

} // namespace stonks::broker
