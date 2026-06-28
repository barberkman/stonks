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

core::OrderID BacktestBroker::place_order(const core::OrderParams& p)
{
    const core::OrderID id = m_next_order_id++;

    const bool structural = p.quantity > 0.0
        && (p.type == core::OrderType::Market || (p.price.has_value() && *p.price > 0.0));
    const bool clear = !m_positions.contains(p.symbol);   // one position per symbol
    const bool ok = structural && clear;

    // Cancel-and-replace: a new entry supersedes any still-resting entry on the
    // same symbol, so only the latest signal is ever working. (No exits to chase.)
    if (ok) {
        for (auto& [oid, o] : m_orders) {
            if (o.symbol == p.symbol && o.status == core::OrderStatus::Open) {
                o.status = core::OrderStatus::Cancelled;
                STONKS_LOG("broker", "ev=entry_supersede id={} sym={} by={}", oid, p.symbol, id);
            }
        }
        std::erase_if(m_open_orders, [&](core::OrderID oid) {
            return m_orders.at(oid).status != core::OrderStatus::Open;
        });
    }

    STONKS_LOG("broker",
        "ev=submit id={} ts={} sym={} side={} type={} qty={:.8f} price={:.4f} sl={:.4f} tp={:.4f} "
        "structural={} clear={} status={}",
        id, log::ts_ms(m_now), p.symbol, log::side_str(p.side), log::type_str(p.type),
        p.quantity, p.price.value_or(-1.0), p.stop_loss.value_or(-1.0), p.take_profit.value_or(-1.0),
        int(structural), int(clear), ok ? "Open" : "Rejected");

    m_orders.try_emplace(id, core::Order{
        .id = id, .timestamp = m_now, .symbol = p.symbol, .side = p.side, .type = p.type,
        .status = ok ? core::OrderStatus::Open : core::OrderStatus::Rejected,
        .price = p.price, .stop_loss = p.stop_loss, .take_profit = p.take_profit,
        .quantity = p.quantity, .ttl = p.ttl });
    if (ok) { m_open_orders.push_back(id); }
    return id;
}

bool BacktestBroker::close(const core::Symbol& symbol)
{
    if (!m_positions.contains(symbol)) { return false; }
    m_pending_close.insert(symbol);
    STONKS_LOG("broker", "ev=close_request sym={}", symbol);
    return true;
}

bool BacktestBroker::update_exits(const core::Symbol& symbol,
                                  std::optional<core::Price> stop_loss,
                                  std::optional<core::Price> take_profit)
{
    if (const auto it = m_positions.find(symbol); it != m_positions.end()) {
        it->second.stop_loss = stop_loss;          // filled: re-arm the position's exits
        it->second.take_profit = take_profit;
        STONKS_LOG("broker", "ev=update_exits sym={} on=position sl={:.4f} tp={:.4f}",
                   symbol, stop_loss.value_or(-1.0), take_profit.value_or(-1.0));
        return true;
    }
    for (auto& [oid, o] : m_orders) {              // unfilled: edit the resting entry
        if (o.symbol == symbol && o.status == core::OrderStatus::Open) {
            o.stop_loss = stop_loss;
            o.take_profit = take_profit;
            STONKS_LOG("broker", "ev=update_exits sym={} on=order id={} sl={:.4f} tp={:.4f}",
                       symbol, oid, stop_loss.value_or(-1.0), take_profit.value_or(-1.0));
            return true;
        }
    }
    return false;                                  // nothing to update
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

    // ── Phase A — EXIT an existing position (explicit close > stop > target). Runs
    //    before entries, so a position opened on THIS bar (Phase B) is never checked
    //    on its own entry bar — preserving no-same-bar exits.
    const bool want_close = m_pending_close.erase(bar.symbol) > 0;   // clears stale flags too
    if (const auto it = m_positions.find(bar.symbol); it != m_positions.end()) {
        const core::Position& pos = it->second;
        const bool long_pos = pos.quantity > 0.0;
        if (want_close) {
            close_position(bar, bar.open, "close");                 // market: next-bar open
        } else {
            std::optional<core::Price> px;
            const char* reason = nullptr;
            if (pos.stop_loss &&                                    // stop wins ties: test first
                (long_pos ? bar.low <= *pos.stop_loss : bar.high >= *pos.stop_loss)) {
                px = long_pos ? std::min(*pos.stop_loss, bar.open)  // active: gap-through
                              : std::max(*pos.stop_loss, bar.open);
                reason = "stop_loss";
            } else if (pos.take_profit &&
                (long_pos ? bar.high >= *pos.take_profit : bar.low <= *pos.take_profit)) {
                px = long_pos ? std::max(*pos.take_profit, bar.open) // passive: favorable touch
                              : std::min(*pos.take_profit, bar.open);
                reason = "take_profit";
            }
            if (px) { close_position(bar, *px, reason); }
        }
    }

    // ── Phase B — match working entry orders. ──
    std::vector<core::OrderID> to_remove;
    for (core::OrderID id : m_open_orders) {
        core::Order& order = m_orders.at(id);
        if (order.status != core::OrderStatus::Open) { continue; }
        if (order.symbol != bar.symbol)              { continue; }

        // TTL expiry: an unfilled entry past its lifetime is cancelled, freeing the symbol.
        if (order.ttl && (bar.timestamp - order.timestamp) >= *order.ttl) {
            order.status = core::OrderStatus::Cancelled;
            STONKS_LOG("broker", "ev=expire id={} sym={} age_ms={} ttl_ms={}",
                       id, order.symbol, (bar.timestamp - order.timestamp).count(), order.ttl->count());
            to_remove.push_back(id);
            continue;
        }

        if (order.timestamp >= bar.timestamp)        { continue; }   // no same-bar fill / no lookahead

        const std::optional<core::Price> price = trigger_price(order, bar);
        if (!price) { continue; }

        STONKS_LOG("broker",
            "ev=trigger id={} ts={} sym={} type={} side={} order_px={:.4f} "
            "bar_o={:.4f} bar_h={:.4f} bar_l={:.4f} fill_px={:.4f}",
            id, log::ts_ms(bar.timestamp), order.symbol,
            log::type_str(order.type), log::side_str(order.side), order.price.value_or(-1.0),
            bar.open, bar.high, bar.low, *price);

        if (m_positions.contains(order.symbol)) {
            // One position per symbol: an entry never fills into an existing position.
            STONKS_LOG("broker", "ev=entry_reject id={} sym={} reason=position_exists", id, order.symbol);
            order.status = core::OrderStatus::Rejected;
        } else {
            // ENTRY: open a fresh position on a clear symbol; it inherits the SL/TP levels.
            const core::Balance cost = order.quantity * *price;
            if (cost > m_cash) {
                STONKS_LOG("broker", "ev=entry_reject id={} sym={} reason=insufficient_cash cost={:.4f} cash={:.4f}",
                           id, order.symbol, cost, m_cash);
                order.status = core::OrderStatus::Rejected;
            } else {
                [[maybe_unused]] const core::Balance cash_before = m_cash;
                m_cash -= cost;
                const core::Quantity q = (order.side == core::OrderSide::Buy ? order.quantity : -order.quantity);
                m_positions[order.symbol] = core::Position{
                    .quantity = q, .price = *price, .order_id = order.id,
                    .stop_loss = order.stop_loss, .take_profit = order.take_profit };
                record_fill(order, bar, order.quantity, *price);
                STONKS_LOG("broker",
                    "ev=entry_fill id={} sym={} side={} qty={:.8f} fill_px={:.4f} cost={:.4f} "
                    "cash_before={:.4f} cash_after={:.4f} pos_qty={:.8f} sl={:.4f} tp={:.4f}",
                    id, order.symbol, log::side_str(order.side), order.quantity, *price, cost,
                    cash_before, m_cash, q, order.stop_loss.value_or(-1.0), order.take_profit.value_or(-1.0));
            }
        }
        to_remove.push_back(id);
    }

    std::erase_if(m_open_orders, [&](core::OrderID id) {
        return std::ranges::find(to_remove, id) != to_remove.end();
    });
}

// Trigger + fill price. Limit is passive (favorable touch); Market fills at the open.
// nullopt means it did not trigger on this bar.
std::optional<core::Price> BacktestBroker::trigger_price(const core::Order& order, const core::KLine& bar) const
{
    switch (order.type) {
    case core::OrderType::Market:
        return bar.open;
    case core::OrderType::Limit:
        if (order.side == core::OrderSide::Buy) { if (bar.low  <= *order.price) return std::min(*order.price, bar.open); }
        else                                    { if (bar.high >= *order.price) return std::max(*order.price, bar.open); }
        return std::nullopt;
    }
    return std::nullopt;
}

// Close the whole position at `price`, booking collateral + realized PnL. The exit
// trade attributes back to the entry order via the position's order_id.
void BacktestBroker::close_position(const core::KLine& bar, core::Price price, const char* reason)
{
    const auto it = m_positions.find(bar.symbol);
    core::Position& pos = it->second;
    const bool long_pos = pos.quantity > 0.0;
    const core::Quantity qty = std::abs(pos.quantity);
    const core::OrderSide side = long_pos ? core::OrderSide::Sell : core::OrderSide::Buy;
    const core::Price pnl = (long_pos ? price - pos.price : pos.price - price) * qty;

    [[maybe_unused]] const core::Balance cash_before = m_cash;
    m_cash += qty * pos.price + pnl;
    m_trades.try_emplace(m_next_trade_id,
        core::Trade{ m_next_trade_id, pos.order_id, bar.timestamp, bar.symbol, side, qty, price });
    STONKS_LOG("broker",
        "ev=exit_fill reason={} tid={} order_id={} sym={} side={} qty={:.8f} fill_px={:.4f} "
        "entry_px={:.4f} pnl={:.4f} cash_before={:.4f} cash_after={:.4f}",
        reason, m_next_trade_id, pos.order_id, bar.symbol, log::side_str(side), qty, price,
        pos.price, pnl, cash_before, m_cash);
    ++m_next_trade_id;
    m_positions.erase(it);
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
