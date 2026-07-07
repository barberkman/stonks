#include "stonks/binance/binancebroker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "stonks/core/broker.h"
#include "stonks/core/log.h"

namespace stonks::binance {

using namespace stonks::core;

namespace {

// Synthetic OrderIDs for deferred (not-yet-submitted) bracket children start
// here — far above any real Binance orderId, so the two id spaces never overlap.
constexpr OrderID kSyntheticBase = OrderID{ 1 } << 62;

std::function<long long()> default_clock()
{
    return [] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    };
}

const char* side_str(OrderSide s) { return s == OrderSide::Buy ? "BUY" : "SELL"; }

// A numeric field that Binance may encode as a JSON string ("123.45") or number.
double dnum(const nlohmann::json& j, const char* key)
{
    if (!j.contains(key) || j[key].is_null()) { return 0.0; }
    const auto& v = j[key];
    if (v.is_string()) {
        try { return std::stod(v.get<std::string>()); } catch (...) { return 0.0; }
    }
    if (v.is_number()) { return v.get<double>(); }
    return 0.0;
}

std::int64_t inum(const nlohmann::json& j, const char* key)
{
    if (!j.contains(key) || j[key].is_null()) { return 0; }
    const auto& v = j[key];
    if (v.is_number()) { return v.get<std::int64_t>(); }
    if (v.is_string()) {
        try { return std::stoll(v.get<std::string>()); } catch (...) { return 0; }
    }
    return 0;
}

// Format a value to at most `prec` decimals, trimming trailing zeros (and a bare
// decimal point). Binance rejects both scientific notation and more decimals than
// the symbol's precision, so we snap to exactly what the filters allow.
std::string fmt_num(double v, int prec)
{
    if (prec < 0) { prec = 0; }
    if (prec > 8) { prec = 8; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    std::string s{ buf };
    if (s.find('.') != std::string::npos) {
        while (s.back() == '0') { s.pop_back(); }
        if (s.back() == '.') { s.pop_back(); }
    }
    return s;
}

OrderType map_type(const std::string& binance_type)
{
    if (binance_type.find("STOP") != std::string::npos) { return OrderType::Stop; }
    if (binance_type.find("TAKE_PROFIT") != std::string::npos) { return OrderType::Limit; }
    if (binance_type == "MARKET") { return OrderType::Market; }
    return OrderType::Limit;
}

// clientOrderId scheme: entries "stk-e-<nonce>", children "stk-c-<parent>-<nonce>".
// Recover the parent OrderID from a child id (nullopt for entries / foreign ids).
std::optional<OrderID> decode_parent(const std::string& client_id)
{
    constexpr std::string_view child_prefix = "stk-c-";
    if (client_id.rfind(child_prefix, 0) != 0) { return std::nullopt; }
    const std::string rest = client_id.substr(child_prefix.size());
    const auto dash = rest.find('-');
    const std::string digits = dash == std::string::npos ? rest : rest.substr(0, dash);
    try { return OrderID{ std::stoull(digits) }; } catch (...) { return std::nullopt; }
}

Order parse_order(const nlohmann::json& o)
{
    Order ord{};
    ord.id = static_cast<OrderID>(inum(o, "orderId"));
    ord.parent_id = decode_parent(o.value("clientOrderId", std::string{}));
    ord.timestamp = Timestamp::from_millis(inum(o, "updateTime"));
    ord.symbol = o.value("symbol", std::string{});
    ord.side = o.value("side", std::string{ "BUY" }) == "BUY" ? OrderSide::Buy : OrderSide::Sell;
    const std::string btype = o.value("type", std::string{});
    ord.type = map_type(btype);
    ord.status = OrderStatus::Open;
    const bool trigger = btype.find("STOP") != std::string::npos
                         || btype.find("TAKE_PROFIT") != std::string::npos;
    const double stop = dnum(o, "stopPrice");
    const double price = dnum(o, "price");
    if (trigger && stop > 0.0) { ord.price = stop; }
    else if (price > 0.0) { ord.price = price; }
    ord.quantity = dnum(o, "origQty");
    ord.time_in_force = TimeInForce::GTC;
    ord.leverage = 1.0;
    ord.reduce_only = o.value("reduceOnly", false);
    return ord;
}

} // namespace

BinanceBroker::BinanceBroker(BinanceConfig config, Transport transport,
                             std::function<long long()> now_ms)
: m_client{ config, transport, now_ms },
  m_exchange{},
  m_now_ms{ now_ms ? std::move(now_ms) : default_clock() },
  m_next_synth{ kSyntheticBase },
  m_nonce{ m_now_ms() }
{
    m_exchange = ExchangeInfo::fetch(m_client);
}

BinanceBroker::BinanceBroker(BinanceConfig config, ExchangeInfo exchange,
                             Transport transport, std::function<long long()> now_ms)
: m_client{ config, transport, now_ms },
  m_exchange{ std::move(exchange) },
  m_now_ms{ now_ms ? std::move(now_ms) : default_clock() },
  m_next_synth{ kSyntheticBase },
  m_nonce{ m_now_ms() }
{}

// --- reads -------------------------------------------------------------------

void BinanceBroker::ensure_snapshot() const
{
    if (m_snap) { return; }

    Snapshot s;
    const nlohmann::json acct = m_client.signed_request("GET", "/fapi/v3/account", {});
    s.cash = dnum(acct, "availableBalance");
    s.equity = dnum(acct, "totalMarginBalance");

    const nlohmann::json pr = m_client.signed_request("GET", "/fapi/v3/positionRisk", {});
    for (const auto& p : pr) {
        const double amt = dnum(p, "positionAmt");
        if (amt == 0.0) { continue; }
        const Symbol sym = p.value("symbol", std::string{});
        s.positions[sym] = Position{ amt, dnum(p, "entryPrice"), OrderID{ 0 },
                                     p.contains("leverage") ? dnum(p, "leverage") : 1.0 };
    }

    const nlohmann::json oo = m_client.signed_request("GET", "/fapi/v1/openOrders", {});
    for (const auto& o : oo) {
        const Order ord = parse_order(o);
        s.open_orders[ord.id] = ord;
    }
    m_snap = std::move(s);
}

Balance BinanceBroker::cash() const { ensure_snapshot(); return m_snap->cash; }
Balance BinanceBroker::equity() const { ensure_snapshot(); return m_snap->equity; }

std::optional<Position> BinanceBroker::position(const Symbol& symbol) const
{
    ensure_snapshot();
    const auto it = m_snap->positions.find(symbol);
    if (it == m_snap->positions.end() || it->second.quantity == 0.0) { return std::nullopt; }
    return it->second;
}

std::unordered_map<OrderID, Order> BinanceBroker::orders() const
{
    ensure_snapshot();
    std::unordered_map<OrderID, Order> out = m_snap->open_orders;
    // Deferred children aren't on Binance yet — surface them as synthetic Open
    // orders so a strategy can still query/cancel the id it was handed.
    for (const auto& pc : m_pending) {
        Order o{};
        o.id = pc.synthetic_id;
        o.parent_id = pc.parent_id;
        o.timestamp = m_now;
        o.symbol = pc.symbol;
        o.side = pc.side;
        o.type = pc.type;
        o.status = OrderStatus::Open;
        o.price = pc.price;
        o.quantity = pc.quantity;
        o.time_in_force = TimeInForce::GTC;
        o.leverage = pc.leverage;
        o.reduce_only = true;
        out[pc.synthetic_id] = o;
    }
    return out;
}

std::unordered_map<TradeID, Trade> BinanceBroker::trades() const
{
    std::unordered_map<TradeID, Trade> out;
    for (const auto& sym : m_touched) {
        try {
            const nlohmann::json ut =
                m_client.signed_request("GET", "/fapi/v1/userTrades", { { "symbol", sym } });
            for (const auto& t : ut) {
                const TradeID id = static_cast<TradeID>(inum(t, "id"));
                Trade tr{};
                tr.id = id;
                tr.order_id = static_cast<OrderID>(inum(t, "orderId"));
                tr.timestamp = Timestamp::from_millis(inum(t, "time"));
                tr.symbol = sym;
                tr.side = t.value("side", std::string{ "BUY" }) == "BUY" ? OrderSide::Buy
                                                                         : OrderSide::Sell;
                tr.quantity = dnum(t, "qty");
                tr.price = dnum(t, "price");
                tr.liquidation = false;
                tr.fee = dnum(t, "commission");
                out[id] = tr;
            }
        } catch (const BinanceApiError&) {
            // Skip a symbol we can't query rather than fail the whole report.
        }
    }
    return out;
}

// --- placement ---------------------------------------------------------------

core::OrderID BinanceBroker::place_order(const MarketOrderParams& p, std::optional<OrderID> parent)
{
    return place(p.symbol, p.side, OrderType::Market, p.quantity, std::nullopt,
                 p.leverage, p.reduce_only, parent);
}
core::OrderID BinanceBroker::place_order(const LimitOrderParams& p, std::optional<OrderID> parent)
{
    return place(p.symbol, p.side, OrderType::Limit, p.quantity, p.price,
                 p.leverage, p.reduce_only, parent);
}
core::OrderID BinanceBroker::place_order(const StopOrderParams& p, std::optional<OrderID> parent)
{
    return place(p.symbol, p.side, OrderType::Stop, p.quantity, p.price,
                 p.leverage, p.reduce_only, parent);
}

core::OrderID BinanceBroker::place(const Symbol& symbol, OrderSide side, OrderType type,
                                   Quantity quantity, std::optional<Price> price,
                                   double leverage, bool reduce_only,
                                   std::optional<OrderID> parent)
{
    m_touched.insert(symbol);

    // A reduce-only order needs an open position; Binance rejects it while flat.
    // Defer it until the (resting) entry fills — see reconcile().
    if (reduce_only && !position(symbol).has_value()) {
        const OrderID sid = m_next_synth++;
        m_pending.push_back(PendingChild{ sid, parent.value_or(0), symbol, side, type,
                                          quantity, price, leverage });
        STONKS_LOG("binance", "ev=defer_child synth={} parent={} sym={} type={}",
                   sid, parent.value_or(0), symbol, static_cast<int>(type));
        return sid;
    }

    try {
        const OrderID id = submit(symbol, side, type, quantity, price, leverage, reduce_only, parent);
        invalidate();
        return id;
    } catch (const BinanceApiError& e) {
        // A single rejected order must not tear down the live loop.
        STONKS_LOG("binance", "ev=place_rejected sym={} code={} msg={}", symbol, e.code, e.what());
        invalidate();
        return m_next_synth++;   // untracked id; the strategy sees no fill and moves on
    }
}

core::OrderID BinanceBroker::submit(const Symbol& symbol, OrderSide side, OrderType type,
                                    Quantity quantity, std::optional<Price> price,
                                    double leverage, bool reduce_only,
                                    std::optional<OrderID> parent)
{
    prepare_symbol(symbol, leverage);

    const SymbolFilters& f = m_exchange.filters(symbol);
    const Quantity q = m_exchange.round_qty(symbol, quantity);
    if (q <= 0.0) {
        throw BinanceApiError{ 0, 0, "quantity rounds below step size for " + symbol };
    }

    QueryParams p;
    p.emplace_back("symbol", symbol);
    p.emplace_back("side", side_str(side));
    switch (type) {
    case OrderType::Market:
        p.emplace_back("type", "MARKET");
        break;
    case OrderType::Limit:
        p.emplace_back("type", "LIMIT");
        p.emplace_back("timeInForce", "GTC");
        p.emplace_back("price", fmt_num(m_exchange.round_price(symbol, *price), f.price_precision));
        break;
    case OrderType::Stop:
        // Backtest stop = market-on-trigger against the last price.
        p.emplace_back("type", "STOP_MARKET");
        p.emplace_back("stopPrice", fmt_num(m_exchange.round_price(symbol, *price), f.price_precision));
        p.emplace_back("workingType", "CONTRACT_PRICE");
        break;
    }
    p.emplace_back("quantity", fmt_num(q, f.qty_precision));
    if (reduce_only) { p.emplace_back("reduceOnly", "true"); }
    p.emplace_back("newClientOrderId", make_client_id(parent));
    p.emplace_back("newOrderRespType", "RESULT");

    if (m_client.config().dry_run) {
        const OrderID sid = m_next_synth++;
        STONKS_LOG("binance", "ev=dry_run_order synth={} sym={} side={} type={} qty={} reduce={}",
                   sid, symbol, side_str(side), static_cast<int>(type), fmt_num(q, f.qty_precision),
                   int(reduce_only));
        return sid;
    }

    const nlohmann::json resp = m_client.signed_request("POST", "/fapi/v1/order", std::move(p));
    const OrderID id = static_cast<OrderID>(inum(resp, "orderId"));
    STONKS_LOG("binance", "ev=submitted id={} sym={} side={} type={} qty={} reduce={} parent={}",
               id, symbol, side_str(side), static_cast<int>(type), fmt_num(q, f.qty_precision),
               int(reduce_only), parent.value_or(0));
    return id;
}

void BinanceBroker::prepare_symbol(const Symbol& symbol, double leverage)
{
    if (m_prepared.contains(symbol)) { return; }
    const int lev = std::max(1, static_cast<int>(std::llround(leverage)));
    try {
        m_client.signed_request("POST", "/fapi/v1/leverage",
                                { { "symbol", symbol }, { "leverage", std::to_string(lev) } });
    } catch (const BinanceApiError& e) {
        STONKS_LOG("binance", "ev=set_leverage_failed sym={} code={} msg={}", symbol, e.code, e.what());
    }
    try {
        m_client.signed_request("POST", "/fapi/v1/marginType",
                                { { "symbol", symbol }, { "marginType", "ISOLATED" } });
    } catch (const BinanceApiError& e) {
        // -4046 "No need to change margin type" is expected once it's set.
        STONKS_LOG("binance", "ev=set_margin_note sym={} code={} msg={}", symbol, e.code, e.what());
    }
    m_prepared.insert(symbol);
}

std::string BinanceBroker::make_client_id(std::optional<OrderID> parent)
{
    const long long n = m_nonce++;
    if (parent.has_value()) {
        return "stk-c-" + std::to_string(*parent) + "-" + std::to_string(n);
    }
    return "stk-e-" + std::to_string(n);
}

// --- reconciliation ----------------------------------------------------------

void BinanceBroker::on_tick(const KLine& bar)
{
    m_now = bar.timestamp;
    // Reconcile once per distinct timestamp (on_tick fires per printing symbol).
    if (m_last_reconcile_ts && *m_last_reconcile_ts == bar.timestamp) { return; }
    m_last_reconcile_ts = bar.timestamp;

    invalidate();
    try {
        ensure_snapshot();
        reconcile();
    } catch (const std::exception& e) {
        // A transient reconcile failure shouldn't kill the loop; reads retry.
        STONKS_LOG("binance", "ev=reconcile_error what={}", e.what());
    }
}

void BinanceBroker::reconcile()
{
    // 1) Arm deferred children whose parent entry has filled.
    std::vector<PendingChild> still_pending;
    for (const auto& pc : m_pending) {
        const bool parent_open = m_snap->open_orders.contains(pc.parent_id);
        const auto pit = m_snap->positions.find(pc.symbol);
        const bool have_pos = pit != m_snap->positions.end() && pit->second.quantity != 0.0;

        if (parent_open) {
            still_pending.push_back(pc);   // entry still resting; keep waiting
        } else if (have_pos) {
            try {
                submit(pc.symbol, pc.side, pc.type, pc.quantity, pc.price, pc.leverage,
                       /*reduce_only=*/true, pc.parent_id);
                invalidate();
                STONKS_LOG("binance", "ev=arm_child parent={} sym={}", pc.parent_id, pc.symbol);
            } catch (const BinanceApiError& e) {
                STONKS_LOG("binance", "ev=arm_failed parent={} sym={} msg={}",
                           pc.parent_id, pc.symbol, e.what());
            }
        } else {
            STONKS_LOG("binance", "ev=drop_child parent={} sym={} why=entry_gone_flat",
                       pc.parent_id, pc.symbol);
        }
    }
    m_pending = std::move(still_pending);

    // Arming may have invalidated the snapshot; refresh for cleanup.
    ensure_snapshot();

    // 2) Orphan cleanup: a flat symbol should carry no resting protection.
    std::vector<std::pair<Symbol, OrderID>> to_cancel;
    for (const auto& [id, ord] : m_snap->open_orders) {
        const auto pit = m_snap->positions.find(ord.symbol);
        const bool flat = pit == m_snap->positions.end() || pit->second.quantity == 0.0;
        if (flat && ord.reduce_only) { to_cancel.emplace_back(ord.symbol, id); }
    }
    for (const auto& [sym, id] : to_cancel) {
        try {
            m_client.signed_request("DELETE", "/fapi/v1/order",
                                    { { "symbol", sym }, { "orderId", std::to_string(id) } });
            STONKS_LOG("binance", "ev=cancel_orphan sym={} id={}", sym, id);
        } catch (const BinanceApiError&) {}
    }
    if (!to_cancel.empty()) { invalidate(); }
}

bool BinanceBroker::cancel_order(OrderID id)
{
    // A deferred child that was never submitted: just forget it.
    const auto pend = std::find_if(m_pending.begin(), m_pending.end(),
                                   [id](const PendingChild& pc) { return pc.synthetic_id == id; });
    if (pend != m_pending.end()) {
        m_pending.erase(pend);
        return true;
    }

    ensure_snapshot();
    const auto it = m_snap->open_orders.find(id);
    if (it == m_snap->open_orders.end()) { return false; }   // unknown or already terminal
    const Symbol symbol = it->second.symbol;

    try {
        m_client.signed_request("DELETE", "/fapi/v1/order",
                                { { "symbol", symbol }, { "orderId", std::to_string(id) } });
    } catch (const BinanceApiError& e) {
        STONKS_LOG("binance", "ev=cancel_failed id={} code={} msg={}", id, e.code, e.what());
        invalidate();
        return false;
    }

    // Cascade to this order's children (live on Binance, or still deferred).
    for (const auto& [cid, ord] : m_snap->open_orders) {
        if (ord.parent_id == id) {
            try {
                m_client.signed_request("DELETE", "/fapi/v1/order",
                                        { { "symbol", ord.symbol }, { "orderId", std::to_string(cid) } });
            } catch (const BinanceApiError&) {}
        }
    }
    std::erase_if(m_pending, [id](const PendingChild& pc) { return pc.parent_id == id; });

    invalidate();
    return true;
}

} // namespace stonks::binance

// Compile-time guarantee that BinanceBroker is a drop-in for BacktestBroker.
static_assert(stonks::core::Broker<stonks::binance::BinanceBroker>,
              "BinanceBroker must satisfy the core::Broker concept");
