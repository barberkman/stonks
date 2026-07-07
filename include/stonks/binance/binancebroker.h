#pragma once

#include <functional>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "stonks/binance/exchangeinfo.h"
#include "stonks/binance/restclient.h"
#include "stonks/core/types.h"

namespace stonks::binance {

// Live USDⓈ-M Futures broker. Satisfies core::Broker, so it is a compile-time
// drop-in for BacktestBroker in Engine<Strategy, DataFeed, Broker>.
//
// State philosophy: no local financial ledger. Positions, cash, equity, orders,
// and fills are always read from Binance. The only local memory is:
//   * a per-tick read snapshot (fetched once per timestamp, discarded on any
//     mutation) so a strategy's repeated queries in one tick don't spam REST;
//   * a transient table of reduce-only bracket children whose entry has not yet
//     filled — Binance rejects reduce-only-while-flat, so they are held and
//     submitted once the parent fills (see on_tick reconciliation).
//
// Bracket linkage is reconstructed from Binance itself: each child's
// newClientOrderId encodes its parent's OrderID, so orders() rebuilds parent_id
// without a shadow map.
class BinanceBroker
{
public:
    // Production: builds a REST client from `config` and fetches exchange filters
    // once. `transport`/`now_ms` are injectable for tests (a fake transport that
    // answers /fapi/v1/exchangeInfo needs no network).
    explicit BinanceBroker(BinanceConfig config,
                           Transport transport = curl_transport(),
                           std::function<long long()> now_ms = {});

    // Test/advanced: supply pre-fetched exchange filters to skip the construction
    // -time network call.
    BinanceBroker(BinanceConfig config, ExchangeInfo exchange,
                  Transport transport, std::function<long long()> now_ms = {});

    BinanceBroker(BinanceBroker&&) = default;
    BinanceBroker& operator=(BinanceBroker&&) = default;
    BinanceBroker(const BinanceBroker&) = delete;
    BinanceBroker& operator=(const BinanceBroker&) = delete;

    // --- core::Broker surface ------------------------------------------------
    core::Balance cash() const;      // availableBalance
    core::Balance equity() const;    // totalMarginBalance (wallet + uPnL)

    std::optional<core::Position> position(const core::Symbol& symbol) const;
    std::unordered_map<core::OrderID, core::Order> orders() const;
    std::unordered_map<core::TradeID, core::Trade> trades() const;

    core::OrderID place_order(const core::MarketOrderParams& p,
                              std::optional<core::OrderID> parent = std::nullopt);
    core::OrderID place_order(const core::LimitOrderParams& p,
                              std::optional<core::OrderID> parent = std::nullopt);
    core::OrderID place_order(const core::StopOrderParams& p,
                              std::optional<core::OrderID> parent = std::nullopt);

    void on_tick(const core::KLine& bar);
    bool cancel_order(core::OrderID id);

private:
    // A reduce-only child waiting for its (resting) parent entry to fill.
    struct PendingChild
    {
        core::OrderID synthetic_id;
        core::OrderID parent_id;
        core::Symbol symbol;
        core::OrderSide side;
        core::OrderType type;                 // Limit (TP) or Stop (SL)
        core::Quantity quantity;
        std::optional<core::Price> price;
        double leverage;
    };

    struct Snapshot
    {
        core::Balance cash = 0.0;
        core::Balance equity = 0.0;
        std::unordered_map<core::Symbol, core::Position> positions;
        std::unordered_map<core::OrderID, core::Order> open_orders;
    };

    // Shared body of the three place_order overloads.
    core::OrderID place(const core::Symbol& symbol, core::OrderSide side, core::OrderType type,
                        core::Quantity quantity, std::optional<core::Price> price,
                        double leverage, bool reduce_only, std::optional<core::OrderID> parent);

    // POST the order to Binance and return its orderId. Rounds qty/price, sets
    // leverage/margin for the symbol once, encodes parent linkage into the
    // clientOrderId. Throws BinanceApiError on rejection (caller decides).
    core::OrderID submit(const core::Symbol& symbol, core::OrderSide side, core::OrderType type,
                         core::Quantity quantity, std::optional<core::Price> price,
                         double leverage, bool reduce_only, std::optional<core::OrderID> parent);

    void prepare_symbol(const core::Symbol& symbol, double leverage);
    std::string make_client_id(std::optional<core::OrderID> parent);

    void ensure_snapshot() const;
    void invalidate() const { m_snap.reset(); }
    void reconcile();

    mutable BinanceRestClient m_client;
    ExchangeInfo m_exchange;
    std::function<long long()> m_now_ms;

    mutable std::optional<Snapshot> m_snap;
    std::vector<PendingChild> m_pending;
    std::set<core::Symbol> m_prepared;   // symbols whose leverage/margin were set
    std::set<core::Symbol> m_touched;    // symbols we've traded (for trades())

    core::Timestamp m_now;
    std::optional<core::Timestamp> m_last_reconcile_ts;
    mutable core::OrderID m_next_synth;  // synthetic ids for deferred children
    long long m_nonce;                   // clientOrderId uniqueness
};

} // namespace stonks::binance
