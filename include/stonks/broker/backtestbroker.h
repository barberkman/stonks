#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <optional>

#include "stonks/core/types.h"

namespace stonks::broker {

// How one bar resolves when several orders could fill on it. Market orders
// always execute first (at the open, before any intrabar trigger can be
// touched); the policy decides stops vs limits — Conservative fills protective
// stops before profit-taking limits (a worst-case intrabar path), Optimistic
// the reverse. Placement order breaks ties within the same kind.
enum class IntrabarFillPolicy : std::uint8_t
{
    Conservative,
    Optimistic,
};

// Knobs for the fill simulation. Zero/false defaults reproduce the fee-free,
// bankruptcy-price-liquidation behavior; flat_epsilon deliberately defaults on
// (it only rounds float dust on closes).
struct BrokerConfig
{
    IntrabarFillPolicy fill_policy = IntrabarFillPolicy::Conservative;
    // Fees, charged per fill on the filled notional plus a flat amount. Maker
    // rate applies to a limit filled at its own price (it rested and was hit);
    // everything that crosses on arrival — market, stop, a limit filled at the
    // open, forced closes — pays the taker rate.
    double maker_fee_bps = 0.0;
    double taker_fee_bps = 0.0;
    double fee_per_fill = 0.0;            // flat amount in quote currency
    // Forced-close model (formulas doc §8): liquidation triggers at
    // entry*(1∓1/L)/(1∓m); at m = 0 this is the bankruptcy price. The loss cap
    // bounds a forced close's loss at the posted margin (isolated semantics).
    double maintenance_margin_rate = 0.0;
    bool isolated_loss_cap = false;
    // Float-dust guard: a close leaving |quantity| within this fraction of the
    // pre-close size snaps to exactly flat.
    double flat_epsilon = 1e-9;
    // Floors: account halts once equity <= min_equity; opening fills below
    // min_notional are rejected (closes are never blocked).
    double min_equity = 0.0;
    double min_notional = 0.0;
};

class BacktestBroker
{
public:
    explicit BacktestBroker(core::Balance initial_cash, BrokerConfig config = {});

    core::Balance cash() const;
    core::Balance equity() const;
    bool bankrupt() const;
    const std::unordered_map<core::TradeID, core::Trade>& trades() const;
    const std::unordered_map<core::OrderID, core::Order>& orders() const;

    // The open position on `symbol`, or nullopt when flat (one per symbol).
    std::optional<core::Position> position(const core::Symbol& symbol) const;

    core::OrderID place_order(const core::MarketOrderParams& parameters,
                              std::optional<core::OrderID> parent = std::nullopt);
    core::OrderID place_order(const core::LimitOrderParams& parameters,
                              std::optional<core::OrderID> parent = std::nullopt);
    core::OrderID place_order(const core::StopOrderParams& parameters,
                              std::optional<core::OrderID> parent = std::nullopt);

    void on_tick(const core::KLine& bar);

    // Cancel a still-Open order and its dormant bracket subtree. Returns false
    // when the id is unknown, already terminal, or the account is bankrupt.
    bool cancel_order(core::OrderID id);

private:
    core::OrderID register_order(core::Order order);

    // Attempt to fill an eligible order (Open, this bar's symbol, parent filled,
    // past the lookahead gate) against `bar`. Returns true when the order left
    // the Open state — filled, or rejected with its bracket subtree cancelled.
    bool try_fill(core::Order& order, const core::KLine& bar,
                  std::vector<core::OrderID>& to_remove);

    // Cancel every still-open order in `parent`'s subtree (children, grandchildren, ...),
    // skipping `keep` (the order currently being filled, which lives in the same subtree).
    void cancel_subtree(core::OrderID parent, std::vector<core::OrderID>& to_remove,
                        core::OrderID keep = 0);

    // Force-close the whole position at `fill_price`: realize the P&L, return the
    // posted margin, record a liquidation Trade backed by a synthetic Filled order,
    // and cancel the position's bracket subtree.
    void liquidate_position(const core::Symbol& symbol, core::Price fill_price,
                            std::vector<core::OrderID>& to_remove);

    // Drop the given ids from the m_open_orders working set.
    void prune_open_orders(const std::vector<core::OrderID>& to_remove);

    core::Balance m_cash;
    BrokerConfig m_config;
    core::Timestamp m_now;
    std::unordered_map<core::Symbol, core::Position> m_positions;
    std::unordered_map<core::Symbol, core::Price> m_last_prices;
    std::unordered_map<core::OrderID, core::Order> m_orders;
    std::unordered_map<core::TradeID, core::Trade> m_trades;
    std::vector<core::OrderID> m_open_orders;
    
    core::OrderID m_next_order_id;
    core::TradeID m_next_trade_id;
    bool m_bankrupt{ false };
};

} // namespace stonks::broker
