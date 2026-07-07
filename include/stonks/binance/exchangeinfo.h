#pragma once

#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "stonks/binance/restclient.h"
#include "stonks/core/types.h"

namespace stonks::binance {

// The order-validity filters Binance enforces per symbol. An order whose
// quantity/price aren't multiples of step/tick, or whose notional is below the
// minimum, is rejected — so the broker snaps to these before sending.
struct SymbolFilters
{
    double step_size = 0.0;      // LOT_SIZE: quantity granularity
    double tick_size = 0.0;      // PRICE_FILTER: price granularity
    double min_notional = 0.0;   // MIN_NOTIONAL: floor on quantity*price
    int qty_precision = 8;       // decimals to snap quantity to (kills float dust)
    int price_precision = 8;     // decimals to snap price to
};

// Per-symbol filter cache, parsed once from GET /fapi/v1/exchangeInfo. Immutable
// after construction; the broker consults it to round and validate every order.
class ExchangeInfo
{
public:
    ExchangeInfo() = default;

    // Parse a /fapi/v1/exchangeInfo payload. Symbols without the expected filters
    // are still recorded with permissive defaults (no rounding, no min-notional).
    explicit ExchangeInfo(const nlohmann::json& exchange_info);

    // Fetch and parse in one step.
    static ExchangeInfo fetch(BinanceRestClient& client);

    // Floor a quantity down to the symbol's step size (never rounds up past what
    // the strategy asked to trade), then snaps to qty_precision decimals.
    core::Quantity round_qty(const core::Symbol& symbol, core::Quantity qty) const;

    // Round a price to the nearest tick, snapped to price_precision decimals.
    core::Price round_price(const core::Symbol& symbol, core::Price price) const;

    // True when quantity*price meets the symbol's MIN_NOTIONAL (always true when
    // the symbol has no such filter).
    bool passes_min_notional(const core::Symbol& symbol, core::Quantity qty, core::Price price) const;

    bool has(const core::Symbol& symbol) const { return m_filters.contains(symbol); }
    const SymbolFilters& filters(const core::Symbol& symbol) const;

private:
    std::unordered_map<core::Symbol, SymbolFilters> m_filters;
    SymbolFilters m_default{};   // returned for unknown symbols: no constraints
};

} // namespace stonks::binance
