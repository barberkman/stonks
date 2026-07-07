#include "stonks/binance/exchangeinfo.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace stonks::binance {

namespace {

// Binance encodes numeric filter values as strings ("0.00100000"). Parse to
// double, tolerating a missing/non-string field by returning 0.
double num(const nlohmann::json& j, const char* key)
{
    if (!j.contains(key)) { return 0.0; }
    const auto& v = j[key];
    if (v.is_string()) { return std::stod(v.get<std::string>()); }
    if (v.is_number()) { return v.get<double>(); }
    return 0.0;
}

// Decimals implied by a step like 0.001 -> 3. Falls back to 8 for a zero/absent
// step. Used only when the symbol omits the explicit precision fields.
int decimals_of(double step)
{
    if (step <= 0.0) { return 8; }
    int d = 0;
    double s = step;
    while (s < 1.0 && d < 12) { s *= 10.0; ++d; }
    return d;
}

double snap(double value, double granularity, int precision, bool floor_it)
{
    if (granularity > 0.0) {
        const double steps = value / granularity;
        const double n = floor_it ? std::floor(steps + 1e-9) : std::round(steps);
        value = n * granularity;
    }
    const double scale = std::pow(10.0, precision);
    return std::round(value * scale) / scale;
}

} // namespace

ExchangeInfo::ExchangeInfo(const nlohmann::json& exchange_info)
{
    if (!exchange_info.contains("symbols")) { return; }
    for (const auto& sym : exchange_info["symbols"]) {
        SymbolFilters f;
        for (const auto& filt : sym.value("filters", nlohmann::json::array())) {
            const std::string type = filt.value("filterType", "");
            if (type == "LOT_SIZE") {
                f.step_size = num(filt, "stepSize");
            } else if (type == "PRICE_FILTER") {
                f.tick_size = num(filt, "tickSize");
            } else if (type == "MIN_NOTIONAL") {
                // Futures uses "notional"; some payloads use "minNotional".
                f.min_notional = filt.contains("notional") ? num(filt, "notional")
                                                            : num(filt, "minNotional");
            }
        }
        f.qty_precision = sym.contains("quantityPrecision")
            ? sym["quantityPrecision"].get<int>() : decimals_of(f.step_size);
        f.price_precision = sym.contains("pricePrecision")
            ? sym["pricePrecision"].get<int>() : decimals_of(f.tick_size);
        m_filters.emplace(sym.value("symbol", ""), f);
    }
}

ExchangeInfo ExchangeInfo::fetch(BinanceRestClient& client)
{
    return ExchangeInfo{ client.public_request("GET", "/fapi/v1/exchangeInfo", {}) };
}

const SymbolFilters& ExchangeInfo::filters(const core::Symbol& symbol) const
{
    const auto it = m_filters.find(symbol);
    return it != m_filters.end() ? it->second : m_default;
}

core::Quantity ExchangeInfo::round_qty(const core::Symbol& symbol, core::Quantity qty) const
{
    const SymbolFilters& f = filters(symbol);
    return snap(qty, f.step_size, f.qty_precision, /*floor_it=*/true);
}

core::Price ExchangeInfo::round_price(const core::Symbol& symbol, core::Price price) const
{
    const SymbolFilters& f = filters(symbol);
    return snap(price, f.tick_size, f.price_precision, /*floor_it=*/false);
}

bool ExchangeInfo::passes_min_notional(const core::Symbol& symbol,
                                       core::Quantity qty, core::Price price) const
{
    const SymbolFilters& f = filters(symbol);
    if (f.min_notional <= 0.0) { return true; }
    return qty * price >= f.min_notional;
}

} // namespace stonks::binance
