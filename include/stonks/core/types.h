#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace stonks::core {

using Price = double;
using Volume = double;
using Balance = double;
using Quantity = double;

using Symbol = std::string;
using SymbolID = std::uint32_t;
using OrderID = std::uint64_t;
using TradeID = std::uint64_t;

enum class OrderSide : std::uint8_t
{
    Buy,
    Sell,
};

enum class OrderType : std::uint8_t
{
    Market,
    Limit,
    Stop,
};

enum class OrderStatus : std::uint8_t
{
    Open,
    Filled,
    Rejected,
    Cancelled
};

enum class TimeInForce : std::uint8_t
{
    GTC,
};

struct Timestamp
{
    using clock = std::chrono::system_clock;
    using duration = std::chrono::milliseconds;
    using time_point = std::chrono::sys_time<duration>;

    time_point value{};

    constexpr auto operator<=>(const Timestamp&) const = default;

    constexpr Timestamp operator+(duration d) const { return { value + d }; }
    constexpr Timestamp operator-(duration d) const { return { value - d }; }
    constexpr duration operator-(Timestamp other) const { return value - other.value; }

    static constexpr Timestamp from_millis(std::int64_t ms)
    {
        return Timestamp{ time_point{ duration{ ms } } };
    }
};

struct KLine
{
    Timestamp timestamp;
    Symbol symbol;
    Price open, high, low, close;
    Volume volume;
};

// Columnar, read-only view of one symbol's contiguous bars. The spans point
// into storage owned by the feed and stay valid for the current tick; re-query
// each tick rather than caching a view. The column layout is what lets the
// Python boundary build numpy arrays cheaply.
struct SeriesView
{
    std::span<const std::int64_t> timestamp;  // ms since epoch
    std::span<const double> open, high, low, close, volume;

    std::size_t size() const { return timestamp.size(); }
};

// One symbol's slice within a MarketWindow: its ticker plus its last-n bars.
struct SymbolSeries
{
    std::string_view symbol;   // view into the feed's intern table
    SeriesView bars;
};

// What a strategy sees on a tick: every symbol that printed at the current
// timestamp, each with its own last-n bars (ragged across symbols). Built fresh
// each tick from the timestamp's rows — re-query each tick.
struct MarketWindow
{
    std::vector<SymbolSeries> series;

    std::size_t size() const { return series.size(); }
};

struct Order
{
    OrderID id;
    std::optional<OrderID> parent_id;
    Timestamp timestamp;
    Symbol symbol;
    OrderSide side;
    OrderType type;
    OrderStatus status;
    std::optional<Price> price;
    Quantity quantity;
    TimeInForce time_in_force;
    double leverage = 1.0;   // isolated margin: applies when this order opens a position; ignored on closes
    bool reduce_only = false;   // may only reduce an existing position; cancelled if it would open or add

    auto operator<=>(const Order&) const = default;
};

struct Trade
{
    TradeID id;
    OrderID order_id;
    Timestamp timestamp;
    Symbol symbol;
    OrderSide side;
    Quantity quantity;
    Price price;
    bool liquidation = false;   // fill came from a forced close, not a strategy order
    double fee = 0.0;           // charged on this fill (already deducted from cash)

    auto operator<=>(const Trade&) const = default;
};

struct Position
{
    Quantity quantity;
    Price price;
    OrderID entry_id;
    double leverage = 1.0;   // margin divisor, fixed by the order that opened the position

    auto operator<=>(const Position&) const = default;
};

// One sample of account equity at a given timestamp. The engine records one per
// timestamp (the equity curve); the reporter derives drawdown and the time
// range from it.
struct EquityPoint
{
    Timestamp timestamp;
    Balance equity;

    auto operator<=>(const EquityPoint&) const = default;
};

// One (timestamp, value) sample of a strategy-computed indicator series,
// recorded via Context::plot for the GUI's chart overlays. Display-only:
// never read back by engine or broker logic.
struct IndicatorPoint
{
    Timestamp timestamp;
    double value;

    auto operator<=>(const IndicatorPoint&) const = default;
};

// Every point plotted during a run, keyed by symbol then series name (the
// string passed to Context::plot); each vector is chronological because
// on_tick fires once per increasing timestamp. Owned by the Engine as part of
// its run record, referenced (not owned) by Context — like the equity curve.
using IndicatorSeries = std::vector<IndicatorPoint>;
using IndicatorStore = std::map<Symbol, std::map<std::string, IndicatorSeries>>;

struct MarketOrderParams
{
    Symbol symbol;
    OrderSide side;
    Quantity quantity;
    TimeInForce time_in_force = TimeInForce::GTC;
    double leverage = 1.0;
    bool reduce_only = false;
};

struct LimitOrderParams
{
    Symbol symbol;
    OrderSide side;
    Quantity quantity;
    Price price;
    TimeInForce time_in_force = TimeInForce::GTC;
    double leverage = 1.0;
    bool reduce_only = false;
};

// Stop-market order: dormant until the market touches `price` (the trigger),
// then fills like a market order at the trigger or worse — a buy-stop triggers
// on high >= price and fills at max(price, open); a sell-stop triggers on
// low <= price and fills at min(price, open). The gap case (open beyond the
// trigger) fills at the open, mirroring the broker's liquidation fill.
struct StopOrderParams
{
    Symbol symbol;
    OrderSide side;
    Quantity quantity;
    Price price;
    TimeInForce time_in_force = TimeInForce::GTC;
    double leverage = 1.0;
    bool reduce_only = false;
};

inline std::ostream& operator<<(std::ostream& os, Timestamp ts)
{
    using namespace std::chrono;
    const auto& tp = ts.value;
    const auto day_point = floor<days>(tp);
    const year_month_day ymd{ day_point };
    const hh_mm_ss<milliseconds> tod{ tp - day_point };

    char buf[32];
    std::snprintf(buf, sizeof(buf),
        "%04d-%02u-%02uT%02d:%02d:%02d.%03dZ",
        static_cast<int>(ymd.year()),
        static_cast<unsigned>(ymd.month()),
        static_cast<unsigned>(ymd.day()),
        static_cast<int>(tod.hours().count()),
        static_cast<int>(tod.minutes().count()),
        static_cast<int>(tod.seconds().count()),
        static_cast<int>(tod.subseconds().count()));
    return os << buf;
}

} // namespace stonks::core
