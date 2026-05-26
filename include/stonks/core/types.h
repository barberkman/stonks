#pragma once

#include <chrono>
#include <compare>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

namespace stonks::core {

using Price = double;
using Volume = double;
using Balance = double;
using Quantity = double;

using Symbol = std::string;
using OrderID = std::uint64_t;

enum class OrderSide : std::uint8_t
{
    Buy,
    Sell,
};

enum class OrderType : std::uint8_t
{
    Market,
    Limit,
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

struct Order
{
    OrderID id;
    Timestamp timestamp;
    Symbol symbol;
    OrderSide side;
    OrderType type;
    std::optional<Price> price;
    Quantity quantity;
    TimeInForce time_in_force;

    auto operator<=>(const Order&) const = default;

private:
    Order(OrderID id_, Timestamp ts, Symbol sym, OrderSide s, OrderType t,
          std::optional<Price> p, Quantity q, TimeInForce tif)
    : id{ id_ }, timestamp{ ts }, symbol{ std::move(sym) }, side{ s }, type{ t },
      price{ p }, quantity{ q }, time_in_force{ tif }
    {}

    template <class BrokerT, class DataFeedT> friend class Context;
};

struct MarketOrderParams
{
    Symbol symbol;
    OrderSide side;
    Quantity quantity;
    TimeInForce time_in_force = TimeInForce::GTC;
};

struct LimitOrderParams
{
    Symbol symbol;
    OrderSide side;
    Quantity quantity;
    Price price;
    TimeInForce time_in_force = TimeInForce::GTC;
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
