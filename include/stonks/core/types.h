#pragma once

#include <chrono>
#include <compare>
#include <cstdint>
#include <cstdio>
#include <ostream>
#include <string>

namespace stonks::core {

using Price = double;
using Volume = double;
using Balance = double;
using Quantity = double;
using Symbol = std::string;

enum class OrderSide : uint8_t
{
    Buy,
    Sell,
};

enum class OrderType : uint8_t
{
    MARKET,
    LIMIT
};

enum class TimeInForce : uint8_t 
{ 
    GTC
};

struct Timestamp
{
    std::int64_t ms{};
    auto operator<=>(const Timestamp&) const = default;
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
    Timestamp timestamp;
    Symbol symbol;
    OrderType type;
    OrderSide side;
    Price price;
    Quantity quantity;
    TimeInForce time_in_force;
};

inline std::ostream& operator<<(std::ostream& os, Timestamp ts)
{
    using namespace std::chrono;
    const sys_time<milliseconds> tp{milliseconds{ts.ms}};
    const auto day_point = floor<days>(tp);
    const year_month_day ymd{day_point};
    const hh_mm_ss<milliseconds> tod{tp - day_point};

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
