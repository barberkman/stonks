#pragma once

#include <cstdint>
#include <string>

namespace stonks::core {

using Price = double;
using Volume = double;
using Balance = double;
using Quantity = double;
using Timestamp = std::int64_t;
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

}
