#pragma once

#include <cstdint>

namespace stonks::core {

using Price = double;
using Quantity = double;
using Timestamp = std::int64_t;

enum class Side {
    Buy,
    Sell,
};

}
