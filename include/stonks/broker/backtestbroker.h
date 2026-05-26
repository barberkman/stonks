#pragma once

#include "stonks/core/types.h"

namespace stonks::broker {

class BacktestBroker
{
public:
    core::Balance cash() const { return {}; }
    core::Balance equity() const { return {}; }
    bool place_order(const core::Order&) { return true; }
};

} // namespace stonks::broker
