#pragma once

#include "stonks/core/types.h"

class BacktestBroker
{
public:
    stonks::core::Balance cash() const { return {}; }
    stonks::core::Balance equity() const { return {}; }
    bool place_order(const stonks::core::Order&) { return true; }
};
