#include "stonks/broker/backtestbroker.h"

namespace stonks::broker {

core::Balance BacktestBroker::cash() const { return {}; }

core::Balance BacktestBroker::equity() const { return {}; }

bool BacktestBroker::place_order(const core::Order&) { return true; }

} // namespace stonks::broker
