#pragma once

#include <concepts>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::core {

template <class BrokerT>
concept HasCash = requires(const BrokerT& broker) {
    { broker.cash() } -> std::convertible_to<Balance>;
};

template <class BrokerT>
concept HasEquity = requires(const BrokerT& broker) {
    { broker.equity() } -> std::convertible_to<Balance>;
};

template <class BrokerT>
concept HasPlaceOrder = requires(BrokerT& broker, const Order& order) {
    { broker.place_order(order) } -> std::same_as<bool>;
};

template <class BrokerT>
concept HasOnTickBar = requires(BrokerT& broker, const KLine& bar) {
    { broker.on_tick(bar) } -> std::same_as<void>;
};

template <class BrokerT>
concept HasTrades = requires(const BrokerT& broker) {
    { broker.trades() } -> std::convertible_to<std::vector<Trade>>;
};

template <class BrokerT>
concept HasOrders = requires(const BrokerT& broker) {
    { broker.orders() } -> std::convertible_to<std::vector<Order>>;
};

template <class BrokerT>
concept Broker = std::movable<BrokerT>
    && HasCash<BrokerT>
    && HasEquity<BrokerT>
    && HasPlaceOrder<BrokerT>
    && HasOnTickBar<BrokerT>
    && HasTrades<BrokerT>
    && HasOrders<BrokerT>;

} // namespace stonks::core
