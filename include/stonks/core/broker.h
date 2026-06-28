#pragma once

#include <concepts>
#include <unordered_map>

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
concept HasPlaceOrder = requires (BrokerT& broker, const OrderParams& params,
                                  const Symbol& symbol, std::optional<Price> level) {
    { broker.place_order(params) }                  -> std::same_as<OrderID>;
    { broker.close(symbol) }                        -> std::same_as<bool>;
    { broker.update_exits(symbol, level, level) }   -> std::same_as<bool>;
};

template <class BrokerT>
concept HasOnTickBar = requires(BrokerT& broker, const KLine& bar) {
    { broker.on_tick(bar) } -> std::same_as<void>;
};

template <class BrokerT>
concept HasTrades = requires(const BrokerT& broker) {
    { broker.trades() } -> std::convertible_to<std::unordered_map<TradeID, Trade>>;
};

template <class BrokerT>
concept HasOrders = requires(const BrokerT& broker) {
    { broker.orders() } -> std::convertible_to<std::unordered_map<OrderID, Order>>;
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
