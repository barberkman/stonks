#pragma once

#include <concepts>

namespace stonks::core {

class Context;

template <class StrategyT>
concept HasOnStart = requires(StrategyT strategy, Context& context) { strategy.on_start(context); };

template <class StrategyT>
concept HasOnStop = requires(StrategyT strategy, Context& context) { strategy.on_stop(context); };

template <class StrategyT>
concept HasOnKLine = requires(StrategyT strategy, Context& context) { strategy.on_kline(context); };

template <class StrategyT>
concept Strategy = std::movable<StrategyT> && HasOnKLine<StrategyT>;

} // namespace stonks::core
