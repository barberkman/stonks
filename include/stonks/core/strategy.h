#pragma once

#include <concepts>

namespace stonks::core {

template <class StrategyT, class ContextT>
concept HasOnStart = requires(StrategyT strategy, ContextT& context) { strategy.on_start(context); };

template <class StrategyT, class ContextT>
concept HasOnStop = requires(StrategyT strategy, ContextT& context) { strategy.on_stop(context); };

template <class StrategyT, class ContextT>
concept HasOnTick = requires(StrategyT strategy, ContextT& context) { strategy.on_tick(context); };

template <class StrategyT, class ContextT>
concept Strategy = std::movable<StrategyT> && HasOnTick<StrategyT, ContextT>;

} // namespace stonks::core
