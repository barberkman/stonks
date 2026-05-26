#pragma once

#include <concepts>
#include <optional>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::core {

template <class DataFeedT>
concept HasPeek = requires(const DataFeedT& dataFeed, Timestamp now) {
    { dataFeed.peek(now) } -> std::convertible_to<std::optional<Timestamp>>;
};

template <class DataFeedT>
concept HasKLine = requires(const DataFeedT& dataFeed, int count) {
    { dataFeed.kline(count) } -> std::same_as<std::vector<KLine>>;
};

template <class DataFeedT>
concept DataFeed = std::movable<DataFeedT> && HasPeek<DataFeedT> && HasKLine<DataFeedT>;

} // namespace stonks::core
