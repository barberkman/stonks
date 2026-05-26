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
concept HasKLines = requires(const DataFeedT& dataFeed, int count) {
    { dataFeed.klines(count) } -> std::same_as<std::vector<KLine>>;
};

template <class DataFeedT>
concept DataFeed = std::movable<DataFeedT> && HasPeek<DataFeedT> && HasKLines<DataFeedT>;

} // namespace stonks::core
