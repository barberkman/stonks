#pragma once

#include <concepts>
#include <cstddef>
#include <optional>
#include <vector>

#include "stonks/core/types.h"

namespace stonks::core {

template <class DataFeedT>
concept HasNextTimestamp = requires(const DataFeedT& dataFeed) {
    { dataFeed.next_timestamp() } -> std::convertible_to<std::optional<Timestamp>>;
};

template <class DataFeedT>
concept HasAdvance = requires(DataFeedT& dataFeed) {
    { dataFeed.advance() } -> std::same_as<void>;
};

template <class DataFeedT>
concept HasResolution = requires(const DataFeedT& dataFeed) {
    { dataFeed.resolution() } -> std::convertible_to<Timestamp::duration>;
};

// The bars at the current timestamp (one per symbol that printed), fed to the
// broker before the strategy runs.
template <class DataFeedT>
concept HasCurrentBars = requires(const DataFeedT& dataFeed) {
    { dataFeed.current_bars() } -> std::convertible_to<std::vector<KLine>>;
};

// The multi-symbol lookback window handed to the strategy: each printing symbol's
// last `count` bars.
template <class DataFeedT>
concept HasWindow = requires(const DataFeedT& dataFeed, int count) {
    { dataFeed.window(count) } -> std::convertible_to<MarketWindow>;
};

// Optional capability: a feed that knows its total bar count up front. Used by
// the engine to render a percentage/ETA progress bar; not required by DataFeed.
template <class DataFeedT>
concept HasSize = requires(const DataFeedT& dataFeed) {
    { dataFeed.size() } -> std::convertible_to<std::size_t>;
};

template <class DataFeedT>
concept DataFeed = std::movable<DataFeedT>
    && HasNextTimestamp<DataFeedT>
    && HasAdvance<DataFeedT>
    && HasResolution<DataFeedT>
    && HasCurrentBars<DataFeedT>
    && HasWindow<DataFeedT>;

} // namespace stonks::core
