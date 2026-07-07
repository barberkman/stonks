#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "stonks/binance/restclient.h"
#include "stonks/core/types.h"

namespace stonks::binance {

// Live market-data feed. Satisfies core::DataFeed, so it drives the same Engine
// as the historical KLineFeed. It polls GET /fapi/v1/klines and emits one tick
// per closed candle, going forward from construction; the strategy's history()
// window is served from a bounded rolling buffer seeded at startup.
//
// No-lookahead by construction: a candle for open time T is only emitted after
// it has closed (now >= T + interval), so a strategy never sees a forming bar.
class LiveKlineFeed
{
public:
    // `interval` is a Binance kline interval ("1m", "5m", "1h", "1d", ...).
    // `seed_bars` closed candles per symbol are fetched up front so history()
    // works from the first live tick. `cancel`, if set, lets a blocking wait for
    // the next candle abort cleanly (returns end-of-feed). `transport`/`now_ms`
    // are injectable for tests.
    LiveKlineFeed(BinanceConfig config, std::vector<core::Symbol> symbols, std::string interval,
                  int seed_bars = 200, const std::atomic<bool>* cancel = nullptr,
                  Transport transport = curl_transport(), std::function<long long()> now_ms = {});

    LiveKlineFeed(LiveKlineFeed&&) = default;
    LiveKlineFeed& operator=(LiveKlineFeed&&) = default;
    LiveKlineFeed(const LiveKlineFeed&) = delete;
    LiveKlineFeed& operator=(const LiveKlineFeed&) = delete;

    // --- core::DataFeed surface ----------------------------------------------
    // Blocks (polling) until the next candle closes, then returns its open time.
    // Returns nullopt only when `cancel` fires — ending the engine loop.
    std::optional<core::Timestamp> next_timestamp() const;
    void advance();
    std::vector<core::KLine> current_bars() const;
    core::MarketWindow window(int count) const;
    core::Timestamp::duration resolution() const;

private:
    struct SymBuf
    {
        std::vector<std::int64_t> ts;
        std::vector<double> open, high, low, close, volume;
    };

    void seed();
    void append(const core::Symbol& symbol, const core::KLine& k) const;
    long long now_ms() const { return m_now_ms(); }

    mutable BinanceRestClient m_client;
    std::vector<core::Symbol> m_symbols;   // owned; window() views its strings
    std::string m_interval;
    std::int64_t m_interval_ms;
    int m_seed_bars;
    const std::atomic<bool>* m_cancel;
    std::function<long long()> m_now_ms;

    mutable std::unordered_map<core::Symbol, SymBuf> m_buf;   // rolling history
    mutable std::vector<core::KLine> m_current_bars;          // bars at the cursor
    mutable std::int64_t m_cursor_open_ms = 0;                // open time of the next tick
    mutable bool m_current_ready = false;                     // cursor's bars fetched?
};

} // namespace stonks::binance
