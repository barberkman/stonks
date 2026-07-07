#include "stonks/binance/liveklinefeed.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

#include "stonks/core/datafeed.h"

namespace stonks::binance {

using namespace stonks::core;

namespace {

constexpr int kMaxBuffer = 1500;          // cap the rolling history per symbol
constexpr long long kCloseGraceMs = 1500; // small wait past close so Binance has the final candle
constexpr long long kPollMs = 1000;       // poll cadence while waiting for a candle to close

std::function<long long()> default_clock()
{
    return [] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    };
}

// Parse a Binance interval string ("1m","5m","1h","1d","1w") into milliseconds.
std::int64_t interval_to_ms(const std::string& iv)
{
    if (iv.size() < 2) { throw std::runtime_error{ "bad kline interval: " + iv }; }
    const char unit = iv.back();
    const long long n = std::stoll(iv.substr(0, iv.size() - 1));
    switch (unit) {
    case 'm': return n * 60'000LL;
    case 'h': return n * 3'600'000LL;
    case 'd': return n * 86'400'000LL;
    case 'w': return n * 604'800'000LL;
    default: throw std::runtime_error{ "unsupported kline interval unit: " + iv };
    }
}

// One /fapi/v1/klines row: [openTime, open, high, low, close, volume, closeTime, ...].
KLine parse_kline(const Symbol& symbol, const nlohmann::json& row)
{
    return KLine{
        Timestamp::from_millis(row[0].get<std::int64_t>()),
        symbol,
        std::stod(row[1].get<std::string>()),
        std::stod(row[2].get<std::string>()),
        std::stod(row[3].get<std::string>()),
        std::stod(row[4].get<std::string>()),
        std::stod(row[5].get<std::string>()),
    };
}

} // namespace

LiveKlineFeed::LiveKlineFeed(BinanceConfig config, std::vector<Symbol> symbols, std::string interval,
                             int seed_bars, const std::atomic<bool>* cancel,
                             Transport transport, std::function<long long()> now_ms)
: m_client{ std::move(config), std::move(transport), now_ms },
  m_symbols{ std::move(symbols) },
  m_interval{ std::move(interval) },
  m_interval_ms{ interval_to_ms(m_interval) },
  m_seed_bars{ seed_bars },
  m_cancel{ cancel },
  m_now_ms{ now_ms ? std::move(now_ms) : default_clock() }
{
    seed();
}

void LiveKlineFeed::seed()
{
    // Fetch recent closed candles per symbol; the last row may still be forming,
    // so keep only rows whose close time has passed. Set the emit cursor to the
    // candle immediately after the latest seeded one — we act only on candles
    // that close after startup.
    std::int64_t latest_open = 0;
    for (const auto& sym : m_symbols) {
        const nlohmann::json rows = m_client.public_request(
            "GET", "/fapi/v1/klines",
            { { "symbol", sym }, { "interval", m_interval }, { "limit", std::to_string(m_seed_bars) } });

        SymBuf& b = m_buf[sym];
        for (const auto& row : rows) {
            const std::int64_t open_ms = row[0].get<std::int64_t>();
            const std::int64_t close_ms = open_ms + m_interval_ms;
            if (close_ms > now_ms()) { continue; }   // still forming — skip
            const KLine k = parse_kline(sym, row);
            b.ts.push_back(open_ms);
            b.open.push_back(k.open);
            b.high.push_back(k.high);
            b.low.push_back(k.low);
            b.close.push_back(k.close);
            b.volume.push_back(k.volume);
            latest_open = std::max(latest_open, open_ms);
        }
    }
    // If seeding found nothing (fresh symbol / clock), align the cursor to the
    // current interval boundary.
    if (latest_open == 0) {
        latest_open = (now_ms() / m_interval_ms) * m_interval_ms - m_interval_ms;
    }
    m_cursor_open_ms = latest_open + m_interval_ms;
}

void LiveKlineFeed::append(const Symbol& symbol, const KLine& k) const
{
    SymBuf& b = m_buf[symbol];
    b.ts.push_back(k.timestamp.value.time_since_epoch().count());
    b.open.push_back(k.open);
    b.high.push_back(k.high);
    b.low.push_back(k.low);
    b.close.push_back(k.close);
    b.volume.push_back(k.volume);
    if (static_cast<int>(b.ts.size()) > kMaxBuffer) {
        const std::size_t drop = b.ts.size() - kMaxBuffer;
        for (auto* col : { &b.open, &b.high, &b.low, &b.close, &b.volume }) {
            col->erase(col->begin(), col->begin() + drop);
        }
        b.ts.erase(b.ts.begin(), b.ts.begin() + drop);
    }
}

std::optional<Timestamp> LiveKlineFeed::next_timestamp() const
{
    if (m_current_ready) { return Timestamp::from_millis(m_cursor_open_ms); }

    const std::int64_t open_ms = m_cursor_open_ms;
    const std::int64_t ready_at = open_ms + m_interval_ms + kCloseGraceMs;

    // Block until the cursor's candle has closed (interruptible via cancel).
    while (now_ms() < ready_at) {
        if (m_cancel && m_cancel->load(std::memory_order_relaxed)) { return std::nullopt; }
        const long long remaining = ready_at - now_ms();
        std::this_thread::sleep_for(std::chrono::milliseconds{ std::min(kPollMs, remaining) });
    }
    if (m_cancel && m_cancel->load(std::memory_order_relaxed)) { return std::nullopt; }

    // Fetch the closed candle at open_ms for each symbol and roll it in.
    m_current_bars.clear();
    for (const auto& sym : m_symbols) {
        const nlohmann::json rows = m_client.public_request(
            "GET", "/fapi/v1/klines",
            { { "symbol", sym }, { "interval", m_interval },
              { "startTime", std::to_string(open_ms) }, { "limit", "1" } });
        if (rows.empty()) { continue; }
        const KLine k = parse_kline(sym, rows[0]);
        append(sym, k);
        m_current_bars.push_back(k);
    }
    m_current_ready = true;
    return Timestamp::from_millis(open_ms);
}

void LiveKlineFeed::advance()
{
    m_cursor_open_ms += m_interval_ms;
    m_current_ready = false;
    m_current_bars.clear();
}

std::vector<KLine> LiveKlineFeed::current_bars() const { return m_current_bars; }

MarketWindow LiveKlineFeed::window(int count) const
{
    MarketWindow w;
    for (const auto& sym : m_symbols) {
        const auto it = m_buf.find(sym);
        if (it == m_buf.end() || it->second.ts.empty()) { continue; }
        const SymBuf& b = it->second;
        const std::size_t n = b.ts.size();
        const std::size_t k = count > 0 ? std::min(static_cast<std::size_t>(count), n) : 0;
        const std::size_t start = n - k;
        SeriesView sv{
            std::span<const std::int64_t>{ b.ts }.subspan(start, k),
            std::span<const double>{ b.open }.subspan(start, k),
            std::span<const double>{ b.high }.subspan(start, k),
            std::span<const double>{ b.low }.subspan(start, k),
            std::span<const double>{ b.close }.subspan(start, k),
            std::span<const double>{ b.volume }.subspan(start, k),
        };
        w.series.push_back(SymbolSeries{ std::string_view{ sym }, sv });
    }
    return w;
}

Timestamp::duration LiveKlineFeed::resolution() const
{
    return Timestamp::duration{ m_interval_ms };
}

} // namespace stonks::binance

// Compile-time guarantee that LiveKlineFeed drives the Engine like KLineFeed.
static_assert(stonks::core::DataFeed<stonks::binance::LiveKlineFeed>,
              "LiveKlineFeed must satisfy the core::DataFeed concept");
