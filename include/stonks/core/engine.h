#pragma once

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <utility>

#include "stonks/core/broker.h"
#include "stonks/core/clock.h"
#include "stonks/core/context.h"
#include "stonks/core/datafeed.h"
#include "stonks/core/progressbar.h"
#include "stonks/core/strategy.h"
#include "stonks/core/types.h"

namespace stonks::core {

template <class StrategyT, DataFeed DataFeedT, Broker BrokerT>
class Engine
{
    static_assert(Strategy<StrategyT, Context<BrokerT, DataFeedT>>,
        "StrategyT must satisfy the Strategy concept");

public:
    Engine(StrategyT strategy, DataFeedT dataFeed, BrokerT broker)
    : m_strategy{ std::move(strategy) },
      m_dataFeed{ std::move(dataFeed) },
      m_broker{ std::move(broker) }
    {}

    void run()
    {
        using ContextT = Context<BrokerT, DataFeedT>;
        ContextT context{ m_broker, m_dataFeed, m_clock };

        const auto t0 = std::chrono::steady_clock::now();
        if constexpr (HasOnStart<StrategyT, ContextT>) { m_strategy.on_start(context); }
        std::cout << "Engine::on_start" << std::endl;

        const Balance starting_cash = m_broker.cash();
        std::size_t bar_count = 0;
        std::optional<Timestamp> first_ts;
        Timestamp last_ts{};
        Balance peak_equity = starting_cash;
        double max_drawdown_pct = 0.0;

        std::optional<std::size_t> total;
        if constexpr (HasSize<DataFeedT>) { total = m_dataFeed.size(); }
        ProgressBar progress{ total };

        while (auto ts = m_dataFeed.next_timestamp()) {
            m_clock.set(*ts);
            if (!first_ts) { first_ts = *ts; }
            last_ts = *ts;
            ++bar_count;
            progress.update(bar_count);
            m_broker.on_tick(m_dataFeed.current_kline());

            const Balance current_equity = m_broker.equity();
            if (current_equity > peak_equity) { peak_equity = current_equity; }
            if (peak_equity > 0.0) {
                const double dd = (peak_equity - current_equity) / peak_equity * 100.0;
                if (dd > max_drawdown_pct) { max_drawdown_pct = dd; }
            }

            m_strategy.on_tick(context);
            m_dataFeed.advance();
        }
        progress.finish();

        if constexpr (HasOnStop<StrategyT, ContextT>) { m_strategy.on_stop(context); }
        std::cout << "Engine::on_stop" << std::endl;
        const auto t1 = std::chrono::steady_clock::now();
        const std::chrono::nanoseconds strategy_time = t1 - t0;

        print_report(starting_cash, bar_count, first_ts, last_ts, max_drawdown_pct, strategy_time);
    }

private:
    void print_report(Balance starting_cash,
                      std::size_t bar_count,
                      std::optional<Timestamp> first_ts,
                      Timestamp last_ts,
                      double max_drawdown_pct,
                      std::chrono::nanoseconds strategy_time) const
    {
        const auto& trades = m_broker.trades();
        Balance notional = 0.0;
        for (const auto& t : trades) { notional += t.quantity * t.price; }

        const Balance ending_cash = m_broker.cash();
        const Balance ending_equity = m_broker.equity();

        std::ostream& os = std::cout;
        std::ios old_state{ nullptr };
        old_state.copyfmt(os);
        os << std::fixed << std::setprecision(2);

        os << "=== Backtest report ===\n";
        os << "Bars processed:  " << bar_count << '\n';
        if (first_ts) {
            os << "Time range:      " << *first_ts << " -> " << last_ts << '\n';
        }
        os << "Trades:          " << trades.size() << '\n';
        if (!trades.empty()) {
            os << "Notional traded: " << notional << '\n';
        }
        os << "Starting cash:   " << starting_cash << '\n';
        os << "Ending cash:     " << ending_cash << '\n';
        os << "Ending equity:   " << ending_equity << '\n';
        if (starting_cash != 0.0) {
            const double pct = (ending_equity - starting_cash) / starting_cash * 100.0;
            os << "Return:          " << pct << " %\n";
        }
        os << "Max drawdown:    " << max_drawdown_pct << " %\n";

        const double total_ms = std::chrono::duration<double, std::milli>{ strategy_time }.count();
        os << "Strategy time:   " << total_ms << " ms\n";
        if (bar_count > 0) {
            const double per_bar_us =
                std::chrono::duration<double, std::micro>{ strategy_time }.count()
                / static_cast<double>(bar_count);
            os << "  per bar:       " << per_bar_us << " us\n";
        }

        os.copyfmt(old_state);
    }

    StrategyT m_strategy;
    DataFeedT m_dataFeed;
    BrokerT m_broker;
    Clock m_clock;
};

} // namespace stonks::core
