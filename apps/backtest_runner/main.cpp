#include <iostream>
#include <optional>
#include <vector>

#include "stonks/core/context.h"
#include "stonks/core/engine.h"
#include "stonks/core/types.h"

struct PlaceholderStrategy
{
    void on_start(auto& context)
    {
        std::cout << "on_start\n";
        context.cash();
        context.equity();
    }

    void on_kline(auto& context)
    {
        std::cout << "on_kline\n";
        context.now();
        context.kline(100);
    }

    void on_stop(auto& context)
    {
        std::cout << "on_stop\n";
        context.cash();
        context.equity();
    }
};

class KLineFeed
{
public:
    KLineFeed()
    : m_klines{
        {1'700'000'000, "TEST", 100.0, 101.0,  99.5, 100.5, 1'000.0},
        {1'700'000'060, "TEST", 100.5, 102.0, 100.0, 101.5, 1'200.0},
        {1'700'000'120, "TEST", 101.5, 101.8, 100.8, 101.0,   900.0},
      }
    {}

    std::optional<stonks::core::Timestamp> peek(stonks::core::Timestamp current) const
    {
        for (const auto& kl : m_klines) {
            if (kl.timestamp > current) { return kl.timestamp; }
        }
        return std::nullopt;
    }

    std::vector<stonks::core::KLine> kline(int /*count*/) const
    {
        return m_klines;
    }

private:
    std::vector<stonks::core::KLine> m_klines;
};

class Broker
{
public:
    stonks::core::Balance cash() const
    {
        return 0.0;
    }

    stonks::core::Balance equity() const
    {
        return 0.0;
    }
};

int main() {
    std::cout << "stonks backtest_runner v0.0.1\n";

    stonks::core::Engine engine
    { 
        PlaceholderStrategy{}, 
        KLineFeed{},
        Broker{}
    };
    engine.run();
    
    return 0;
}
