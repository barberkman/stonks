#include <iostream>

#include "stonks/core/context.h"
#include "stonks/core/engine.h"

struct PlaceholderStrategy {
    void on_start(stonks::core::Context&) { std::cout << "on_start\n"; }
    void on_kline(stonks::core::Context&) { std::cout << "on_kline\n"; }
    void on_stop(stonks::core::Context&)  { std::cout << "on_stop\n";  }
};

int main() {
    std::cout << "stonks backtest_runner v0.0.1\n";

    stonks::core::Engine engine{ PlaceholderStrategy{} };
    engine.run();
    
    return 0;
}
