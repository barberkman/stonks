#include <iostream>

#include "stonks/core/engine.h"

int main() {
    stonks::core::Engine engine;
    engine.run();
    std::cout << "stonks backtest_runner v0.0.1\n";
    return 0;
}
