#pragma once

#include <iostream>

struct PlaceholderStrategy
{
    void on_start(auto& context)
    {
        context.cash();
        context.equity();
    }

    void on_tick(auto& context)
    {
        auto now = context.now();
        auto klines = context.klines(100);
        std::cout << "PlaceholderStrategy::tick: " << now << "\t" << klines.front().symbol << std::endl;
    }

    void on_stop(auto& context)
    {
        context.cash();
        context.equity();
    }
};
