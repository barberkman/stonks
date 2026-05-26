#pragma once

#include <iostream>

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
        auto now = context.now();
        std::cout << "Strategy now: " << now << std::endl;
        context.klines(100);
    }

    void on_stop(auto& context)
    {
        std::cout << "on_stop\n";
        context.cash();
        context.equity();
    }
};
