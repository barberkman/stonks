#pragma once

// Temporary diagnostic logging facility — env-gated, no performance constraint.
// Set STONKS_LOG to any value other than empty/"0" to enable; output goes to
// std::cerr (the report owns std::cout, and the ProgressBar self-suppresses on
// non-TTY stderr, so `2>run.log` gives a clean capture). Remove all
// STONKS_LOG(...) call sites and this header when no longer needed.

#include <cstdlib>
#include <format>
#include <iostream>
#include <string_view>

namespace stonks::log {

inline bool enabled()
{
    static const bool on = [] {
        const char* v = std::getenv("STONKS_LOG");
        return v && *v && std::string_view{ v } != "0";
    }();
    return on;
}

template <class... Args>
void line(std::string_view tag, std::string_view fmt, const Args&... args)
{
    std::cerr << '[' << tag << "] "
              << std::vformat(fmt, std::make_format_args(args...)) << '\n';
}

} // namespace stonks::log

// Guard on enabled() first so arguments are only evaluated when logging is on.
#define STONKS_LOG(tag, ...) \
    do { if (::stonks::log::enabled()) ::stonks::log::line(tag, __VA_ARGS__); } while (0)
