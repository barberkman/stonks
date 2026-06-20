#pragma once

// Temporary diagnostic logging facility — compile-time gated, no performance
// constraint. Configure with `-DSTONKS_LOG=ON` to define STONKS_LOG_ENABLED and
// compile the logging in; without it, every STONKS_LOG(...) expands to a no-op
// and no logging code enters the binary. When enabled, logging is always on
// (no runtime switch) and output goes to std::cerr (the report owns std::cout,
// and the ProgressBar self-suppresses on non-TTY stderr, so `2>run.log` gives a
// clean capture). Remove all STONKS_LOG(...) call sites and this header when no
// longer needed.

#ifdef STONKS_LOG_ENABLED

#include <format>
#include <iostream>
#include <string_view>

namespace stonks::log {

template <class... Args>
void line(std::string_view tag, std::string_view fmt, const Args&... args)
{
    std::cerr << '[' << tag << "] "
              << std::vformat(fmt, std::make_format_args(args...)) << '\n';
}

} // namespace stonks::log

#define STONKS_LOG(tag, ...) ::stonks::log::line(tag, __VA_ARGS__)

#else

#define STONKS_LOG(tag, ...) ((void)0)

#endif
