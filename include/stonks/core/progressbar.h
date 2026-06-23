#pragma once

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace stonks::core {

inline constexpr int kBarWidth = 22;

namespace detail {

inline bool stderr_is_tty()
{
#ifdef _WIN32
    return _isatty(_fileno(stderr)) != 0;
#else
    return ::isatty(STDERR_FILENO) != 0;
#endif
}

// MM:SS, or H:MM:SS once past an hour.
inline std::string format_duration(std::chrono::nanoseconds elapsed)
{
    auto total_s = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    if (total_s < 0) { total_s = 0; }
    const long long h = total_s / 3600;
    const long long m = (total_s % 3600) / 60;
    const long long s = total_s % 60;
    char buf[32];
    if (h > 0) {
        std::snprintf(buf, sizeof(buf), "%lld:%02lld:%02lld", h, m, s);
    } else {
        std::snprintf(buf, sizeof(buf), "%02lld:%02lld", m, s);
    }
    return buf;
}

// A `width`-column bar for fraction `frac` in [0, 1], using Unicode eighth
// blocks for sub-cell resolution. Padded with spaces to a fixed column count.
inline std::string render_bar(double frac, int width)
{
    if (frac < 0.0) { frac = 0.0; }
    if (frac > 1.0) { frac = 1.0; }

    static const char* const eighths[] = {
        "", "▏", "▎", "▍", "▌", "▋", "▊", "▉"
    };
    static const char* const full_block = "█";

    const double filled = frac * width;
    int full = static_cast<int>(filled);
    if (full > width) { full = width; }
    const int part = static_cast<int>((filled - full) * 8.0); // 0..7

    std::string bar;
    int cols = 0;
    for (int i = 0; i < full; ++i) { bar += full_block; ++cols; }
    if (cols < width && part > 0) { bar += eighths[part]; ++cols; }
    while (cols < width) { bar += ' '; ++cols; }
    return bar;
}

} // namespace detail

// Mode for ProgressBar output: print a live line to the console (default), or
// stay silent and only track values for an external consumer (e.g. a GUI).
enum class ProgressOutput { Console, Silent };

// A snapshot of progress for an external consumer to render itself. `percent`
// is 0..100, or -1 when `total` is unknown; `eta` is 0 when unknown or complete.
struct ProgressState
{
    std::optional<std::size_t> total;
    std::size_t current{ 0 };
    int percent{ -1 };
    std::chrono::nanoseconds elapsed{ 0 };
    double rate{ 0.0 };
    std::chrono::nanoseconds eta{ 0 };
};

// Single source of truth for the derived progress values, shared by the console
// formatter and ProgressBar::state() so they can never drift apart.
inline ProgressState compute_progress(std::optional<std::size_t> total,
                                      std::size_t current,
                                      std::chrono::nanoseconds elapsed)
{
    const double elapsed_s = std::chrono::duration<double>{ elapsed }.count();
    ProgressState state;
    state.total = total;
    state.current = current;
    state.elapsed = elapsed;
    state.rate = elapsed_s > 0.0 ? static_cast<double>(current) / elapsed_s : 0.0;
    if (total) {
        const std::size_t tot = *total;
        state.percent = tot > 0 ? static_cast<int>(current * 100 / tot) : 100;
        if (current > 0 && current < tot && elapsed_s > 0.0) {
            const double remaining_s = elapsed_s
                * static_cast<double>(tot - current) / static_cast<double>(current);
            state.eta = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>{ remaining_s });
        }
    }
    return state;
}

// Pure + deterministic: no clock, no stream, no TTY. `elapsed` is supplied by
// the caller so this is directly unit-testable. total == nullopt selects the
// count-only fallback (for feeds that cannot report a size).
inline std::string format_progress(std::optional<std::size_t> total,
                                   std::size_t current,
                                   std::chrono::nanoseconds elapsed,
                                   std::string_view unit = "bars")
{
    const ProgressState state = compute_progress(total, current, elapsed);
    const int unit_len = static_cast<int>(unit.size());

    char buf[512];
    if (total) {
        const std::size_t tot = *total;
        const double frac = tot > 0
            ? static_cast<double>(current) / static_cast<double>(tot)
            : 1.0;

        std::snprintf(buf, sizeof(buf),
            "%3d%%|%s| %zu/%zu [%s<%s, %.0f %.*s/s]",
            state.percent,
            detail::render_bar(frac, kBarWidth).c_str(),
            current, tot,
            detail::format_duration(elapsed).c_str(),
            detail::format_duration(state.eta).c_str(),
            state.rate,
            unit_len, unit.data());
        return buf;
    }

    // Count-only fallback: spinner + count + elapsed + rate, no percentage.
    static const char* const frames[] = {
        "⠋", "⠙", "⠹", "⠸", "⠼",
        "⠴", "⠦", "⠧", "⠇", "⠏"
    };
    const char* const spinner = frames[current % 10];
    std::snprintf(buf, sizeof(buf),
        "%s %zu %.*s [%s, %.0f %.*s/s]",
        spinner,
        current,
        unit_len, unit.data(),
        detail::format_duration(elapsed).c_str(),
        state.rate,
        unit_len, unit.data());
    return buf;
}

// tqdm-style progress bar. Renders to std::cerr, and only when std::cerr is a
// terminal, so piped output / log files / ctest stay clean. Self-throttles so
// update() is cheap to call every iteration.
class ProgressBar
{
public:
    explicit ProgressBar(std::optional<std::size_t> total,
                         std::string_view unit = "bars",
                         ProgressOutput mode = ProgressOutput::Console)
    : m_total{ total },
      m_unit{ unit },
      m_console{ mode == ProgressOutput::Console && detail::stderr_is_tty() },
      m_start{ std::chrono::steady_clock::now() }
    {}

    void update(std::size_t current)
    {
        m_current = current;
        if (!m_console) { return; }
        if (m_total) {
            const std::size_t tot = *m_total;
            const int pct = tot > 0 ? static_cast<int>(current * 100 / tot) : 100;
            if (pct == m_last_percent) { return; }
            m_last_percent = pct;
        } else {
            if (current != 0 && current - m_last_drawn < kUnknownStep) { return; }
            m_last_drawn = current;
        }
        render(current);
    }

    void finish()
    {
        if (!m_console) { return; }
        render(m_current);
        std::cerr << '\n';
        std::cerr.flush();
    }

    // Snapshot of the latest progress for an external consumer (e.g. a GUI) to
    // render itself. Valid in any mode; tracks values even when not printing.
    ProgressState state() const
    {
        return compute_progress(m_total, m_current,
            std::chrono::steady_clock::now() - m_start);
    }

private:
    static constexpr std::size_t kUnknownStep = 4096;

    void render(std::size_t current)
    {
        const auto elapsed = std::chrono::steady_clock::now() - m_start;
        // '\r' returns to column 0; "\033[K" erases any leftover from a longer
        // previous line (e.g. when the ETA shrinks).
        std::cerr << '\r' << format_progress(m_total, current, elapsed, m_unit)
                  << "\033[K";
        std::cerr.flush();
    }

    std::optional<std::size_t> m_total;
    std::string m_unit;
    bool m_console;
    std::chrono::steady_clock::time_point m_start;
    std::size_t m_current{ 0 };
    int m_last_percent{ -1 };
    std::size_t m_last_drawn{ 0 };
};

} // namespace stonks::core
