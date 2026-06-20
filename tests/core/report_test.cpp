// Unit tests for the external reporter. compute_metrics/print_report are pure
// functions over plain data, so they're exercised here with hand-built inputs —
// no engine. Note: Order's constructor is private to Context, so orders_placed
// > 0 is covered by the end-to-end scenario test, not here.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "stonks/core/types.h"

#include "src/report.h"

namespace stonks::app {
namespace {

using core::Balance;
using core::EquityPoint;
using core::OrderSide;
using core::Timestamp;
using core::Trade;

EquityPoint eq(std::int64_t ms, Balance equity)
{
    return EquityPoint{ Timestamp::from_millis(ms), equity };
}

Trade trade(std::int64_t ms, OrderSide side, core::Quantity qty, core::Price price)
{
    return Trade{
        core::TradeID{ 1 }, core::OrderID{ 1 }, Timestamp::from_millis(ms),
        core::Symbol{ "X" }, side, qty, price,
    };
}

TEST(Report, MaxDrawdownFromEquityCurve)
{
    std::vector<EquityPoint> curve{ eq(1, 100.0), eq(2, 120.0), eq(3, 90.0), eq(4, 110.0) };
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 100.0,
        .bars_processed = 4,
        .trades = {},
        .orders = {},
        .equity_curve = curve,
        .ending_cash = 110.0,
        .ending_equity = 110.0,
        .elapsed = std::chrono::nanoseconds{ 0 },
    });
    EXPECT_DOUBLE_EQ(m.max_drawdown_pct, 25.0);   // peak 120 -> trough 90
}

TEST(Report, ReturnPctComputedAndGuardedAtZeroStart)
{
    std::vector<EquityPoint> curve{ eq(1, 1'000.0), eq(2, 1'100.0) };
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 1'000.0,
        .bars_processed = 2,
        .trades = {},
        .orders = {},
        .equity_curve = curve,
        .ending_cash = 1'100.0,
        .ending_equity = 1'100.0,
        .elapsed = {},
    });
    ASSERT_TRUE(m.return_pct.has_value());
    EXPECT_DOUBLE_EQ(*m.return_pct, 10.0);

    const auto z = compute_metrics(ReportInput{
        .starting_cash = 0.0,
        .bars_processed = 0,
        .trades = {},
        .orders = {},
        .equity_curve = {},
        .ending_cash = 0.0,
        .ending_equity = 0.0,
        .elapsed = {},
    });
    EXPECT_FALSE(z.return_pct.has_value());
}

TEST(Report, NotionalSumsTradeValue)
{
    std::vector<Trade> trades{
        trade(1, OrderSide::Buy, 2.0, 100.0),
        trade(2, OrderSide::Sell, 2.0, 110.0),
    };
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 1'000.0,
        .bars_processed = 1,
        .trades = trades,
        .orders = {},
        .equity_curve = {},
        .ending_cash = 1'020.0,
        .ending_equity = 1'020.0,
        .elapsed = {},
    });
    EXPECT_EQ(m.trade_count, 2u);
    EXPECT_DOUBLE_EQ(m.notional, 2.0 * 100.0 + 2.0 * 110.0);
}

TEST(Report, WinRateCountsProfitableRoundTrips)
{
    // Two round trips on the same symbol: buy@100/sell@110 (win), then
    // buy@110/sell@105 (loss). One win out of two closed trades -> 50%.
    std::vector<Trade> trades{
        trade(1, OrderSide::Buy, 1.0, 100.0),
        trade(2, OrderSide::Sell, 1.0, 110.0),
        trade(3, OrderSide::Buy, 1.0, 110.0),
        trade(4, OrderSide::Sell, 1.0, 105.0),
    };
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 1'000.0,
        .bars_processed = 4,
        .trades = trades,
        .orders = {},
        .equity_curve = {},
        .ending_cash = 1'005.0,
        .ending_equity = 1'005.0,
        .elapsed = {},
    });
    EXPECT_EQ(m.closed_trades, 2u);
    EXPECT_EQ(m.winning_trades, 1u);
    ASSERT_TRUE(m.win_rate_pct.has_value());
    EXPECT_DOUBLE_EQ(*m.win_rate_pct, 50.0);
}

TEST(Report, WinRateScaleInOutCollapsesToOneRoundTrip)
{
    // Build a position over two buys (avg 110), then exit it over two sells.
    // The whole open-to-flat cycle is a single closed trade, not four.
    // Realized = (120-110) + (130-110) = +30 -> a win.
    std::vector<Trade> trades{
        trade(1, OrderSide::Buy, 1.0, 100.0),
        trade(2, OrderSide::Buy, 1.0, 120.0),   // avg entry now 110
        trade(3, OrderSide::Sell, 1.0, 120.0),
        trade(4, OrderSide::Sell, 1.0, 130.0),
    };
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 1'000.0,
        .bars_processed = 4,
        .trades = trades,
        .orders = {},
        .equity_curve = {},
        .ending_cash = 1'030.0,
        .ending_equity = 1'030.0,
        .elapsed = {},
    });
    EXPECT_EQ(m.closed_trades, 1u);
    EXPECT_EQ(m.winning_trades, 1u);
    ASSERT_TRUE(m.win_rate_pct.has_value());
    EXPECT_DOUBLE_EQ(*m.win_rate_pct, 100.0);
}

TEST(Report, WinRateShortRoundTripAndFlip)
{
    // Sell 2 short @100, then buy 3 @90: closes the short for +20 (a win) and
    // flips into a fresh long of 1 @90 that never closes -> 1 closed trade.
    std::vector<Trade> trades{
        trade(1, OrderSide::Sell, 2.0, 100.0),
        trade(2, OrderSide::Buy, 3.0, 90.0),
    };
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 1'000.0,
        .bars_processed = 2,
        .trades = trades,
        .orders = {},
        .equity_curve = {},
        .ending_cash = 1'000.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
    });
    EXPECT_EQ(m.closed_trades, 1u);
    EXPECT_EQ(m.winning_trades, 1u);
    ASSERT_TRUE(m.win_rate_pct.has_value());
    EXPECT_DOUBLE_EQ(*m.win_rate_pct, 100.0);
}

TEST(Report, WinRateBreakevenIsNotAWin)
{
    std::vector<Trade> trades{
        trade(1, OrderSide::Buy, 1.0, 100.0),
        trade(2, OrderSide::Sell, 1.0, 100.0),   // realized exactly 0
    };
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 1'000.0,
        .bars_processed = 2,
        .trades = trades,
        .orders = {},
        .equity_curve = {},
        .ending_cash = 1'000.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
    });
    EXPECT_EQ(m.closed_trades, 1u);
    EXPECT_EQ(m.winning_trades, 0u);
    ASSERT_TRUE(m.win_rate_pct.has_value());
    EXPECT_DOUBLE_EQ(*m.win_rate_pct, 0.0);
}

TEST(Report, WinRatePrintsZeroWhenNothingCloses)
{
    // A lone opening fill leaves the position open at run end: no round trip.
    // The metric stays undefined, but the line is always printed (like Return
    // and Max drawdown) so the report never silently drops it.
    std::vector<Trade> trades{ trade(1, OrderSide::Buy, 1.0, 100.0) };
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 1'000.0,
        .bars_processed = 1,
        .trades = trades,
        .orders = {},
        .equity_curve = {},
        .ending_cash = 900.0,
        .ending_equity = 1'000.0,
        .elapsed = {},
    });
    EXPECT_EQ(m.closed_trades, 0u);
    EXPECT_FALSE(m.win_rate_pct.has_value());

    std::ostringstream os;
    print_report(os, m);
    EXPECT_NE(os.str().find("Win rate:        0.00 % (0/0)"), std::string::npos);
}

TEST(Report, TimeRangeFromCurveEndpoints)
{
    std::vector<EquityPoint> curve{ eq(1000, 100.0), eq(5000, 100.0) };
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 100.0,
        .bars_processed = 2,
        .trades = {},
        .orders = {},
        .equity_curve = curve,
        .ending_cash = 100.0,
        .ending_equity = 100.0,
        .elapsed = {},
    });
    ASSERT_TRUE(m.first_ts.has_value());
    EXPECT_EQ(*m.first_ts, Timestamp::from_millis(1000));
    EXPECT_EQ(*m.last_ts, Timestamp::from_millis(5000));
}

TEST(Report, EmptyRunDoesNotCrash)
{
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 100.0,
        .bars_processed = 0,
        .trades = {},
        .orders = {},
        .equity_curve = {},
        .ending_cash = 100.0,
        .ending_equity = 100.0,
        .elapsed = {},
    });
    EXPECT_EQ(m.trade_count, 0u);
    EXPECT_EQ(m.orders_placed, 0u);
    EXPECT_DOUBLE_EQ(m.max_drawdown_pct, 0.0);
    EXPECT_FALSE(m.first_ts.has_value());
}

TEST(Report, PrintIncludesExpectedLines)
{
    std::vector<Trade> trades{ trade(1, OrderSide::Buy, 1.0, 100.0) };
    std::vector<EquityPoint> curve{ eq(1000, 100.0) };
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 100.0,
        .bars_processed = 1,
        .trades = trades,
        .orders = {},
        .equity_curve = curve,
        .ending_cash = 0.0,
        .ending_equity = 100.0,
        .elapsed = std::chrono::milliseconds{ 2 },
    });
    std::ostringstream os;
    print_report(os, m);
    const std::string s = os.str();
    EXPECT_NE(s.find("=== Backtest report ==="), std::string::npos);
    EXPECT_NE(s.find("Time range:"), std::string::npos);
    EXPECT_NE(s.find("Orders placed:   0"), std::string::npos);
    EXPECT_NE(s.find("Notional traded:"), std::string::npos);
    EXPECT_NE(s.find("Elapsed:"), std::string::npos);
    EXPECT_NE(s.find("per bar:"), std::string::npos);
}

TEST(Report, PrintShowsNotionalZeroButOmitsTimeRangeAndPerBarWhenNoData)
{
    const auto m = compute_metrics(ReportInput{
        .starting_cash = 100.0,
        .bars_processed = 0,
        .trades = {},
        .orders = {},
        .equity_curve = {},
        .ending_cash = 100.0,
        .ending_equity = 100.0,
        .elapsed = {},
    });
    std::ostringstream os;
    print_report(os, m);
    const std::string s = os.str();
    EXPECT_NE(s.find("Notional traded: 0.00"), std::string::npos);   // always printed now
    // Time range and per bar stay guarded: they would dereference an empty
    // timestamp / divide by zero bars, not just print an "empty" value.
    EXPECT_EQ(s.find("Time range:"), std::string::npos);
    EXPECT_EQ(s.find("per bar:"), std::string::npos);   // bars_processed == 0
}

} // namespace
} // namespace stonks::app
