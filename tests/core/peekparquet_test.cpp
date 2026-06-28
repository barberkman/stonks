#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include "stonks/datafeed/parquetmeta.h"

namespace {

void ok(const arrow::Status& status)
{
    if (!status.ok()) { throw std::runtime_error{ status.ToString() }; }
}

// Writes a small OHLCV parquet (rows deliberately out of time order, multiple
// symbols, extra OHLCV columns) so the peek exercises real column projection.
std::filesystem::path write_fixture()
{
    using namespace arrow;

    const auto ts_type = timestamp(TimeUnit::MILLI, "UTC");
    TimestampBuilder ts_b{ ts_type, default_memory_pool() };
    StringBuilder sym_b;
    DoubleBuilder open_b, close_b;

    struct Row { std::int64_t ts; const char* sym; double open; double close; };
    const std::vector<Row> rows = {
        { 1000, "BTCUSDT", 10.0, 11.0 },
        { 3000, "ETHUSDT", 20.0, 19.0 },
        { 2000, "BTCUSDT", 11.0, 12.0 },
        { 5000, "ETHUSDT", 19.0, 21.0 },
        { 4000, "SOLUSDT", 5.0, 6.0 },
    };
    for (const auto& r : rows) {
        ok(ts_b.Append(r.ts));
        ok(sym_b.Append(r.sym));
        ok(open_b.Append(r.open));
        ok(close_b.Append(r.close));
    }
    std::shared_ptr<Array> ts_a, sym_a, open_a, close_a;
    ok(ts_b.Finish(&ts_a));
    ok(sym_b.Finish(&sym_a));
    ok(open_b.Finish(&open_a));
    ok(close_b.Finish(&close_a));

    const auto sch = schema({
        field("timestamp", ts_type),
        field("symbol", utf8()),
        field("open", float64()),
        field("close", float64()),
    });
    const auto table = Table::Make(sch, { ts_a, sym_a, open_a, close_a });

    const auto path = std::filesystem::temp_directory_path() / "stonks_peek_fixture.parquet";
    auto outfile = io::FileOutputStream::Open(path.string()).ValueOrDie();
    ok(parquet::arrow::WriteTable(*table, default_memory_pool(), outfile, 1024));
    ok(outfile->Close());
    return path;
}

} // namespace

TEST(PeekParquet, ReportsDistinctSymbolsAndTimestampSpan)
{
    const auto path = write_fixture();
    const auto meta = stonks::datafeed::peek_parquet(path);

    EXPECT_EQ(meta.rows, 5);
    ASSERT_EQ(meta.symbols.size(), 3u);
    EXPECT_EQ(meta.symbols[0], "BTCUSDT");
    EXPECT_EQ(meta.symbols[1], "ETHUSDT");
    EXPECT_EQ(meta.symbols[2], "SOLUSDT");
    ASSERT_TRUE(meta.min_ts_ms.has_value());
    ASSERT_TRUE(meta.max_ts_ms.has_value());
    EXPECT_EQ(*meta.min_ts_ms, 1000);
    EXPECT_EQ(*meta.max_ts_ms, 5000);

    std::filesystem::remove(path);
}

TEST(PeekParquet, MissingFileThrows)
{
    EXPECT_THROW(stonks::datafeed::peek_parquet("/nonexistent/path/to.parquet"),
                 std::runtime_error);
}
