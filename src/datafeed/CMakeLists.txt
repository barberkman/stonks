# QUIET propagates through Arrow's find_dependency() calls to the inner
# Find<lz4|re2|Thrift>Alt.cmake modules, silencing their config-mode warnings
# on Debian-family systems (where those libs ship only pkg-config).
find_package(Arrow REQUIRED QUIET)
find_package(Parquet REQUIRED QUIET)

add_library(stonks_datafeed STATIC klinefeed.cpp parquetmeta.cpp)
target_link_libraries(stonks_datafeed
    PUBLIC stonks_core
    PRIVATE Arrow::arrow_shared Parquet::parquet_shared
)
