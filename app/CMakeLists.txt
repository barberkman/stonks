add_executable(app main.cpp)
target_link_libraries(app PRIVATE
    stonks_core
    stonks_broker
    stonks_datafeed
)
