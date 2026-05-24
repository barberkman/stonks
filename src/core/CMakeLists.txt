add_library(stonks_core STATIC
    engine.cpp
)

target_include_directories(stonks_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(stonks_core PUBLIC cxx_std_20)
