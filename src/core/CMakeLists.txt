add_library(stonks_core INTERFACE)

target_include_directories(stonks_core INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(stonks_core INTERFACE cxx_std_20)
