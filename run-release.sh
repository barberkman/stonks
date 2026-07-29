#!/usr/bin/env bash
cmake --preset macos-release && cmake --build --preset macos-release && ./build/macos-release/app/app
