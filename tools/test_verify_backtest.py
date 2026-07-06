"""Subprocess smoke tests for tools/verify_backtest.py.

The verifier is a top-to-bottom script (module-scope argparse), so these drive
it as a subprocess against a minimal hand-built report + parquet fixture: a run
with zero trades/orders replays trivially CLEAN, which is enough to pin the
cooldown-source selection and the exit contract.

Run from the project root with the app-local venv (pandas/pyarrow needed):

    app/python/.venv/bin/pytest tools/ -q
"""

import json
import pathlib
import subprocess
import sys

import pandas as pd

TOOL = pathlib.Path(__file__).with_name("verify_backtest.py")


def _fixture(tmp_path, strategy_block=None):
    ts = ["2024-01-01T00:00:00.000Z", "2024-01-02T00:00:00.000Z"]
    report = {
        "metrics": {"starting_cash": 1000.0, "ending_cash": 1000.0},
        "trades": [],
        "orders": [],
        "equity_curve": [{"timestamp": t, "equity": 1000.0} for t in ts],
    }
    if strategy_block is not None:
        report["strategy"] = strategy_block
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    bars = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "symbol": ["BTCUSDT", "BTCUSDT"],
        "open": [100.0, 101.0], "high": [102.0, 103.0],
        "low": [99.0, 100.0], "close": [101.0, 102.0],
        "volume": [1.0, 1.0],
    })
    parquet_path = tmp_path / "bars.parquet"
    bars.to_parquet(parquet_path)
    return report_path, parquet_path


def _run(report_path, parquet_path, *extra):
    return subprocess.run(
        [sys.executable, str(TOOL), str(report_path), str(parquet_path), *extra],
        capture_output=True, text=True)


def test_cooldown_bars_read_from_report_when_present(tmp_path):
    report, parquet = _fixture(tmp_path, strategy_block={
        "module": "qmsignals", "class": "QMSignalsStrategy",
        "params": {"cooldown_bars": 7.0},   # serialized as a float, like the C++ stamp
    })
    result = _run(report, parquet)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cooldown-bars audit setting: 7 (from report)" in result.stdout
    assert "RESULT: CLEAN" in result.stdout


def test_cooldown_bars_falls_back_to_cli_flag_when_absent(tmp_path):
    report, parquet = _fixture(tmp_path)   # pre-feature report: no "strategy" key
    result = _run(report, parquet, "--cooldown-bars", "9")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cooldown-bars audit setting: 9 (CLI default)" in result.stdout
    assert "RESULT: CLEAN" in result.stdout


def test_cooldown_bars_defaults_to_five_when_neither_present(tmp_path):
    report, parquet = _fixture(tmp_path)
    result = _run(report, parquet)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "cooldown-bars audit setting: 5 (CLI default)" in result.stdout
