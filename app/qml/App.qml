pragma Singleton
import QtQuick
import Stonks

// Application state machine + actions. The single data seam: every view reads
// through this singleton, which is now backed by the real C++ `Backtest`
// controller (engine results) instead of mock generators.
QtObject {
    id: app

    // --- view / UI state ---
    property string view: "backtests"     // backtests | setup | running | detail | logs
    property string detailTab: "report"   // report | trades
    property string selectedBacktest: ""
    property string symbol: ""
    property var selectedTrade: null       // null | int (trade index)
    property real progress: Backtest.progress
    property int barsDone: Backtest.barsDone

    // --- setup state ---
    property string strategy: ""           // strategy module id
    property string dataFile: ""           // data key
    property var availableSymbols: []      // all symbols in the data file
    property var symbols: []               // selected symbol allowlist
    property string startDate: ""
    property string endDate: ""
    property string startCash: "100,000"

    // --- results (populated from the controller) ---
    property var results: ({})             // runId -> full result object
    property var completed: []             // result objects, newest first
    property string runError: ""

    // cached controller lookups
    property var _strats: ({})             // module -> { display, module, cls }
    property var _files: ({})              // key -> { key, label, source }
    property int _nextId: 1

    // --- list caches (lazily filled from the controller) ---
    function strategyList() {
        var list = Backtest.listStrategies();
        var m = {};
        for (var i = 0; i < list.length; i++) m[list[i].module] = list[i];
        _strats = m;
        return list;
    }
    function dataFileList() {
        var list = Backtest.listDataFiles();
        var m = {};
        for (var i = 0; i < list.length; i++) m[list[i].key] = list[i];
        _files = m;
        return list;
    }
    function _fileMap() {
        if (Object.keys(_files).length === 0) dataFileList();
        return _files;
    }
    function _stratEntry(mod) {
        if (Object.keys(_strats).length === 0) strategyList();
        return _strats[mod] || {};
    }

    // --- data accessors (signatures unchanged; backed by the real result) ---
    function allBacktests() { return completed; }
    function currentBacktest() {
        if (view === "running") return _runningSummary();
        var r = results[selectedBacktest];
        return r ? r : {};
    }
    function latestRun() { return completed.length ? completed[0] : {}; }
    function dataFilesObj() { return _fileMap(); }
    function strategySource() { return strategy ? ("app/python/" + strategy + ".py") : ""; }

    function logs() { return _buildLogs(latestRun()); }
    function liveLogList() { return _liveLog(progress); }
    function symMeta(id) { return { seed: id, name: id }; }   // seed carries the symbol id
    function candlesFor(id) { return _drill(id).candles || []; }
    function tradesFor(id) { return _drill(id).trades || []; }
    function equityFor(bt) { return (bt && bt.equity) ? bt.equity : []; }
    function drawdownFor(bt) { return (bt && bt.drawdown) ? bt.drawdown : []; }
    function sparkSeries(seed, bias) { return _drill(seed).spark || []; }

    function _drill(symbolId) {
        var r = results[selectedBacktest];
        if (!r || !r.perSymbol) return {};
        return r.perSymbol[symbolId] || {};
    }

    // --- nav / actions ---
    function go(v) { view = v }
    function goBacktests() { view = "backtests" }
    function goSetup() { view = "setup" }

    function openBacktest(id) {
        var bt = results[id];
        var sym = (bt && bt.symbols && bt.symbols[0]) ? bt.symbols[0].id : "";
        selectedBacktest = id;
        detailTab = "report";
        symbol = sym;
        selectedTrade = null;
        view = "detail";
    }

    function showReport() { detailTab = "report"; selectedTrade = null; }
    function showTrades() { detailTab = "trades"; selectedTrade = null; }
    function selectSymbol(id) { symbol = id; selectedTrade = null; }
    function selectTrade(i) { detailTab = "trades"; selectedTrade = i; }
    function clearTrade() { selectedTrade = null; }
    function openSymbolTrades(id) { detailTab = "trades"; symbol = id; selectedTrade = null; }

    // --- setup actions ---
    function setDataFile(key) {
        dataFile = key;
        availableSymbols = Backtest.peekSymbols(key);
        symbols = availableSymbols.slice();      // default: all symbols selected
        var r = Backtest.peekDateRange(key);
        startDate = r.start || "";
        endDate = r.end || "";
    }
    function toggleSymbol(id) {
        var arr = symbols.slice();
        var i = arr.indexOf(id);
        if (i >= 0) arr.splice(i, 1); else arr.push(id);
        symbols = arr;
    }

    // --- run lifecycle (driven by the controller) ---
    function runBacktest() {
        runError = "";
        var s = _stratEntry(strategy);
        view = "running";
        Backtest.run({
            strategyModule: s.module || strategy,
            strategyClass: s.cls || "",
            strategyDisplay: s.display || strategy,
            dataFile: (_fileMap()[dataFile] || {}).source || ("app/data/" + dataFile + ".parquet"),
            dataKey: dataFile,
            symbols: symbols,
            start: startDate,
            end: endDate,
            startCash: parseFloat(String(startCash).replace(/[^0-9.]/g, '')) || 0
        });
    }
    function runSaved() { runBacktest(); }
    function cancelRun() { Backtest.cancel(); view = "backtests"; }

    function _onFinished(result) {
        var id = String(_nextId++);
        result.id = id;
        var r = {};
        for (var k in results) r[k] = results[k];
        r[id] = result;
        results = r;
        completed = [result].concat(completed);
        selectedBacktest = id;
        symbol = (result.symbols && result.symbols[0]) ? result.symbols[0].id : "";
        detailTab = "report";
        selectedTrade = null;
        view = "detail";
    }

    property Connections _conn: Connections {
        target: Backtest
        function onFinished(result) { app._onFinished(result); }
        function onFailed(message) { app.runError = message; app.view = "setup"; }
    }

    // --- transient summary while a run is in flight (no completed run yet) ---
    function _runningSummary() {
        var s = _stratEntry(strategy);
        return {
            id: "", strategy: (s.display || strategy), dataKey: dataFile,
            status: "running", range: startDate + " → " + endDate,
            symbols: symbols.map(function (x) { return { id: x }; })
        };
    }

    // --- log synthesis (engine has no structured event stream) ---
    function _logCol(lvl) {
        var m = { INFO: '#9aa0a8', DEBUG: '#6b727c', WARN: '#cbb26a', ERROR: '#e1574c' };
        return m[lvl] || '#9aa0a8';
    }
    function _liveLog(p) {
        var seq = [
            [2, 'INFO', 'loading strategy ' + ((_stratEntry(strategy).display) || strategy)],
            [6, 'INFO', 'data ' + dataFile + ' · ' + (symbols || []).join(' · ')],
            [10, 'DEBUG', 'filter ' + (startDate || '∅') + ' → ' + (endDate || '∅')],
            [24, 'INFO', 'streaming bars · ' + barsDone + ' processed'],
            [42, 'INFO', 'simulating fills · cash-secured broker'],
            [60, 'INFO', 'starting cash $' + startCash],
            [82, 'INFO', 'reconciling round-trips'],
            [92, 'DEBUG', 'computing performance metrics'],
            [98, 'INFO', 'run complete · assembling report']
        ];
        var out = [];
        for (var i = 0; i < seq.length; i++)
            if (p >= seq[i][0]) out.push({ t: Math.round(p) + '%', lvl: seq[i][1], lc: _logCol(seq[i][1]), m: seq[i][2] });
        return out;
    }
    function _buildLogs(bt) {
        if (!bt || !bt.id) return [];
        var syms = bt.symbols || [];
        var L = [
            ['00:00.000', 'INFO', 'loaded strategy ' + (bt.strategy || '')],
            ['00:00.012', 'INFO', 'data ' + (bt.dataKey || '') + ' · ' + (bt.bars || '?') + ' bars'],
            ['00:00.044', 'DEBUG', 'universe: ' + syms.map(function (s) { return s.id; }).join(' · ')]
        ];
        for (var i = 0; i < syms.length; i++)
            L.push(['00:0' + (2 + i) + '.000', 'INFO', syms[i].id + ': ' + (syms[i].trades || 0)
                + ' trades · win ' + (syms[i].win || '—')]);
        L.push(['00:08.020', 'INFO', 'profit factor ' + (bt.pf || '—') + ' · max dd ' + (bt.maxdd || '—')]);
        L.push(['00:09.440', 'INFO', 'ending equity ' + (bt.endEqStr || '') + ' · return ' + (bt.ret || '')]);
        L.push(['00:09.840', 'INFO', 'run #' + bt.id + ' completed in ' + (bt.elapsed || '')
            + ' · ' + (bt.perbar || '') + '/bar']);
        return L.map(function (r) { return { t: r[0], lvl: r[1], lc: _logCol(r[1]), m: r[2] }; });
    }

    // --- header strings ---
    function sectionLabel() {
        switch (view) {
        case "backtests": return "Backtests"
        case "setup": return "Configure"
        case "running": return "Running backtest"
        case "detail": return currentBacktest().strategy || ""
        case "logs": return "Diagnostic logs"
        }
        return ""
    }
    function topRight() {
        switch (view) {
        case "backtests": return allBacktests().length + " runs"
        case "setup": return "New run"
        case "running": return "RUNNING · " + Math.round(progress) + "%"
        case "detail": { var b = currentBacktest(); return b.status === "saved" ? "SAVED" : "COMPLETED · " + (b.elapsed || "") }
        case "logs": return "#" + (latestRun().id || "—")
        }
        return ""
    }
}
