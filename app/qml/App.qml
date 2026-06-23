pragma Singleton
import QtQuick
import "js/mockdata.js" as MD

// Application state machine + actions (mirrors the design's Component).
QtObject {
    id: app

    // --- state ---
    property string view: "backtests"     // backtests | setup | running | detail | logs
    property string detailTab: "report"   // report | trades
    property string selectedBacktest: "4271"
    property string symbol: "NVDA"
    property var selectedTrade: null       // null | int (trade index)
    property real progress: 0
    property int barsDone: 0
    property string startCash: "100,000"
    property string dataFile: "us_megacaps_1h"
    property string strategy: "Momentum Breakout v3"
    property var params: ({ lookback: 20, breakout: 2.0, stop: 4.0, take: 12.0, size: 20, maxpos: 3, cooldown: 5, atr: 14 })
    property var extraRuns: []

    property int _totalBars: 122835

    // --- data accessors ---
    function allBacktests() { return extraRuns.concat(MD.baseBacktests()); }
    function currentBacktest() {
        var all = allBacktests();
        for (var i = 0; i < all.length; i++)
            if (all[i].id === selectedBacktest) return all[i];
        return MD.baseBacktests()[0];
    }
    function latestRun() {
        var all = allBacktests();
        for (var i = 0; i < all.length; i++)
            if (all[i].status === "completed") return all[i];
        return MD.baseBacktests()[0];
    }
    function dataFilesObj() { return MD.dataFiles(); }
    function strategySource() { return MD.strategySources()[strategy] || ""; }

    // data passthroughs (App is the single data seam for the views/charts)
    function logs() { return MD.buildLogs(latestRun()); }
    function liveLogList() { return MD.liveLog(progress, params, dataFile, strategy); }
    function symMeta(id) { return MD.symMeta(id); }
    function candlesFor(id) { return MD.candlesFor(id); }
    function tradesFor(id) { return MD.tradesFor(id); }
    function equityFor(bt) { return MD.equitySeries(bt); }
    function drawdownFor(bt) { return MD.drawdownOf(MD.equitySeries(bt)); }
    function sparkSeries(seed, bias) { return MD.sparkSeries(seed, bias); }

    // --- nav / actions ---
    function go(v) { view = v }
    function goBacktests() { view = "backtests" }
    function goSetup() { view = "setup" }

    function openBacktest(id) {
        var all = allBacktests();
        var bt = null;
        for (var i = 0; i < all.length; i++) if (all[i].id === id) { bt = all[i]; break; }
        var sym = (bt && bt.symbols && bt.symbols[0]) ? bt.symbols[0].id : "NVDA";
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

    function setParam(k, val) {
        var p = {};
        for (var key in params) p[key] = params[key];
        p[k] = val;
        params = p;
    }

    // --- run lifecycle ---
    function nextId() {
        var mx = 4271;
        var all = allBacktests();
        for (var i = 0; i < all.length; i++) {
            var n = parseInt(all[i].id, 10);
            if (!isNaN(n)) mx = Math.max(mx, n);
        }
        return String(mx + 1);
    }

    function runBacktest() {
        progress = 0;
        barsDone = 0;
        view = "running";
        var f = MD.dataFiles()[dataFile];
        _totalBars = f ? (parseInt(f.bars.replace(/[^0-9]/g, ''), 10) || 122835) : 122835;
        _runTimer.start();
    }
    function _tickRun() {
        var p = Math.min(100, progress + (2.2 + Math.random() * 4.2));
        progress = p;
        barsDone = Math.round(_totalBars * p / 100);
        if (p >= 100) { _runTimer.stop(); _finishTimer.start(); }
    }
    function finishRun() {
        var base = MD.baseBacktests();
        var tmpl = base[0];
        for (var i = 0; i < base.length; i++)
            if (base[i].dataKey === dataFile && base[i].status === "completed") { tmpl = base[i]; break; }
        var nid = nextId();
        var nb = {};
        for (var k in tmpl) nb[k] = tmpl[k];
        nb.id = nid;
        nb.strategy = strategy;
        nb.dataKey = dataFile;
        nb.status = "completed";
        nb.seed = (parseInt(nid, 10) % 97) + 3;
        extraRuns = [nb].concat(extraRuns);
        selectedBacktest = nid;
        detailTab = "report";
        symbol = nb.symbols[0].id;
        selectedTrade = null;
        view = "detail";
    }
    function runSaved() {
        var bt = currentBacktest();
        dataFile = bt.dataKey;
        strategy = bt.strategy;
        runBacktest();
    }
    function cancelRun() {
        _runTimer.stop();
        _finishTimer.stop();
        progress = 0;
        view = "backtests";
    }

    property Timer _runTimer: Timer { interval: 95; repeat: true; onTriggered: app._tickRun() }
    property Timer _finishTimer: Timer { interval: 400; repeat: false; onTriggered: app.finishRun() }

    // --- header strings ---
    function sectionLabel() {
        switch (view) {
        case "backtests": return "Backtests"
        case "setup": return "Configure"
        case "running": return "Running backtest"
        case "detail": return currentBacktest().strategy
        case "logs": return "Diagnostic logs"
        }
        return ""
    }
    function topRight() {
        switch (view) {
        case "backtests": return allBacktests().length + " runs"
        case "setup": return "New run"
        case "running": return "RUNNING · " + Math.round(progress) + "%"
        case "detail": { var b = currentBacktest(); return b.status === "saved" ? "SAVED" : "COMPLETED · " + b.elapsed }
        case "logs": return "#" + latestRun().id
        }
        return ""
    }
}
