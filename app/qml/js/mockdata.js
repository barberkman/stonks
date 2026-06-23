.pragma library

// Mock/synthetic data + series generators ported near-verbatim from the design's
// Component class. This is the single seam for future real-engine data: replace
// these generators with real results and the views/charts are unchanged.

// ---------- seeded RNG (LCG) ----------
function mkRng(seed) {
    var s = (seed >>> 0) || 1;
    return function () {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

// ---------- static data ----------
function dataFiles() {
    return {
        us_megacaps_1h: { label: 'US Megacaps · 1H', symbols: 'NVDA · AAPL · MSFT · AMZN · TSLA', source: 'data/us_megacaps_1h.parquet', bars: '122,835' },
        us_equities_15m: { label: 'US Equities · 15m', symbols: 'AMD · QCOM · AVGO · MU · INTC', source: 'data/us_equities_15m.parquet', bars: '486,300' },
        sp500_daily: { label: 'S&P 500 · Daily', symbols: 'GOOGL · META · NFLX · DIS', source: 'data/sp500_daily.parquet', bars: '628,500' },
        crypto_majors_1h: { label: 'Crypto Majors · 1H', symbols: 'SOL · BTC · ETH · BNB · XRP', source: 'data/crypto_majors_1h.parquet', bars: '215,640' }
    };
}

function strategySources() {
    return {
        'Momentum Breakout v3': 'strategies/momo_v3.py',
        'Mean Reversion (Bollinger)': 'strategies/mean_rev_bb.py',
        'Dual MA Crossover': 'strategies/dual_ma.py',
        'Opening Range Breakout': 'strategies/orb.py'
    };
}

function symMeta(id) {
    var m = {
        NVDA: { name: 'NVIDIA Corp', start: 140, bias: 0.0060, vol: 0.024, seed: 11 }, AAPL: { name: 'Apple Inc', start: 152, bias: 0.0032, vol: 0.018, seed: 23 },
        MSFT: { name: 'Microsoft', start: 285, bias: 0.0028, vol: 0.017, seed: 31 }, AMZN: { name: 'Amazon.com', start: 122, bias: 0.0024, vol: 0.020, seed: 41 },
        TSLA: { name: 'Tesla Inc', start: 255, bias: -0.0026, vol: 0.030, seed: 53 },
        AMD: { name: 'Adv Micro Dev', start: 95, bias: 0.0040, vol: 0.028, seed: 13 }, QCOM: { name: 'Qualcomm', start: 130, bias: 0.0030, vol: 0.022, seed: 29 },
        AVGO: { name: 'Broadcom', start: 480, bias: 0.0050, vol: 0.020, seed: 43 }, MU: { name: 'Micron Tech', start: 70, bias: 0.0035, vol: 0.030, seed: 37 },
        INTC: { name: 'Intel Corp', start: 48, bias: -0.0010, vol: 0.020, seed: 17 },
        GOOGL: { name: 'Alphabet A', start: 140, bias: 0.0022, vol: 0.018, seed: 67 }, META: { name: 'Meta Platforms', start: 330, bias: 0.0010, vol: 0.026, seed: 73 },
        NFLX: { name: 'Netflix', start: 380, bias: -0.0008, vol: 0.030, seed: 79 }, DIS: { name: 'Walt Disney', start: 95, bias: -0.0016, vol: 0.020, seed: 83 },
        BTC: { name: 'Bitcoin', start: 280, bias: 0.0060, vol: 0.030, seed: 5 }, ETH: { name: 'Ethereum', start: 160, bias: 0.0055, vol: 0.035, seed: 91 },
        SOL: { name: 'Solana', start: 22, bias: 0.0090, vol: 0.050, seed: 97 }, BNB: { name: 'BNB', start: 240, bias: 0.0030, vol: 0.030, seed: 101 },
        XRP: { name: 'XRP', start: 55, bias: -0.0014, vol: 0.040, seed: 103 }
    };
    return m[id] || { name: id, start: 100, bias: 0.003, vol: 0.02, seed: 7 };
}

function baseBacktests() {
    return [
        { id: '4271', strategy: 'Momentum Breakout v3', dataKey: 'us_megacaps_1h', range: '2021-05-27 → 2026-05-22', status: 'completed', seed: 7,
            ret: '+63.4%', retPos: true, maxdd: '-18.7%', win: '57.3%', trades: 248, sharpe: '1.84',
            startEqStr: '$100,000', endEqStr: '$163,420', bars: '122,835', orders: '496', notional: '$4.82M', pf: '1.71', elapsed: '9.84s', perbar: '80.1µs',
            symbols: [ { id: 'NVDA', ret: '+156.0%', retPos: true, pnl: '+$31,200', trades: 58, win: '62%' }, { id: 'AAPL', ret: '+67.0%', retPos: true, pnl: '+$13,400', trades: 52, win: '56%' },
                { id: 'MSFT', ret: '+59.0%', retPos: true, pnl: '+$11,800', trades: 44, win: '59%' }, { id: 'AMZN', ret: '+48.0%', retPos: true, pnl: '+$9,600', trades: 49, win: '53%' },
                { id: 'TSLA', ret: '-12.9%', retPos: false, pnl: '-$2,580', trades: 45, win: '49%' } ] },
        { id: '4263', strategy: 'Mean Reversion (Bollinger)', dataKey: 'us_equities_15m', range: '2022-01-03 → 2026-05-22', status: 'completed', seed: 19,
            ret: '+28.9%', retPos: true, maxdd: '-11.2%', win: '61.0%', trades: 412, sharpe: '1.42',
            startEqStr: '$100,000', endEqStr: '$128,900', bars: '486,300', orders: '824', notional: '$9.10M', pf: '1.39', elapsed: '34.10s', perbar: '70.1µs',
            symbols: [ { id: 'AMD', ret: '+44.0%', retPos: true, pnl: '+$8,800', trades: 96, win: '63%' }, { id: 'AVGO', ret: '+38.0%', retPos: true, pnl: '+$7,600', trades: 71, win: '64%' },
                { id: 'QCOM', ret: '+31.0%', retPos: true, pnl: '+$6,200', trades: 84, win: '60%' }, { id: 'MU', ret: '+12.0%', retPos: true, pnl: '+$2,400', trades: 90, win: '58%' },
                { id: 'INTC', ret: '-6.5%', retPos: false, pnl: '-$1,300', trades: 71, win: '52%' } ] },
        { id: '4258', strategy: 'Dual MA Crossover', dataKey: 'sp500_daily', range: '2018-01-02 → 2026-05-22', status: 'completed', seed: 29,
            ret: '-7.2%', retPos: false, maxdd: '-24.6%', win: '41.0%', trades: 96, sharpe: '0.38',
            startEqStr: '$100,000', endEqStr: '$92,800', bars: '628,500', orders: '192', notional: '$3.05M', pf: '0.89', elapsed: '41.90s', perbar: '66.7µs',
            symbols: [ { id: 'GOOGL', ret: '+8.0%', retPos: true, pnl: '+$1,600', trades: 24, win: '46%' }, { id: 'META', ret: '-4.0%', retPos: false, pnl: '-$800', trades: 26, win: '42%' },
                { id: 'NFLX', ret: '-12.0%', retPos: false, pnl: '-$2,400', trades: 22, win: '38%' }, { id: 'DIS', ret: '-18.0%', retPos: false, pnl: '-$3,600', trades: 24, win: '36%' } ] },
        { id: '4251', strategy: 'Opening Range Breakout', dataKey: 'crypto_majors_1h', range: '2023-06-01 → 2026-05-22', status: 'completed', seed: 5,
            ret: '+112.5%', retPos: true, maxdd: '-29.3%', win: '52.0%', trades: 318, sharpe: '1.61',
            startEqStr: '$100,000', endEqStr: '$212,500', bars: '215,640', orders: '636', notional: '$14.2M', pf: '1.58', elapsed: '18.30s', perbar: '84.9µs',
            symbols: [ { id: 'SOL', ret: '+260.0%', retPos: true, pnl: '+$52,000', trades: 64, win: '55%' }, { id: 'BTC', ret: '+88.0%', retPos: true, pnl: '+$17,600', trades: 70, win: '54%' },
                { id: 'ETH', ret: '+74.0%', retPos: true, pnl: '+$14,800', trades: 68, win: '53%' }, { id: 'BNB', ret: '+41.0%', retPos: true, pnl: '+$8,200', trades: 58, win: '50%' },
                { id: 'XRP', ret: '-9.0%', retPos: false, pnl: '-$1,800', trades: 58, win: '47%' } ] },
        { id: '4248', strategy: 'Momentum Breakout v3', dataKey: 'us_megacaps_1h', range: '2021-05-27 → 2024-05-22', status: 'saved', seed: 7,
            ret: '—', retPos: true, maxdd: '—', win: '—', trades: 0, sharpe: '—',
            startEqStr: '$100,000', endEqStr: '—', bars: '73,200', orders: '—', notional: '—', pf: '—', elapsed: '—', perbar: '—',
            symbols: [ { id: 'NVDA' }, { id: 'AAPL' }, { id: 'MSFT' }, { id: 'AMZN' }, { id: 'TSLA' } ] }
    ];
}

// ---------- series (cached at module scope) ----------
var _cc = {};
function candlesFor(id) {
    if (_cc[id]) return _cc[id];
    var m = symMeta(id);
    var r = mkRng(m.seed * 13 + 1);
    var c = m.start;
    var out = [];
    for (var i = 0; i < 120; i++) {
        var o = c;
        var ch = o * (m.bias + (r() - 0.5) * m.vol);
        var cl = Math.max(0.2, o + ch);
        var hi = Math.max(o, cl) + o * r() * m.vol * 0.55;
        var lo = Math.max(0.1, Math.min(o, cl) - o * r() * m.vol * 0.55);
        out.push({ o: o, h: hi, l: lo, c: cl, v: 0.4 + r() * 0.9 });
        c = cl;
    }
    _cc[id] = out;
    return out;
}

var _tc = {};
function tradesFor(id) {
    if (_tc[id]) return _tc[id];
    var candles = candlesFor(id);
    var r = mkRng(symMeta(id).seed * 7 + 5);
    var out = [];
    var i = 5, id2 = 1;
    var alloc = 20000;
    while (i < candles.length - 9 && out.length < 11) {
        var hold = 3 + Math.floor(r() * 7);
        var eIdx = i, xIdx = i + hold;
        var ep = candles[eIdx].c, xp = candles[xIdx].c;
        var qty = Math.max(1, Math.floor(alloc / ep));
        var pnlNum = (xp - ep) * qty;
        var mfe = -1e9, mae = 1e9;
        for (var k = eIdx; k <= xIdx; k++) {
            mfe = Math.max(mfe, (candles[k].h / ep - 1) * 100);
            mae = Math.min(mae, (candles[k].l / ep - 1) * 100);
        }
        out.push({ n: id2++, side: 'LONG', entryIdx: eIdx, exitIdx: xIdx, entryPrice: ep, exitPrice: xp, qty: qty, pnlNum: pnlNum, retNum: (xp / ep - 1) * 100, bars: hold, mfe: mfe, mae: mae });
        i = xIdx + 2 + Math.floor(r() * 6);
    }
    _tc[id] = out;
    return out;
}

var _eq = {};
function equitySeries(bt) {
    if (_eq[bt.id]) return _eq[bt.id];
    var startEq = parseFloat((bt.startEqStr || '$100,000').replace(/[^0-9.]/g, '')) || 100000;
    var endEq = parseFloat((bt.endEqStr || '$100,000').replace(/[^0-9.]/g, '')) || startEq;
    var n = 260, r = mkRng((bt.seed || 7) * 3 + 1);
    var driftBase = Math.pow(endEq / startEq, 1 / n) - 1;
    var v = startEq, a = [v];
    for (var i = 1; i < n; i++) {
        var drift = driftBase;
        if (i > 150 && i < 186) drift -= 0.0045;
        var noise = (r() - 0.5) * 0.022;
        v = Math.max(v * (1 + drift + noise), startEq * 0.2);
        a.push(v);
    }
    _eq[bt.id] = a;
    return a;
}

// sparkline series for a per-symbol seed/bias (mirrors the design's bt-spark canvas)
function sparkSeries(seed, bias) {
    var r = mkRng((seed || 3) * 1 + 1);
    var v = 100, ser = [v];
    for (var i = 1; i < 64; i++) {
        v *= (1 + (bias || 0.005) + (r() - 0.5) * 0.03);
        ser.push(v);
    }
    return ser;
}

// drawdown series derived from an equity series
function drawdownOf(equity) {
    var peak = -Infinity;
    return equity.map(function (v) {
        if (v > peak) peak = v;
        return (v - peak) / peak;
    });
}

// ---------- logs ----------
var _logCol = { INFO: '#9aa0a8', DEBUG: '#6b727c', WARN: '#cbb26a', ERROR: '#e1574c' };

function liveLog(progress, params, dataFileKey, strategy) {
    var f = dataFiles()[dataFileKey] || dataFiles().us_megacaps_1h;
    var seq = [
        [2, 'INFO', 'engine 2.3.1 · loading strategy (' + strategy + ')'],
        [5, 'INFO', 'data: ' + f.source + ' · ' + f.bars + ' bars'],
        [9, 'DEBUG', 'warmup: indicators primed (lookback=' + params.lookback + ', atr=' + params.atr + ')'],
        [14, 'DEBUG', 'calendar aligned to XNYS regular session'],
        [24, 'INFO', 'streaming bars · ' + f.symbols],
        [34, 'INFO', 'fills simulated · slippage model active'],
        [43, 'WARN', 'signals skipped — max positions (' + params.maxpos + ') reached'],
        [55, 'INFO', 'risk: position sizing ' + params.size + '% equity'],
        [64, 'DEBUG', 'risk: peak gross exposure 61.4% of equity'],
        [72, 'WARN', 'bar gap detected — forward-filled'],
        [83, 'INFO', 'accounting: round-trips reconciled'],
        [92, 'DEBUG', 'computing performance metrics'],
        [98, 'INFO', 'run complete · writing report']
    ];
    var out = [];
    for (var i = 0; i < seq.length; i++) {
        var s = seq[i];
        if (progress >= s[0]) out.push({ t: '+' + ((s[0] / 100) * 9.84).toFixed(2) + 's', lvl: s[1], m: s[2], lc: _logCol[s[1]] });
    }
    return out;
}

function buildLogs(bt) {
    var f = dataFiles()[bt.dataKey] || dataFiles().us_megacaps_1h;
    function sym(n) { return bt.symbols[n] || {}; }
    var L = [
        ['00:00.000', 'INFO', 'engine 2.3.1 · loading strategy ' + bt.strategy],
        ['00:00.004', 'INFO', 'universe resolved: ' + f.symbols],
        ['00:00.012', 'DEBUG', 'data adapter: local parquet · ' + f.bars + ' bars total'],
        ['00:00.031', 'INFO', 'warmup: indicators primed (lookback=20, atr=14)'],
        ['00:00.044', 'DEBUG', 'calendar: aligned to XNYS regular session'],
        ['00:01.220', 'INFO', (sym(0).id) + ': ' + (sym(0).trades || 0) + ' trades reconciled'],
        ['00:02.880', 'WARN', 'signals skipped — max positions (3) reached'],
        ['00:03.410', 'DEBUG', 'risk: peak gross exposure 61.4% of equity'],
        ['00:04.105', 'INFO', (sym(1).id) + ': win rate ' + (sym(1).win || '—')],
        ['00:05.660', 'WARN', 'bar gap detected (holiday) — forward-filled'],
        ['00:06.910', 'INFO', (sym(2).id) + ': profit factor ' + bt.pf],
        ['00:08.020', 'DEBUG', 'accounting: ' + bt.orders + ' orders reconciled'],
        ['00:09.440', 'INFO', 'portfolio: ending equity ' + bt.endEqStr + ' · return ' + bt.ret],
        ['00:09.840', 'INFO', 'run #' + bt.id + ' completed in ' + bt.elapsed + ' · ' + bt.perbar + '/bar']
    ];
    return L.map(function (r) { return { t: r[0], lvl: r[1], lc: _logCol[r[1]], m: r[2] }; });
}
