import QtQuick
import Stonks
import "../js/format.js" as Fmt

// Candlestick chart with volume + trade markers (ports support.js drawChart).
// When a trade is selected, zooms to its window and overlays entry/exit guides + tags.
Canvas {
    id: cv
    property string symbol: "NVDA"
    property var selectedTrade: null
    property string accentHex: "#4eb36e"

    onSymbolChanged: requestPaint()
    onSelectedTradeChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()

    onPaint: {
        var ctx = getContext('2d');
        var w = width, h = height;
        ctx.clearRect(0, 0, w, h);

        var candles = App.candlesFor(symbol), trades = App.tradesFor(symbol);
        if (!candles || candles.length === 0) return;
        var sel = (selectedTrade === null || selectedTrade === undefined) ? null : selectedTrade;

        var lo = 0, hi = candles.length - 1;
        if (sel !== null && trades[sel]) { var ts = trades[sel]; lo = Math.max(0, ts.entryIdx - 9); hi = Math.min(candles.length - 1, ts.exitIdx + 9); }

        var pmin = Infinity, pmax = -Infinity;
        for (var i = lo; i <= hi; i++) { pmin = Math.min(pmin, candles[i].l); pmax = Math.max(pmax, candles[i].h); }
        var pd = (pmax - pmin) * 0.10; pmin -= pd; pmax += pd;

        var padL = 10, padR = 64, padB = 24, volH = 40;
        var cTop = 14, cBot = h - padB - volH - 8, cH = cBot - cTop, plotW = w - padL - padR, n = hi - lo + 1, step = plotW / n;
        var cw = Math.min(Math.max(step * 0.6, 2), 16);
        function X(i) { return padL + (i - lo + 0.5) * step; }
        function Y(p) { return cTop + (1 - (p - pmin) / (pmax - pmin)) * cH; }
        var ACC = accentHex;

        // price grid + axis labels
        ctx.font = "11px 'IBM Plex Mono'"; ctx.textBaseline = 'middle';
        for (var g = 0; g <= 4; g++) {
            var gp = pmin + (g / 4) * (pmax - pmin), yy = Y(gp);
            ctx.strokeStyle = 'rgba(255,255,255,0.045)'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
            ctx.fillStyle = '#5b626c'; ctx.textAlign = 'left';
            ctx.fillText('$' + gp.toFixed(gp < 5 ? 3 : 1), w - padR + 9, yy);
        }

        // selected-trade overlay: P&L band + dashed guides + tags
        if (sel !== null && trades[sel]) {
            var t = trades[sel]; var x1 = X(t.entryIdx), x2 = X(t.exitIdx); var win = t.pnlNum >= 0; var c = win ? '#4eb36e' : '#e1574c';
            ctx.fillStyle = Fmt.hexA(c, 0.06); ctx.fillRect(x1, cTop, x2 - x1, cH);
            var yE = Y(t.entryPrice), yX = Y(t.exitPrice);
            ctx.save(); ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
            ctx.strokeStyle = Fmt.hexA(ACC, 0.7); ctx.beginPath(); ctx.moveTo(padL, yE); ctx.lineTo(w - padR, yE); ctx.stroke();
            ctx.strokeStyle = Fmt.hexA(c, 0.7); ctx.beginPath(); ctx.moveTo(padL, yX); ctx.lineTo(w - padR, yX); ctx.stroke();
            ctx.restore();
            var tag = function (y, txt, col) {
                ctx.font = "bold 11px 'IBM Plex Mono'";
                var tw = ctx.measureText(txt).width + 12;
                ctx.fillStyle = col; ctx.fillRect(padL, y - 9, tw, 18);
                ctx.fillStyle = '#181818'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                ctx.fillText(txt, padL + 6, y + 1);
            };
            tag(yE, 'ENTRY ' + t.entryPrice.toFixed(2), ACC);
            tag(yX, 'EXIT ' + t.exitPrice.toFixed(2), c);
        }

        // volume bars
        var vmax = 0; for (var iv = lo; iv <= hi; iv++) vmax = Math.max(vmax, candles[iv].v);
        var volBot = h - padB;
        for (var ib = lo; ib <= hi; ib++) {
            var cb = candles[ib], upb = cb.c >= cb.o, xb = X(ib), vh = (cb.v / vmax) * volH;
            ctx.fillStyle = upb ? 'rgba(78,179,110,0.20)' : 'rgba(225,87,76,0.20)';
            ctx.fillRect(xb - cw / 2, volBot - vh, cw, vh);
        }

        // candles
        for (var ic = lo; ic <= hi; ic++) {
            var cc = candles[ic], up = cc.c >= cc.o, col = up ? '#4eb36e' : '#e1574c', xc = X(ic);
            ctx.strokeStyle = col; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(xc, Y(cc.h)); ctx.lineTo(xc, Y(cc.l)); ctx.stroke();
            var yo = Y(cc.o), yc = Y(cc.c), top = Math.min(yo, yc), bh = Math.abs(yc - yo); if (bh < 1) bh = 1;
            ctx.fillStyle = col; ctx.fillRect(xc - cw / 2, top, cw, bh);
        }

        // date ticks
        ctx.fillStyle = '#5b626c'; ctx.font = "11px 'IBM Plex Mono'"; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
        var ticks = Math.min(6, n - 1);
        for (var k = 0; k <= ticks; k++) {
            var it = Math.round(lo + (k / ticks) * (hi - lo));
            ctx.fillText(Fmt.dShort(it), Math.max(padL + 14, Math.min(w - padR - 14, X(it))), h - 7);
        }

        // trade markers
        var tri = function (x, y, s, dir, color) {
            ctx.fillStyle = color; ctx.beginPath();
            if (dir === 'up') { ctx.moveTo(x, y); ctx.lineTo(x - s, y + s * 1.4); ctx.lineTo(x + s, y + s * 1.4); }
            else { ctx.moveTo(x, y); ctx.lineTo(x - s, y - s * 1.4); ctx.lineTo(x + s, y - s * 1.4); }
            ctx.closePath(); ctx.fill();
        };
        for (var ti = 0; ti < trades.length; ti++) {
            var tt = trades[ti];
            if (sel !== null && ti !== sel) continue;
            if (tt.exitIdx < lo || tt.entryIdx > hi) continue;
            var eX = X(tt.entryIdx), xX = X(tt.exitIdx), win2 = tt.pnlNum >= 0, tc = win2 ? '#4eb36e' : '#e1574c';
            if (sel === null) {
                ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = Fmt.hexA(tc, 0.45); ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(eX, Y(tt.entryPrice)); ctx.lineTo(xX, Y(tt.exitPrice)); ctx.stroke(); ctx.restore();
            }
            tri(eX, Y(candles[tt.entryIdx].l) + 13, 6, 'up', ACC);
            tri(xX, Y(candles[tt.exitIdx].h) - 13, 6, 'down', tc);
        }
    }
}
