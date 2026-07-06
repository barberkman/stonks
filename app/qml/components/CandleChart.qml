import QtQuick
import Stonks
import "../js/format.js" as Fmt
import "../js/chartview.js" as CV

// TradingView-style candlestick chart: volume pane + trade markers, drag to
// pan, wheel/trackpad/pinch zoom anchored at the cursor, crosshair with an
// OHLCV readout. The visible window [viewLo, viewHi] lives in fractional
// candle-index space and the price axis auto-fits the visible slice; when the
// window shows more candles than ~2px each, the paint draws per-pixel OHLC
// buckets instead of raw candles (see chartview.js). Selecting a trade seeks
// the view to its range once and overlays entry/exit guides + the P&L band;
// the camera stays free afterwards.
Item {
    id: cv
    property string symbol: "NVDA"
    property var selectedTrade: null
    property string accentHex: "#4eb36e"

    readonly property var series: App.candlesFor(symbol)   // columnar {t,o,h,l,c,v}
    readonly property var trades: App.tradesFor(symbol)
    // strategy-published overlays: [{name, doc, color, values}], each `values`
    // parallel to series.t with null gaps (e.g. warmup)
    readonly property var indicators: App.indicatorsFor(symbol)
    readonly property int candleCount: (series && series.t) ? series.t.length : 0

    // visible window (fractional candle indices) + zoom bounds
    property real viewLo: 0
    property real viewHi: Math.max(0, candleCount - 1)
    readonly property real minSpan: 8

    // plot gutters shared by both canvases and the pixel<->index mapping
    readonly property real padL: Theme.sp(10)
    readonly property real padR: Theme.sp(64)
    readonly property real padB: Theme.sp(24)
    readonly property real volH: Theme.sp(40)
    readonly property real plotW: width - padL - padR

    property real hoverX: -1
    property real hoverY: -1
    property bool hoverActive: false
    property var _view: null   // geometry/scale cache written by dataCanvas for the crosshair

    onSeriesChanged: { fit(); repaint() }
    onIndicatorsChanged: repaint()
    onSelectedTradeChanged: { seekToTrade(); repaint() }
    onViewLoChanged: repaint()
    onViewHiChanged: repaint()
    onWidthChanged: repaint()
    onHeightChanged: repaint()
    onHoverXChanged: crosshairCanvas.requestPaint()
    onHoverYChanged: crosshairCanvas.requestPaint()
    onHoverActiveChanged: crosshairCanvas.requestPaint()

    function repaint() { dataCanvas.requestPaint(); crosshairCanvas.requestPaint() }
    function fit() { viewLo = 0; viewHi = Math.max(0, candleCount - 1) }
    function seekToTrade() {
        var sel = selectedTrade
        if (sel === null || sel === undefined || !trades[sel]) return
        var t = trades[sel]
        var r = CV.clampPan(t.entryIdx - 9, t.exitIdx + 9, candleCount, minSpan)
        viewLo = r.lo; viewHi = r.hi
    }
    function pan(dCandles) {
        var r = CV.clampPan(viewLo + dCandles, viewHi + dCandles, candleCount, minSpan)
        viewLo = r.lo; viewHi = r.hi
    }
    function zoomAt(px, factor) {
        var r = CV.zoomedRange(viewLo, viewHi, (px - padL) / plotW, factor, candleCount, minSpan)
        viewLo = r.lo; viewHi = r.hi
    }

    Canvas {
        id: dataCanvas
        anchors.fill: parent
        onPaint: {
            var ctx = getContext('2d');
            var w = width, h = height;
            ctx.clearRect(0, 0, w, h);

            var s = cv.series, trades = cv.trades;
            if (!s || !s.t || s.t.length === 0) { cv._view = null; return; }
            var sel = (cv.selectedTrade === null || cv.selectedTrade === undefined) ? null : cv.selectedTrade;

            var lo = cv.viewLo, hi = cv.viewHi;
            if (!(hi - lo > 0)) { cv._view = null; return; }
            var ds = CV.visibleSeries(s.t, s.o, s.h, s.l, s.c, s.v, lo, hi, cv.plotW);
            if (ds.t.length === 0) { cv._view = null; return; }

            // indicator overlays, reduced onto the same visible buckets as the
            // candles; their values join the price-axis autofit (below) so a
            // series outside the candle range never clips off the pane
            var inds = cv.indicators, indVals = [];
            for (var ii = 0; ii < inds.length; ii++) {
                indVals.push(CV.bucketIndicator(inds[ii].values, ds.r0, ds.r1));
            }

            var pmin = ds.pmin, pmax = ds.pmax;
            for (var ij = 0; ij < indVals.length; ij++) {
                for (var ik = 0; ik < indVals[ij].length; ik++) {
                    var iw = indVals[ij][ik];
                    if (iw === null || iw === undefined) continue;
                    if (iw < pmin) pmin = iw;
                    if (iw > pmax) pmax = iw;
                }
            }
            var pd = (pmax - pmin) * 0.10;
            if (pd <= 0) pd = Math.max(Math.abs(pmax) * 0.01, 1e-9);   // flat window
            pmin -= pd; pmax += pd;

            var padL = cv.padL, padR = cv.padR, padB = cv.padB, volH = cv.volH;
            var cTop = Theme.sp(14), cBot = h - padB - volH - Theme.sp(8), cH = cBot - cTop;
            var step = cv.plotW / (hi - lo);
            var cw = Math.min(Math.max((cv.plotW / ds.t.length) * 0.6, 1), Theme.sp(16));
            function X(i) { return padL + (i - lo) * step; }
            function Y(p) { return cTop + (1 - (p - pmin) / (pmax - pmin)) * cH; }
            var ACC = cv.accentHex;

            // price grid + axis labels
            ctx.font = Theme.sp(11) + "px '" + Theme.mono + "'"; ctx.textBaseline = 'middle';
            for (var g = 0; g <= 4; g++) {
                var gp = pmin + (g / 4) * (pmax - pmin), yy = Y(gp);
                ctx.strokeStyle = 'rgba(255,255,255,0.045)'; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(padL, yy); ctx.lineTo(w - padR, yy); ctx.stroke();
                ctx.fillStyle = '#5b626c'; ctx.textAlign = 'left';
                ctx.fillText('$' + gp.toFixed(gp < 5 ? 3 : 1), w - padR + Theme.sp(9), yy);
            }

            // selected-trade overlay: P&L band + dashed guides + tags (culled off-view)
            if (sel !== null && trades[sel] && !(trades[sel].exitIdx < lo || trades[sel].entryIdx > hi)) {
                var t = trades[sel]; var x1 = X(t.entryIdx), x2 = X(t.exitIdx); var win = t.pnlNum >= 0; var c = win ? '#4eb36e' : '#e1574c';
                ctx.fillStyle = Fmt.hexA(c, 0.06); ctx.fillRect(x1, cTop, x2 - x1, cH);
                var yE = Y(t.entryPrice), yX = Y(t.exitPrice);
                ctx.save(); ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
                ctx.strokeStyle = Fmt.hexA(ACC, 0.7); ctx.beginPath(); ctx.moveTo(padL, yE); ctx.lineTo(w - padR, yE); ctx.stroke();
                ctx.strokeStyle = Fmt.hexA(c, 0.7); ctx.beginPath(); ctx.moveTo(padL, yX); ctx.lineTo(w - padR, yX); ctx.stroke();
                ctx.restore();
                var tag = function (y, txt, col) {
                    ctx.font = "bold " + Theme.sp(11) + "px '" + Theme.mono + "'";
                    var tw = ctx.measureText(txt).width + Theme.sp(12);
                    ctx.fillStyle = col; ctx.fillRect(padL, y - Theme.sp(9), tw, Theme.sp(18));
                    ctx.fillStyle = '#181818'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                    ctx.fillText(txt, padL + Theme.sp(6), y + 1);
                };
                tag(yE, 'ENTRY ' + t.entryPrice.toFixed(2), ACC);
                tag(yX, 'EXIT ' + t.exitPrice.toFixed(2), c);
                ctx.font = Theme.sp(11) + "px '" + Theme.mono + "'";
            }

            // volume bars
            var volBot = h - padB;
            for (var ib = 0; ib < ds.t.length; ib++) {
                var upb = ds.c[ib] >= ds.o[ib], xb = X(ds.xi[ib]), vh = ds.vmax > 0 ? (ds.v[ib] / ds.vmax) * volH : 0;
                ctx.fillStyle = upb ? 'rgba(78,179,110,0.20)' : 'rgba(225,87,76,0.20)';
                ctx.fillRect(xb - cw / 2, volBot - vh, cw, vh);
            }

            // candles (raw or LOD buckets — same drawing either way)
            for (var ic = 0; ic < ds.t.length; ic++) {
                var up = ds.c[ic] >= ds.o[ic], col = up ? '#4eb36e' : '#e1574c', xc = X(ds.xi[ic]);
                ctx.strokeStyle = col; ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(xc, Y(ds.h[ic])); ctx.lineTo(xc, Y(ds.l[ic])); ctx.stroke();
                var yo = Y(ds.o[ic]), yc = Y(ds.c[ic]), top = Math.min(yo, yc), bh = Math.abs(yc - yo); if (bh < 1) bh = 1;
                ctx.fillStyle = col; ctx.fillRect(xc - cw / 2, top, cw, bh);
            }

            // indicator overlays: one polyline per series through the same X/Y
            // mapping; a null bucket breaks the path (warmup renders as a gap,
            // never a line to zero)
            ctx.lineJoin = 'round'; ctx.lineCap = 'round';
            for (var ip = 0; ip < inds.length; ip++) {
                var pv = indVals[ip];
                ctx.strokeStyle = inds[ip].color; ctx.lineWidth = 1.5;
                ctx.beginPath();
                var started = false;
                for (var iq = 0; iq < pv.length; iq++) {
                    if (pv[iq] === null || pv[iq] === undefined) { started = false; continue; }
                    var px = X(ds.xi[iq]), py = Y(pv[iq]);
                    if (started) { ctx.lineTo(px, py); } else { ctx.moveTo(px, py); }
                    started = true;
                }
                ctx.stroke();
            }

            // date ticks
            ctx.fillStyle = '#5b626c'; ctx.font = Theme.sp(11) + "px '" + Theme.mono + "'"; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
            var ticks = Math.min(6, ds.t.length - 1);
            for (var k = 0; k <= ticks; k++) {
                var di = Math.round((k / Math.max(1, ticks)) * (ds.t.length - 1));
                ctx.fillText(Fmt.tsShort(ds.t[di]), Math.max(padL + Theme.sp(14), Math.min(w - padR - Theme.sp(14), X(ds.xi[di]))), h - Theme.sp(7));
            }

            // trade markers (raw indices through the same X/Y mapping)
            var tri = function (x, y, sz, dir, color) {
                ctx.fillStyle = color; ctx.beginPath();
                if (dir === 'up') { ctx.moveTo(x, y); ctx.lineTo(x - sz, y + sz * 1.4); ctx.lineTo(x + sz, y + sz * 1.4); }
                else { ctx.moveTo(x, y); ctx.lineTo(x - sz, y - sz * 1.4); ctx.lineTo(x + sz, y - sz * 1.4); }
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
                tri(eX, Y(s.l[tt.entryIdx]) + Theme.sp(13), Theme.sp(6), 'up', ACC);
                tri(xX, Y(s.h[tt.exitIdx]) - Theme.sp(13), Theme.sp(6), 'down', tc);
            }

            // indicator legend: color swatch + series name, pinned top-left of
            // the price pane (drawn last so it stays readable over the data)
            if (inds.length > 0) {
                ctx.font = Theme.sp(11) + "px '" + Theme.mono + "'";
                ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                var lx = padL + Theme.sp(6), ly = cTop + Theme.sp(10);
                for (var il = 0; il < inds.length; il++) {
                    ctx.fillStyle = inds[il].color;
                    ctx.fillRect(lx, ly - Theme.sp(2), Theme.sp(10), Theme.sp(3));
                    ctx.fillStyle = '#9aa0a8';
                    ctx.fillText(inds[il].name, lx + Theme.sp(14), ly);
                    lx += Theme.sp(14) + ctx.measureText(inds[il].name).width + Theme.sp(16);
                }
            }

            cv._view = { pmin: pmin, pmax: pmax, cTop: cTop, cBot: cBot, cH: cH, volBot: volBot };
        }
    }

    // cheap overlay: repaints on hover moves so the data canvas never has to
    Canvas {
        id: crosshairCanvas
        anchors.fill: parent
        onPaint: {
            var ctx = getContext('2d');
            var w = width, h = height;
            ctx.clearRect(0, 0, w, h);
            var v = cv._view;
            if (!cv.hoverActive || !v || cv.candleCount === 0) return;

            var s = cv.series;
            var lo = cv.viewLo, hi = cv.viewHi, span = hi - lo;
            var iMin = Math.max(0, Math.ceil(lo)), iMax = Math.min(cv.candleCount - 1, Math.floor(hi));
            if (iMax < iMin) return;
            var idx = Math.round(lo + (cv.hoverX - cv.padL) / cv.plotW * span);
            idx = Math.max(iMin, Math.min(iMax, idx));
            var x = cv.padL + (idx - lo) / span * cv.plotW;

            ctx.font = Theme.sp(11) + "px '" + Theme.mono + "'";
            var inPricePane = cv.hoverY >= v.cTop && cv.hoverY <= v.cBot;

            // crosshair lines snapped to the hovered candle
            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = 'rgba(255,255,255,0.28)'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x, v.cTop); ctx.lineTo(x, v.volBot); ctx.stroke();
            if (inPricePane) {
                ctx.beginPath(); ctx.moveTo(cv.padL, cv.hoverY); ctx.lineTo(w - cv.padR, cv.hoverY); ctx.stroke();
            }
            ctx.restore();

            // price tag on the right axis
            if (inPricePane) {
                var price = v.pmin + (1 - (cv.hoverY - v.cTop) / v.cH) * (v.pmax - v.pmin);
                var ptxt = '$' + price.toFixed(price < 5 ? 3 : 2);
                var pw = ctx.measureText(ptxt).width + Theme.sp(10);
                ctx.fillStyle = '#323232';
                ctx.fillRect(w - cv.padR + Theme.sp(4), cv.hoverY - Theme.sp(9), pw, Theme.sp(18));
                ctx.fillStyle = '#e6e8ea'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
                ctx.fillText(ptxt, w - cv.padR + Theme.sp(9), cv.hoverY + 1);
            }

            // date tag on the bottom axis
            var dtxt = Fmt.tsShort(s.t[idx]);
            var dw = ctx.measureText(dtxt).width + Theme.sp(12);
            var dx = Math.max(cv.padL, Math.min(w - cv.padR - dw, x - dw / 2));
            ctx.fillStyle = '#323232';
            ctx.fillRect(dx, h - Theme.sp(20), dw, Theme.sp(18));
            ctx.fillStyle = '#e6e8ea'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(dtxt, dx + dw / 2, h - Theme.sp(11) + 1);

            // OHLCV readout, top-left (raw full-resolution values even when LOD draws)
            var up = s.c[idx] >= s.o[idx];
            var vcol = up ? '#4eb36e' : '#e1574c';
            var ry = Math.max(Theme.sp(8), v.cTop - Theme.sp(6));
            var rx = cv.padL + Theme.sp(2);
            ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
            var fields = [['O', s.o[idx]], ['H', s.h[idx]], ['L', s.l[idx]], ['C', s.c[idx]]];
            for (var f = 0; f < fields.length; f++) {
                ctx.fillStyle = '#5b626c'; ctx.fillText(fields[f][0], rx, ry);
                rx += ctx.measureText(fields[f][0]).width + Theme.sp(4);
                var vtxt = fields[f][1].toFixed(fields[f][1] < 5 ? 3 : 2);
                ctx.fillStyle = vcol; ctx.fillText(vtxt, rx, ry);
                rx += ctx.measureText(vtxt).width + Theme.sp(10);
            }
            ctx.fillStyle = '#5b626c'; ctx.fillText('V', rx, ry);
            rx += ctx.measureText('V').width + Theme.sp(4);
            ctx.fillStyle = vcol; ctx.fillText(Fmt.commas(Math.round(s.v[idx])), rx, ry);
        }
    }

    MouseArea {
        id: ma
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: pressed ? Qt.ClosedHandCursor : Qt.CrossCursor
        property real lastX: 0
        onPressed: function (mouse) { lastX = mouse.x }
        onPositionChanged: function (mouse) {
            cv.hoverX = mouse.x; cv.hoverY = mouse.y; cv.hoverActive = true
            if (pressed) {
                var step = cv.plotW / (cv.viewHi - cv.viewLo)
                cv.pan((lastX - mouse.x) / step)
                lastX = mouse.x
            }
        }
        onExited: cv.hoverActive = false
        onDoubleClicked: cv.fit()
    }

    WheelHandler {
        target: null   // handle the numbers ourselves; never transform the Item
        acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
        onWheel: function (event) {
            if (Math.abs(event.pixelDelta.x) > Math.abs(event.pixelDelta.y)) {
                // trackpad horizontal scroll pans, content follows the fingers
                var step = cv.plotW / (cv.viewHi - cv.viewLo)
                cv.pan(-event.pixelDelta.x / step)
            } else {
                var dy = event.pixelDelta.y !== 0 ? event.pixelDelta.y : (event.angleDelta.y / 120) * 20
                cv.zoomAt(event.x, Math.pow(1.005, -dy))
            }
        }
    }

    PinchHandler {
        id: pinch
        target: null
        onScaleChanged: function (delta) { cv.zoomAt(pinch.centroid.position.x, 1 / delta) }
    }
}
