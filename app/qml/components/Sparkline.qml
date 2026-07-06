import QtQuick
import Stonks
import "../js/format.js" as Fmt

// Per-symbol sparkline (green/red by last-vs-first) with a lightweight, value-only
// hover crosshair. It has no MouseArea of its own — the containing table row drives
// `hoverX`/`hoverActive` so the row keeps its own hover/click behaviour.
Item {
    id: cv
    property var series: []
    property real hoverX: -1
    property bool hoverActive: false

    readonly property real pad: Theme.sp(4)
    readonly property real dMin: (cv.series && cv.series.length) ? Math.min.apply(null, cv.series) : 0
    readonly property real dMax: (cv.series && cv.series.length) ? Math.max.apply(null, cv.series) : 1
    function yOf(v) { return pad + (1 - (v - dMin) / ((dMax - dMin) || 1)) * (height - 2 * pad) }

    onSeriesChanged: dataCanvas.requestPaint()
    onHoverXChanged: overlay.requestPaint()
    onHoverActiveChanged: overlay.requestPaint()

    Canvas {
        id: dataCanvas
        anchors.fill: parent
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
        onPaint: {
            var ctx = getContext('2d');
            var w = width, h = height;
            ctx.clearRect(0, 0, w, h);
            var data = cv.series;
            if (!data || data.length < 2) return;
            function X(i) { return i / (data.length - 1) * w; }

            ctx.beginPath();
            for (var i = 0; i < data.length; i++) {
                var px = X(i), py = cv.yOf(data[i]);
                if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
            }
            ctx.strokeStyle = (data[data.length - 1] >= data[0]) ? Theme.positive : Theme.negative;
            ctx.lineWidth = 1.5;
            ctx.lineJoin = 'round';
            ctx.stroke();
        }
    }

    Canvas {
        id: overlay
        anchors.fill: parent
        clip: true
        onPaint: {
            var ctx = getContext('2d');
            var w = width, h = height;
            ctx.clearRect(0, 0, w, h);
            var data = cv.series;
            var n = data ? data.length : 0;
            if (!cv.hoverActive || n < 2) return;

            var idx = Math.round(cv.hoverX / w * (n - 1));
            idx = Math.max(0, Math.min(n - 1, idx));
            var x = idx / (n - 1) * w, y = cv.yOf(data[idx]);

            ctx.strokeStyle = 'rgba(255,255,255,0.30)'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
            ctx.fillStyle = Theme.textPrimary;
            ctx.beginPath(); ctx.arc(x, y, Theme.sp(2), 0, Math.PI * 2); ctx.fill();

            var v = data[idx];
            var vtxt = v >= 100 ? '$' + Fmt.commas(Math.round(v)) : '$' + v.toFixed(2);
            ctx.font = Theme.sp(10) + "px '" + Theme.mono + "'";
            ctx.textBaseline = 'top';
            var tw = ctx.measureText(vtxt).width;
            var tx = Math.max(0, Math.min(w - tw, x - tw / 2));
            ctx.fillStyle = Theme.textPrimary;
            ctx.fillText(vtxt, tx, 0);
        }
    }
}
