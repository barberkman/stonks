import QtQuick
import Stonks
import "../js/format.js" as Fmt

// Portfolio equity: gradient area fill under a 2px line, with a hover crosshair
// (value + date) overlaid.
Item {
    id: cv
    property var series: []
    property var times: []
    property color lineColor: Theme.accent
    property bool grid: true

    // scale shared by the data line and the crosshair so they line up exactly
    readonly property real padTop: Theme.sp(12)
    readonly property real padBottom: 2
    readonly property real dMin: (cv.series && cv.series.length) ? Math.min.apply(null, cv.series) : 0
    readonly property real dMax: (cv.series && cv.series.length) ? Math.max.apply(null, cv.series) : 1
    function yOf(v) { return padTop + (1 - (v - dMin) / ((dMax - dMin) || 1)) * (height - padTop - padBottom) }

    onSeriesChanged: dataCanvas.requestPaint()
    onLineColorChanged: dataCanvas.requestPaint()
    onGridChanged: dataCanvas.requestPaint()

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
            var t = cv.padTop, b = cv.padBottom;
            function X(i) { return i / (data.length - 1) * w; }
            function Y(v) { return cv.yOf(v); }

            if (cv.grid) {
                ctx.strokeStyle = 'rgba(255,255,255,0.05)';
                ctx.lineWidth = 1;
                for (var g = 0; g <= 3; g++) {
                    var yy = t + g / 3 * (h - t - b);
                    ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(w, yy); ctx.stroke();
                }
            }

            var grad = ctx.createLinearGradient(0, 0, 0, h);
            grad.addColorStop(0, Fmt.rgbaColor(cv.lineColor, 0.26));
            grad.addColorStop(1, Fmt.rgbaColor(cv.lineColor, 0));
            ctx.beginPath();
            ctx.moveTo(0, Y(data[0]));
            for (var i = 0; i < data.length; i++) ctx.lineTo(X(i), Y(data[i]));
            ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
            ctx.fillStyle = grad; ctx.fill();

            ctx.beginPath();
            for (var j = 0; j < data.length; j++) {
                var px = X(j), py = Y(data[j]);
                if (j) ctx.lineTo(px, py); else ctx.moveTo(px, py);
            }
            ctx.strokeStyle = cv.lineColor;
            ctx.lineWidth = 2;
            ctx.lineJoin = 'round';
            ctx.stroke();
        }
    }

    LineCrosshair {
        anchors.fill: parent
        series: cv.series
        times: cv.times
        yOf: cv.yOf
        padTop: cv.padTop
        padBottom: cv.padBottom
        markColor: cv.lineColor
        valueText: function (v) { return '$' + Fmt.commas(Math.round(v)) }
    }
}
