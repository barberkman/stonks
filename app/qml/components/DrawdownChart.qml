import QtQuick
import Stonks
import "../js/format.js" as Fmt

// Drawdown: gradient area + line, scaled 0 (top) to the trough, with a hover
// crosshair (value + date) overlaid.
Item {
    id: cv
    property var data: []
    property var times: []
    property color lineColor: Theme.negative

    // scale shared by the data line and the crosshair (min = trough, negative)
    readonly property real padTop: Theme.sp(4)
    readonly property real padBottom: 3
    readonly property real dMin: (data && data.length) ? (Math.min.apply(null, data) || -0.0001) : -0.0001
    function yOf(v) { return padTop + (v / dMin) * (height - padTop - padBottom) }

    onDataChanged: dataCanvas.requestPaint()
    onLineColorChanged: dataCanvas.requestPaint()

    Canvas {
        id: dataCanvas
        anchors.fill: parent
        onWidthChanged: requestPaint()
        onHeightChanged: requestPaint()
        onPaint: {
            var ctx = getContext('2d');
            var w = width, h = height;
            ctx.clearRect(0, 0, w, h);
            var data = cv.data;
            if (!data || data.length < 2) return;
            var top = cv.padTop;
            function X(i) { return i / (data.length - 1) * w; }
            function Y(v) { return cv.yOf(v); }

            var grad = ctx.createLinearGradient(0, 0, 0, h);
            grad.addColorStop(0, Fmt.rgbaColor(cv.lineColor, 0.30));
            grad.addColorStop(1, Fmt.rgbaColor(cv.lineColor, 0.02));
            ctx.beginPath();
            ctx.moveTo(0, top);
            for (var i = 0; i < data.length; i++) ctx.lineTo(X(i), Y(data[i]));
            ctx.lineTo(w, top); ctx.closePath();
            ctx.fillStyle = grad; ctx.fill();

            ctx.beginPath();
            for (var j = 0; j < data.length; j++) {
                var px = X(j), py = Y(data[j]);
                if (j) ctx.lineTo(px, py); else ctx.moveTo(px, py);
            }
            ctx.strokeStyle = cv.lineColor;
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    }

    LineCrosshair {
        anchors.fill: parent
        series: cv.data
        times: cv.times
        yOf: cv.yOf
        padTop: cv.padTop
        padBottom: cv.padBottom
        markColor: cv.lineColor
        valueText: function (v) { return (v * 100).toFixed(2) + '%' }
    }
}
