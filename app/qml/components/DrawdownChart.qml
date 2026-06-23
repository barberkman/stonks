import QtQuick
import Stonks
import "../js/format.js" as Fmt

// Drawdown: gradient area + line, scaled 0 (top) to the trough (ports drawDD).
Canvas {
    id: cv
    property var data: []
    property color lineColor: Theme.negative

    onDataChanged: requestPaint()
    onLineColorChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()

    onPaint: {
        var ctx = getContext('2d');
        var w = width, h = height;
        ctx.clearRect(0, 0, w, h);
        if (!data || data.length < 2) return;
        var min = Math.min.apply(null, data) || -0.0001, top = 4;
        function X(i) { return i / (data.length - 1) * w; }
        function Y(v) { return top + (v / min) * (h - top - 3); }

        var grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, Fmt.rgbaColor(lineColor, 0.30));
        grad.addColorStop(1, Fmt.rgbaColor(lineColor, 0.02));
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
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}
