import QtQuick
import Stonks
import "../js/format.js" as Fmt

// Portfolio equity: gradient area fill under a 2px line (ports support.js drawEquity).
Canvas {
    id: cv
    property var data: []
    property color lineColor: Theme.accent
    property bool grid: true

    onDataChanged: requestPaint()
    onLineColorChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()

    onPaint: {
        var ctx = getContext('2d');
        var w = width, h = height;
        ctx.clearRect(0, 0, w, h);
        if (!data || data.length < 2) return;
        var min = Math.min.apply(null, data), max = Math.max.apply(null, data), t = 12, b = 2;
        function X(i) { return i / (data.length - 1) * w; }
        function Y(v) { return t + (1 - (v - min) / ((max - min) || 1)) * (h - t - b); }

        if (grid) {
            ctx.strokeStyle = 'rgba(255,255,255,0.05)';
            ctx.lineWidth = 1;
            for (var g = 0; g <= 3; g++) {
                var yy = t + g / 3 * (h - t - b);
                ctx.beginPath(); ctx.moveTo(0, yy); ctx.lineTo(w, yy); ctx.stroke();
            }
        }

        var grad = ctx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, Fmt.rgbaColor(lineColor, 0.26));
        grad.addColorStop(1, Fmt.rgbaColor(lineColor, 0));
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
        ctx.strokeStyle = lineColor;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.stroke();
    }
}
