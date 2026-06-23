import QtQuick
import Stonks

// Per-symbol sparkline (ports drawSpark); green/red by last-vs-first.
Canvas {
    id: cv
    property var data: []

    onDataChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()

    onPaint: {
        var ctx = getContext('2d');
        var w = width, h = height;
        ctx.clearRect(0, 0, w, h);
        if (!data || data.length < 2) return;
        var min = Math.min.apply(null, data), max = Math.max.apply(null, data), p = 4;
        function X(i) { return i / (data.length - 1) * w; }
        function Y(v) { return p + (1 - (v - min) / ((max - min) || 1)) * (h - 2 * p); }

        ctx.beginPath();
        for (var i = 0; i < data.length; i++) {
            var px = X(i), py = Y(data[i]);
            if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
        }
        ctx.strokeStyle = (data[data.length - 1] >= data[0]) ? Theme.positive : Theme.negative;
        ctx.lineWidth = 1.5;
        ctx.lineJoin = 'round';
        ctx.stroke();
    }
}
