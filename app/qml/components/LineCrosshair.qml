import QtQuick
import Stonks
import "../js/format.js" as Fmt

// Crosshair overlay for single-series line charts (equity / drawdown). Snaps to
// the nearest data point and draws dashed guides, a marker dot, and value + date
// axis tags. The parent supplies the y-scale (`yOf`) and value formatter
// (`valueText`) so the crosshair lines up exactly with the plotted line. Cheap:
// its own Canvas repaints only on hover, never the data line beneath it.
Item {
    id: xh
    property var series: []
    property var times: []
    property var yOf: null              // function(value) -> pixel y
    property var valueText: null        // function(value) -> string
    property real padTop: 0
    property real padBottom: 0
    property color markColor: Theme.accent

    property real hoverX: -1
    property real hoverY: -1
    property bool hoverActive: false

    onHoverXChanged: overlay.requestPaint()
    onHoverYChanged: overlay.requestPaint()
    onHoverActiveChanged: overlay.requestPaint()
    onSeriesChanged: overlay.requestPaint()
    onWidthChanged: overlay.requestPaint()
    onHeightChanged: overlay.requestPaint()

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.CrossCursor
        onPositionChanged: function (mouse) { xh.hoverX = mouse.x; xh.hoverY = mouse.y; xh.hoverActive = true }
        onExited: xh.hoverActive = false
    }

    Canvas {
        id: overlay
        anchors.fill: parent
        onPaint: {
            var ctx = getContext('2d');
            var w = width, h = height;
            ctx.clearRect(0, 0, w, h);
            var n = xh.series ? xh.series.length : 0;
            if (!xh.hoverActive || n < 2 || !xh.yOf) return;

            var idx = Math.round(xh.hoverX / w * (n - 1));
            idx = Math.max(0, Math.min(n - 1, idx));
            var x = idx / (n - 1) * w;
            var y = xh.yOf(xh.series[idx]);

            // dashed crosshair through the snapped point
            ctx.save();
            ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(255,255,255,0.28)';
            ctx.beginPath(); ctx.moveTo(x, xh.padTop); ctx.lineTo(x, h - xh.padBottom); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
            ctx.restore();

            // marker dot
            ctx.fillStyle = xh.markColor;
            ctx.beginPath(); ctx.arc(x, y, Theme.sp(3), 0, Math.PI * 2); ctx.fill();

            ctx.font = Theme.sp(11) + "px '" + Theme.mono + "'";
            ctx.textBaseline = 'middle';

            // value tag, right edge at the point's y
            var vtxt = xh.valueText ? xh.valueText(xh.series[idx]) : '';
            if (vtxt.length) {
                var vw = ctx.measureText(vtxt).width + Theme.sp(10);
                var vy = Math.max(xh.padTop + Theme.sp(9), Math.min(h - xh.padBottom - Theme.sp(9), y));
                ctx.fillStyle = '#323232';
                ctx.fillRect(w - vw, vy - Theme.sp(9), vw, Theme.sp(18));
                ctx.fillStyle = '#e6e8ea'; ctx.textAlign = 'left';
                ctx.fillText(vtxt, w - vw + Theme.sp(5), vy + 1);
            }

            // date tag, bottom edge at the point's x
            if (xh.times && xh.times.length === n) {
                var dtxt = Fmt.tsShort(xh.times[idx]);
                var dw = ctx.measureText(dtxt).width + Theme.sp(12);
                var dx = Math.max(0, Math.min(w - dw, x - dw / 2));
                ctx.fillStyle = '#323232';
                ctx.fillRect(dx, h - Theme.sp(18), dw, Theme.sp(18));
                ctx.fillStyle = '#e6e8ea'; ctx.textAlign = 'center';
                ctx.fillText(dtxt, dx + dw / 2, h - Theme.sp(9) + 1);
            }
        }
    }
}
