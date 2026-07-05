import QtQuick
import Stonks

// Spinning ring (running indicator).
Item {
    id: root
    width: Theme.sp(22)
    height: Theme.sp(22)

    Canvas {
        anchors.fill: parent
        onPaint: {
            var ctx = getContext('2d');
            var w = width, h = height;
            ctx.clearRect(0, 0, w, h);
            var cx = w / 2, cy = h / 2, r = w / 2 - 1.5;
            ctx.lineWidth = 2;
            ctx.strokeStyle = "#323232";
            ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
            ctx.strokeStyle = Theme.accent;
            ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI / 2, 0); ctx.stroke();
        }
    }

    RotationAnimator on rotation {
        from: 0; to: 360
        duration: 800
        loops: Animation.Infinite
        running: true
    }
}
