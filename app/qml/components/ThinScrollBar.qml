import QtQuick
import QtQuick.Controls
import Stonks

// Slim, theme-tinted vertical scrollbar. Fades in while scrolling or hovering
// and stays out of the way otherwise. Attach with `ScrollBar.vertical: ThinScrollBar {}`.
ScrollBar {
    id: sb
    policy: ScrollBar.AsNeeded
    padding: Theme.sp(2)
    implicitWidth: Theme.sp(10)

    contentItem: Rectangle {
        implicitWidth: Theme.sp(4)
        radius: width / 2
        color: sb.pressed ? Theme.t4 : Theme.t6
        opacity: sb.active ? 0.9 : 0.0
        Behavior on opacity { NumberAnimation { duration: 150 } }
    }
}
