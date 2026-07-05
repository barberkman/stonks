import QtQuick
import QtQuick.Controls
import Stonks

// Vertically-scrolling page body. Children stack in a centered column whose width
// is min(width, maxWidth) - 2*sidePad (mirrors the design's <main> max-width blocks).
// For full-bleed views (Report/Trades/Logs) use maxWidth huge + sidePad 0.
Item {
    id: root
    property int maxWidth: Theme.sp(1180)
    property int sidePad: Theme.sp(40)
    property int topPad: Theme.sp(32)
    property int bottomPad: Theme.sp(60)
    property int spacing: 0
    default property alias content: col.data
    readonly property real innerWidth: Math.min(width, maxWidth) - sidePad * 2

    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth
        clip: true
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        Item {
            width: root.width
            implicitHeight: col.implicitHeight + root.topPad + root.bottomPad

            Column {
                id: col
                width: root.innerWidth
                x: (root.width - width) / 2
                y: root.topPad
                spacing: root.spacing
            }
        }
    }
}
