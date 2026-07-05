import QtQuick
import Stonks

// Small label/value stat (report stats grid, trade-detail fields).
Column {
    property string label: ""
    property string value: ""
    property color valueColor: Theme.t1
    property int valuePx: Theme.fontBodyLg
    spacing: Theme.sp(5)
    Text {
        text: label
        color: Theme.t5
        font.family: Theme.mono
        font.weight: Font.Medium
        font.pixelSize: Theme.fontMicro
        font.letterSpacing: 0.6 * Theme.scale
    }
    Text {
        text: value
        color: valueColor
        font.family: Theme.mono
        font.weight: Font.Medium
        font.pixelSize: valuePx
    }
}
