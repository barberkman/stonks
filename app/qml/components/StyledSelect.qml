import QtQuick
import QtQuick.Controls
import Stonks

// Dark dropdown matching the design's <select>. `model` = display strings;
// optional parallel `values`; emits picked(value) on selection.
ComboBox {
    id: cb
    property var values: []
    signal picked(string value)

    font.family: Theme.mono
    font.pixelSize: Theme.fontBodyLg
    implicitHeight: Theme.sp(44)

    onActivated: function (index) { cb.picked(values.length > index ? values[index] : currentText) }

    background: Rectangle {
        color: Theme.input
        border.color: "#3c3c3c"
        border.width: 1
        radius: Theme.radiusControl
    }
    contentItem: Text {
        leftPadding: Theme.sp(13)
        rightPadding: Theme.sp(30)
        text: cb.displayText
        color: Theme.textPrimary
        font: cb.font
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    indicator: Text {
        x: cb.width - width - Theme.sp(12)
        y: (cb.height - height) / 2
        text: "▾"
        color: Theme.t5
        font.family: Theme.mono
        font.pixelSize: Theme.fontSmall
    }
    delegate: ItemDelegate {
        width: cb.width
        height: Theme.sp(36)
        contentItem: Text {
            text: modelData
            color: Theme.textPrimary
            font.family: Theme.mono
            font.pixelSize: Theme.fontBody
            verticalAlignment: Text.AlignVCenter
        }
        background: Rectangle { color: highlighted ? Theme.accentSoft : Theme.card }
    }
    popup: Popup {
        y: cb.height + Theme.sp(4)
        width: cb.width
        padding: 1
        background: Rectangle { color: Theme.card; border.color: Theme.border; border.width: 1; radius: Theme.radiusControl }
        contentItem: ListView {
            clip: true
            implicitHeight: contentHeight
            model: cb.popup.visible ? cb.delegateModel : null
            currentIndex: cb.highlightedIndex
            ScrollIndicator.vertical: ScrollIndicator {}
        }
    }
}
