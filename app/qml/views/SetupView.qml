import QtQuick
import Stonks

// Configure a new backtest: pick a Python strategy + data file, filter it
// (symbols + date range), set starting cash, and run.
Page {
    id: view
    maxWidth: 1080
    topPad: 36
    spacing: 20

    readonly property var strats: App.strategyList()    // [{display, module, cls}]
    readonly property var files: App.dataFileList()      // [{key, label, source}]
    readonly property var stratModules: strats.map(function (s) { return s.module })
    readonly property var fileKeys: files.map(function (f) { return f.key })

    Component.onCompleted: {
        if (strats.length && !App.strategy) { App.strategy = strats[0].module; }
        if (files.length && !App.dataFile) { App.setDataFile(files[0].key); }
    }

    // --- back + title ---
    Row {
        spacing: 14
        GhostButton { anchors.verticalCenter: parent.verticalCenter; text: "‹ Backtests"; onClicked: App.goBacktests() }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            text: "CONFIGURE RUN"
            color: Theme.accent
            font.family: Theme.mono
            font.weight: Font.Medium
            font.pixelSize: 11
            font.letterSpacing: 1.6
        }
    }
    Text {
        text: "New backtest"
        color: Theme.textBright
        font.family: Theme.sans
        font.weight: Font.Bold
        font.pixelSize: 24
    }

    // --- strategy + data cards ---
    Row {
        width: view.innerWidth
        spacing: 20
        readonly property real cardW: (width - 20) / 2

        // strategy
        Rectangle {
            width: parent.cardW
            height: 158
            radius: 6
            color: Theme.card
            border.color: Theme.border
            border.width: 1
            Column {
                x: 22; y: 20
                width: parent.width - 44
                spacing: 14
                Text { text: "STRATEGY"; color: Theme.t5; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 11; font.letterSpacing: 1 }
                StyledSelect {
                    width: parent.width
                    model: view.strats.map(function (s) { return s.display })
                    values: view.stratModules
                    currentIndex: Math.max(0, view.stratModules.indexOf(App.strategy))
                    onPicked: function (v) { App.strategy = v }
                }
                Column {
                    spacing: 4
                    Text { text: "SOURCE"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10 }
                    Text { text: App.strategySource(); color: Theme.t2; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12 }
                }
            }
        }

        // data
        Rectangle {
            width: parent.cardW
            height: 158
            radius: 6
            color: Theme.card
            border.color: Theme.border
            border.width: 1
            Column {
                x: 22; y: 20
                width: parent.width - 44
                spacing: 12
                Text { text: "DATA"; color: Theme.t5; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 11; font.letterSpacing: 1 }
                StyledSelect {
                    width: parent.width
                    model: view.files.map(function (f) { return f.label })
                    values: view.fileKeys
                    currentIndex: Math.max(0, view.fileKeys.indexOf(App.dataFile))
                    onPicked: function (v) { App.setDataFile(v) }
                }
                Row {
                    width: parent.width
                    spacing: 24
                    Column {
                        spacing: 4
                        width: (parent.width - 24) / 2
                        Text { text: "SYMBOLS"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10 }
                        Text { width: parent.width; elide: Text.ElideRight; text: App.symbols.length + " of " + App.availableSymbols.length + " selected"; color: Theme.t2; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12 }
                    }
                    Column {
                        spacing: 4
                        width: (parent.width - 24) / 2
                        Text { text: "SOURCE"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10 }
                        Text { width: parent.width; elide: Text.ElideRight; text: (App.dataFilesObj()[App.dataFile] || {}).source || ""; color: Theme.t2; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12 }
                    }
                }
            }
        }
    }

    // --- filter: symbols + date range ---
    Rectangle {
        width: view.innerWidth
        radius: 6
        color: Theme.card
        border.color: Theme.border
        border.width: 1
        implicitHeight: filterCol.implicitHeight + 40

        Column {
            id: filterCol
            x: 22; y: 20
            width: parent.width - 44
            spacing: 16

            Text { text: "FILTER"; color: Theme.t5; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 11; font.letterSpacing: 1 }

            // symbol multi-select pills
            Column {
                width: parent.width
                spacing: 8
                Text { text: "SYMBOLS"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10; font.letterSpacing: 0.6 }
                Flow {
                    width: parent.width
                    spacing: 8
                    Repeater {
                        model: App.availableSymbols
                        delegate: Rectangle {
                            required property var modelData
                            readonly property bool on: App.symbols.indexOf(modelData) >= 0
                            width: pillT.implicitWidth + 28
                            height: 30
                            radius: 5
                            color: on ? Theme.accent : "transparent"
                            border.width: 1
                            border.color: on ? Theme.accent : Theme.border
                            Text {
                                id: pillT
                                anchors.centerIn: parent
                                text: modelData
                                color: parent.on ? Theme.accentInk : Theme.t3
                                font.family: Theme.mono
                                font.weight: Font.DemiBold
                                font.pixelSize: 12
                            }
                            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: App.toggleSymbol(modelData) }
                        }
                    }
                }
            }

            // date range
            Row {
                width: parent.width
                spacing: 18
                ParamField {
                    width: (parent.width - 18) / 2
                    label: "Start date (YYYY-MM-DD)"
                    unit: ""
                    value: App.startDate
                    onEdited: function (v) { App.startDate = v }
                }
                ParamField {
                    width: (parent.width - 18) / 2
                    label: "End date (YYYY-MM-DD)"
                    unit: ""
                    value: App.endDate
                    onEdited: function (v) { App.endDate = v }
                }
            }
        }
    }

    // --- starting cash + start ---
    Item {
        width: view.innerWidth
        height: 44
        Row {
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            spacing: 11
            Text { anchors.verticalCenter: parent.verticalCenter; text: "Starting cash"; color: Theme.t4; font.family: Theme.sans; font.pixelSize: 12 }
            Rectangle {
                anchors.verticalCenter: parent.verticalCenter
                width: 160
                height: 40
                radius: 5
                color: Theme.input
                border.color: "#3c3c3c"
                border.width: 1
                Text { id: dollar; anchors.left: parent.left; anchors.leftMargin: 11; anchors.verticalCenter: parent.verticalCenter; text: "$"; color: Theme.t5; font.family: Theme.mono; font.pixelSize: 13 }
                TextInput {
                    anchors.left: dollar.right
                    anchors.leftMargin: 4
                    anchors.right: parent.right
                    anchors.rightMargin: 8
                    anchors.verticalCenter: parent.verticalCenter
                    text: App.startCash
                    color: Theme.textPrimary
                    font.family: Theme.mono
                    font.pixelSize: 14
                    selectByMouse: true
                    clip: true
                    onEditingFinished: App.startCash = text
                }
            }
            Text {
                anchors.verticalCenter: parent.verticalCenter
                visible: App.runError !== ""
                text: "⚠ " + App.runError
                color: Theme.negative
                font.family: Theme.mono
                font.pixelSize: 12
                width: 360
                elide: Text.ElideRight
            }
        }
        AccentButton {
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            text: "▸ Start backtest"
            hpad: 26
            fontPx: 14
            onClicked: App.runBacktest()
        }
    }
}
