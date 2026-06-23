import QtQuick
import Stonks

// Configure a new backtest.
Page {
    id: view
    maxWidth: 1080
    topPad: 36
    spacing: 20

    readonly property var strategyOptions: ["Momentum Breakout v3", "Mean Reversion (Bollinger)", "Dual MA Crossover", "Opening Range Breakout"]
    readonly property var dataKeys: ["us_megacaps_1h", "us_equities_15m", "sp500_daily", "crypto_majors_1h"]
    readonly property var paramMeta: [
        { k: "lookback", label: "Lookback", unit: "bars" },
        { k: "breakout", label: "Breakout", unit: "%" },
        { k: "stop", label: "Stop loss", unit: "%" },
        { k: "take", label: "Take profit", unit: "%" },
        { k: "size", label: "Position size", unit: "% eq" },
        { k: "maxpos", label: "Max positions", unit: "" },
        { k: "cooldown", label: "Cooldown", unit: "bars" },
        { k: "atr", label: "ATR period", unit: "bars" }
    ]
    function dataLabel(key) { return (App.dataFilesObj()[key] || {}).label || key }

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
                    model: view.strategyOptions
                    currentIndex: view.strategyOptions.indexOf(App.strategy)
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
                    model: view.dataKeys.map(view.dataLabel)
                    values: view.dataKeys
                    currentIndex: view.dataKeys.indexOf(App.dataFile)
                    onPicked: function (v) { App.dataFile = v }
                }
                Row {
                    width: parent.width
                    spacing: 24
                    Column {
                        spacing: 4
                        width: (parent.width - 24) / 2
                        Text { text: "SYMBOLS"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10 }
                        Text { width: parent.width; elide: Text.ElideRight; text: (App.dataFilesObj()[App.dataFile] || {}).symbols || ""; color: Theme.t2; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12 }
                    }
                    Column {
                        spacing: 4
                        width: (parent.width - 24) / 2
                        Text { text: "SOURCE"; color: Theme.t6; font.family: Theme.mono; font.pixelSize: 10 }
                        Text { width: parent.width; elide: Text.ElideRight; text: ((App.dataFilesObj()[App.dataFile] || {}).source || "") + " · " + ((App.dataFilesObj()[App.dataFile] || {}).bars || "") + " bars"; color: Theme.t2; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 12 }
                    }
                }
            }
        }
    }

    // --- strategy parameters ---
    Rectangle {
        width: view.innerWidth
        radius: 6
        color: Theme.card
        border.color: Theme.border
        border.width: 1
        implicitHeight: paramCol.implicitHeight + 40

        Column {
            id: paramCol
            x: 22; y: 20
            width: parent.width - 44
            spacing: 16
            Text { text: "STRATEGY PARAMETERS"; color: Theme.t5; font.family: Theme.mono; font.weight: Font.Medium; font.pixelSize: 11; font.letterSpacing: 1 }
            Grid {
                width: parent.width
                columns: 4
                rowSpacing: 16
                columnSpacing: 18
                Repeater {
                    model: view.paramMeta
                    delegate: ParamField {
                        required property var modelData
                        width: (paramCol.width - 18 * 3) / 4
                        label: modelData.label
                        unit: modelData.unit
                        value: String(App.params[modelData.k])
                        onEdited: function (v) { App.setParam(modelData.k, v) }
                    }
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
