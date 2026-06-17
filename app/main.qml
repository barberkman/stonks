import QtQuick
import QtQuick.Controls
import Stonks

ApplicationWindow {
    visible: true
    width: 400
    height: 300
    title: "stonks"

    EngineRunner { id: runner }

    Button {
        anchors.fill: parent
        text: "Run"
        onClicked: runner.run()
    }
}
