.pragma library

// Formatting helpers ported from the design's Component (fmtUsd / dShort / dFull / hexA).

function hexA(hex, a) {
    var n = hex.replace('#', '');
    return 'rgba(' + parseInt(n.slice(0, 2), 16) + ',' + parseInt(n.slice(2, 4), 16)
            + ',' + parseInt(n.slice(4, 6), 16) + ',' + a + ')';
}

// rgba() string from a QML color object (.r/.g/.b in 0..1) — for canvas gradient stops.
function rgbaColor(c, a) {
    return 'rgba(' + Math.round(c.r * 255) + ',' + Math.round(c.g * 255) + ',' + Math.round(c.b * 255) + ',' + a + ')';
}

function commas(n) {
    var s = String(n);
    var out = '';
    var c = 0;
    for (var i = s.length - 1; i >= 0; i--) {
        out = s[i] + out;
        if (++c % 3 === 0 && i > 0) out = ',' + out;
    }
    return out;
}

function fmtUsd(n) {
    var sign = n < 0 ? '-' : '+';
    return sign + '$' + commas(Math.abs(Math.round(n)));
}

function dateAt(i) {
    var base = Date.UTC(2024, 0, 2, 14, 30);
    return new Date(base + Math.round(i * 1.35 * 86400000));
}

function dShort(i) {
    var d = dateAt(i);
    var mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return mon[d.getUTCMonth()] + ' ' + d.getUTCDate();
}

function dFull(i) {
    var d = dateAt(i);
    var hh = String(9 + (i % 6));
    if (hh.length < 2) hh = '0' + hh;
    return dShort(i) + " '" + String(d.getUTCFullYear()).slice(2) + '  ' + hh + ':30';
}

// Real-timestamp variants (epoch ms) used once the engine supplies real bar/trade times.
var _mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function tsShort(ms) {
    var d = new Date(ms);
    return _mon[d.getUTCMonth()] + ' ' + d.getUTCDate();
}

function tsFull(ms) {
    var d = new Date(ms);
    var hh = String(d.getUTCHours()); if (hh.length < 2) hh = '0' + hh;
    var mm = String(d.getUTCMinutes()); if (mm.length < 2) mm = '0' + mm;
    return tsShort(ms) + " '" + String(d.getUTCFullYear()).slice(2) + '  ' + hh + ':' + mm;
}
