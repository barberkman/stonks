.pragma library

// Pure view-window math for CandleChart. Everything works in fractional
// candle-index space ([lo, hi] reals over 0..count-1); nothing touches QML
// state, so each function is independently checkable.

// Largest allowed span shows the whole series; smallest is minSpan candles.
function clampSpan(span, count, minSpan) {
    var maxSpan = Math.max(minSpan, count - 1);
    return Math.max(minSpan, Math.min(maxSpan, span));
}

// Clamp a window against the data edges, allowing a small over-pan margin
// (5% of the span) so the first/last candle is not glued to the border. When
// the span covers the whole series the view centres and panning is a no-op.
function clampPan(lo, hi, count, minSpan) {
    var last = count - 1;
    var span = clampSpan(hi - lo, count, minSpan);
    if (span >= last) {
        var mid = last / 2;
        return { lo: mid - span / 2, hi: mid + span / 2 };
    }
    var margin = Math.max(1, span * 0.05);
    var lo2 = lo, hi2 = lo + span;
    if (lo2 < -margin) { lo2 = -margin; hi2 = lo2 + span; }
    if (hi2 > last + margin) { hi2 = last + margin; lo2 = hi2 - span; }
    return { lo: lo2, hi: hi2 };
}

// Zoom by `factor` keeping the candle under the cursor pixel-fixed.
// anchorFrac is the cursor's horizontal position as a fraction of the plot
// width; the index under it keeps the same fraction of the new span to its
// left, so it stays put on screen.
function zoomedRange(lo, hi, anchorFrac, factor, count, minSpan) {
    var frac = Math.max(0, Math.min(1, anchorFrac));
    var span = hi - lo;
    var anchorIdx = lo + frac * span;
    var newSpan = clampSpan(span * factor, count, minSpan);
    var newLo = anchorIdx - frac * newSpan;
    return clampPan(newLo, newLo + newSpan, count, minSpan);
}

// Slice the visible window out of the full-resolution columns. When less than
// ~2px falls on each candle, aggregate one OHLCV bucket per ~2px column
// instead (o = first open, h/l = extremes, c = last close, v = summed volume)
// — the standard resample, so buckets draw through the same candle code.
// Returns parallel arrays plus each drawn candle's raw fractional index (xi)
// for the shared X() mapping, and the visible price/volume extremes.
function visibleSeries(T, O, H, L, C, V, lo, hi, plotW) {
    var count = T.length;
    var i0 = Math.max(0, Math.floor(lo)), i1 = Math.min(count - 1, Math.ceil(hi));
    var out = { t: [], o: [], h: [], l: [], c: [], v: [], xi: [], pmin: Infinity, pmax: -Infinity, vmax: 0 };
    if (i1 < i0) return out;
    var n = i1 - i0 + 1;
    var pxPerCandle = plotW / Math.max(1e-9, hi - lo);
    if (pxPerCandle >= 2) {
        for (var k = i0; k <= i1; k++) {
            out.t.push(T[k]); out.o.push(O[k]); out.h.push(H[k]); out.l.push(L[k]); out.c.push(C[k]); out.v.push(V[k]);
            out.xi.push(k);
            if (L[k] < out.pmin) out.pmin = L[k];
            if (H[k] > out.pmax) out.pmax = H[k];
            if (V[k] > out.vmax) out.vmax = V[k];
        }
        return out;
    }
    var buckets = Math.max(1, Math.floor(plotW / 2));
    var perBucket = n / buckets;
    for (var b = 0; b < buckets; b++) {
        var b0 = i0 + Math.floor(b * perBucket);
        var b1 = Math.min(i1 + 1, i0 + Math.floor((b + 1) * perBucket));
        if (b1 <= b0) continue;
        var bh = -Infinity, bl = Infinity, bv = 0;
        for (var j = b0; j < b1; j++) {
            if (H[j] > bh) bh = H[j];
            if (L[j] < bl) bl = L[j];
            bv += V[j];
        }
        out.t.push(T[b0]); out.o.push(O[b0]); out.h.push(bh); out.l.push(bl); out.c.push(C[b1 - 1]); out.v.push(bv);
        out.xi.push((b0 + b1 - 1) / 2);
        if (bl < out.pmin) out.pmin = bl;
        if (bh > out.pmax) out.pmax = bh;
        if (bv > out.vmax) out.vmax = bv;
    }
    return out;
}
