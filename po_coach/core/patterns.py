"""The trader's seven named setups, as machine-checkable detectors.

From the blueprint: "ABC break/rejection, Doji break, Quasimodo, third-touch,
breakout-retest, rejection wick, engulfing."

Two rules apply to every detector here:

1. **Causality.** A detector may only use information available at the bar it
   fires on. Swings are used only once `confirmed_at` has passed. A 1-minute
   backtest is unforgiving of one bar of lookahead, and this is the single
   easiest way to manufacture a fake edge.
2. **Direction is a claim, not a description.** Each detector returns
   'call' / 'put' / None plus the reasons, so the orchestrator can show the
   trader *why* rather than just a number.

Performance: the detectors run over every bar of every asset on every 5-minute
close, and the blueprint budgets 20 seconds end to end. Row-wise `df.iloc`
access on a DataFrame costs ~2 orders of magnitude more than indexing a numpy
array, so all the hot paths here read `BarView` numpy buffers instead. Logic is
identical to the straightforward pandas version; the array backend exists only
to make the scan fast enough to be usable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .structure import Swing


@dataclass
class Setup:
    name: str
    direction: Optional[str]      # "call" | "put" | None
    strength: float               # 0..1, this detector's own conviction
    bar: int
    time: pd.Timestamp
    reasons: List[str] = field(default_factory=list)
    context: Dict[str, object] = field(default_factory=dict)
    base_score: float = 0.0       # 0..40, the trader's-rules component

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "time": str(self.time),
            "bar": self.bar,
            "reasons": self.reasons,
            "context": self.context,
            "base_score": round(self.base_score, 1),
        }


class BarView:
    """Read-only numpy buffers over an OHLC frame, plus a rolling ATR column.

    Constructed once per scan. Every accessor is O(1) and returns numpy scalars
    or views, so a detector never touches the DataFrame.
    """

    __slots__ = ("o", "h", "l", "c", "atr", "index", "n")

    def __init__(self, df: pd.DataFrame):
        self.o = np.ascontiguousarray(df["open"].to_numpy(dtype=float))
        self.h = np.ascontiguousarray(df["high"].to_numpy(dtype=float))
        self.l = np.ascontiguousarray(df["low"].to_numpy(dtype=float))
        self.c = np.ascontiguousarray(df["close"].to_numpy(dtype=float))
        if "atr" in df.columns:
            self.atr = np.ascontiguousarray(df["atr"].to_numpy(dtype=float))
        else:
            from .indicators import atr

            self.atr = np.ascontiguousarray(atr(df, 14).to_numpy(dtype=float))
        self.index = df.index
        self.n = len(df)

    def atr_at(self, i: int, fallback: float = 1e-9) -> float:
        if 0 <= i < self.n:
            v = self.atr[i]
            if np.isfinite(v) and v > 0:
                return float(v)
        return max(abs(self.c[min(max(i, 0), self.n - 1)]) * 1e-4, fallback)

    def rng(self, i: int) -> float:
        return float(self.h[i] - self.l[i])


def _known_swings(swings: Sequence[Swing], i: int, window: int = 16) -> List[Swing]:
    """Swings confirmable at bar `i`, most recent last, windowed.

    The window matters: no detector looks further back than ~12 swings, and
    slicing keeps this O(window) instead of O(all swings) per bar.
    """
    out: List[Swing] = []
    for s in swings:
        if s.confirmed_at <= i:
            out.append(s)
    return out[-window:]


def _range_pct(bv: BarView, i: int, bars: int) -> float:
    a = max(0, i - bars + 1)
    lo = float(np.min(bv.l[a: i + 1]))
    hi = float(np.max(bv.h[a: i + 1]))
    return (hi - lo) / max(hi, 1e-12) * 100.0


# --------------------------------------------------------------------------
# 1. Engulfing
# --------------------------------------------------------------------------

def engulfing(bv: BarView, i: int, min_body_frac: float = 0.55) -> Optional[Setup]:
    if i < 2:
        return None
    o1, h1, l1, c1 = bv.o[i - 1], bv.h[i - 1], bv.l[i - 1], bv.c[i - 1]
    o2, h2, l2, c2 = bv.o[i], bv.h[i], bv.l[i], bv.c[i]
    r1, r2 = h1 - l1, h2 - l2
    if r1 <= 0 or r2 <= 0:
        return None
    b1 = abs(c1 - o1) / r1
    b2 = abs(c2 - o2) / r2
    if b1 < 0.3 or b2 < min_body_frac:
        return None

    bullish = c2 > o2 and c1 < o1
    bearish = c2 < o2 and c1 > o1
    if not (bullish or bearish):
        return None
    engulfs = (c2 >= o1 and o2 <= c1) if bullish else (c2 <= o1 and o2 >= c1)
    if not engulfs:
        return None

    a = bv.atr_at(i)
    direction = "call" if bullish else "put"
    strength = min(1.0, b2 * 0.6 + min(1.0, r2 / max(a, 1e-12) / 2.0) * 0.4)
    return Setup(
        name="engulfing",
        direction=direction,
        strength=strength,
        bar=i,
        time=bv.index[i],
        reasons=["%s candle engulfs the prior body (%.0f%% of range)." % ("Bullish" if bullish else "Bearish", b2 * 100)],
        base_score=30.0 + strength * 10.0,
    )


# --------------------------------------------------------------------------
# 2. Doji break
# --------------------------------------------------------------------------

def doji_break(bv: BarView, i: int, doji_body: float = 0.10) -> Optional[Setup]:
    if i < 2:
        return None
    d_o, d_h, d_l, d_c = bv.o[i - 1], bv.h[i - 1], bv.l[i - 1], bv.c[i - 1]
    r = d_h - d_l
    if r <= 0:
        return None
    if abs(d_c - d_o) / r > doji_body:
        return None
    a = bv.atr_at(i)
    n_c = bv.c[i]
    if n_c > d_h + 0.05 * a:
        direction, strength = "call", min(1.0, (n_c - d_h) / max(a, 1e-12))
        why = "closes above the doji high"
    elif n_c < d_l - 0.05 * a:
        direction, strength = "put", min(1.0, (d_l - n_c) / max(a, 1e-12))
        why = "closes below the doji low"
    else:
        return None
    return Setup(
        name="doji_break",
        direction=direction,
        strength=strength,
        bar=i,
        time=bv.index[i],
        reasons=["Doji indecision at %.5f then %s." % (float(d_c), why)],
        base_score=28.0 + strength * 10.0,
    )


# --------------------------------------------------------------------------
# 3. Quasimodo
# --------------------------------------------------------------------------

def quasimodo(
    bv: BarView, swings: Sequence[Swing], i: int, overshoot_atr: float = 0.1
) -> Optional[Setup]:
    """Classic QM: an extended high beyond the prior swing, a reversal leg that
    breaks structure, and a retest of the extreme that fails."""
    known = _known_swings(swings, i)
    highs = [s for s in known if s.kind == "high"][-3:]
    if len(highs) < 2:
        return None
    a = bv.atr_at(i)

    peak = highs[-1]
    peak_px = float(peak.price)
    lo = bv.l[peak.index: i + 1]
    hi = bv.h[peak.index: i + 1]
    cl = bv.c[peak.index: i + 1]
    if len(cl) < 4:
        return None
    if float(np.min(cl)) >= peak_px - 0.15 * a:
        return None
    retest = (hi > peak_px - overshoot_atr * a) & (hi <= peak_px)
    if not retest.any():
        return None
    last_c = bv.c[i]
    if last_c >= peak_px - 0.2 * a:
        return None
    strength = min(1.0, (peak_px - last_c) / max(a, 1e-12) / 2.0)
    return Setup(
        name="quasimodo",
        direction="put",
        strength=strength,
        bar=i,
        time=bv.index[i],
        reasons=["Quasimodo: rejected %.5f after breaking structure below it" % peak_px],
        context={"peak": peak_px, "peak_time": str(peak.time)},
        base_score=32.0 + strength * 8.0,
    )


# --------------------------------------------------------------------------
# 4. Third touch
# --------------------------------------------------------------------------

def third_touch(
    bv: BarView,
    swings: Sequence[Swing],
    i: int,
    min_touches: int = 3,
    max_distance_atr: float = 0.15,
) -> Optional[Setup]:
    """Third test of a horizontal level -- the level usually gives way."""
    known = _known_swings(swings, i)
    if not known:
        return None
    a = bv.atr_at(i)
    tol = max_distance_atr * a
    last_swing = known[-1]
    price = float(last_swing.price)

    same_side = [s for s in known[-12:] if abs(s.price - price) <= tol]
    if len(same_side) < min_touches:
        return None

    level = float(np.mean([s.price for s in same_side]))
    close_now = bv.c[i]
    if close_now > level and last_swing.kind == "high":
        direction, why = "put", "rejected at resistance after %d touches" % len(same_side)
    elif close_now < level and last_swing.kind == "low":
        direction, why = "call", "held support after %d touches" % len(same_side)
    else:
        return None
    strength = min(1.0, 0.4 + 0.15 * (len(same_side) - min_touches + 1))
    return Setup(
        name="third_touch",
        direction=direction,
        strength=strength,
        bar=i,
        time=bv.index[i],
        reasons=["Third touch of %.5f: %s." % (level, why)],
        context={"level": level, "touches": len(same_side)},
        base_score=30.0 + strength * 8.0,
    )


# --------------------------------------------------------------------------
# 5. Breakout + retest
# --------------------------------------------------------------------------

def breakout_retest(
    bv: BarView, swings: Sequence[Swing], i: int, max_bars: int = 25, min_breaks_atr: float = 0.15
) -> Optional[Setup]:
    """A level broken by a decisive candle, then retested and held."""
    known = _known_swings(swings, i)
    if not known:
        return None
    a = bv.atr_at(i)
    close_now = bv.c[i]
    for brk in reversed(known[-6:]):
        if i - brk.confirmed_at > max_bars:
            continue
        level = float(brk.price)
        a_i = max(brk.confirmed_at, 0)
        hi = bv.h[a_i: i + 1]
        lo = bv.l[a_i: i + 1]
        if hi.size < 3:
            continue
        if brk.kind == "high":
            if float(np.max(hi)) <= level + min_breaks_atr * a:
                continue
            if not (close_now > level):
                continue
            touched = (lo <= level) & (hi >= level)
            direction = "call"
        else:
            if float(np.min(lo)) >= level - min_breaks_atr * a:
                continue
            if not (close_now < level):
                continue
            touched = (hi >= level) & (lo <= level)
            direction = "put"
        if not touched.any():
            continue
        strength = min(1.0, 0.5 + 0.1 * int(touched.sum()))
        return Setup(
            name="breakout_retest",
            direction=direction,
            strength=strength,
            bar=i,
            time=bv.index[i],
            reasons=["Break of %.5f held on retest." % level],
            context={"level": level, "break_bar": int(brk.index)},
            base_score=30.0 + strength * 10.0,
        )
    return None


# --------------------------------------------------------------------------
# 6. Rejection wick
# --------------------------------------------------------------------------

def rejection_wick(
    bv: BarView, i: int, min_wick: float = 0.55, min_body: float = 0.2
) -> Optional[Setup]:
    """Long wick, small body: price was pushed and rejected at that level."""
    o, h, l, c = bv.o[i], bv.h[i], bv.l[i], bv.c[i]
    r = h - l
    if r <= 0:
        return None
    rng = _range_pct(bv, i, 20)
    if not np.isfinite(rng) or rng <= 0:
        return None
    body = abs(c - o) / r
    body_top = max(o, c)
    body_bot = min(o, c)
    upper = (h - body_top) / r
    lower = (body_bot - l) / r

    if upper >= min_wick and body <= min_body:
        direction, strength, why = "put", min(1.0, upper), "upper wick %.0f%% of range" % (upper * 100)
    elif lower >= min_wick and body <= min_body:
        direction, strength, why = "call", min(1.0, lower), "lower wick %.0f%% of range" % (lower * 100)
    else:
        return None
    return Setup(
        name="rejection_wick",
        direction=direction,
        strength=strength,
        bar=i,
        time=bv.index[i],
        reasons=["Rejection: %s (range %.2f%%)." % (why, rng)],
        base_score=26.0 + strength * 10.0,
    )


# --------------------------------------------------------------------------
# 7. ABC break / rejection
# --------------------------------------------------------------------------

def abc_pattern(
    bv: BarView, swings: Sequence[Swing], i: int, ratio_lo: float = 0.382, ratio_hi: float = 0.886
) -> Optional[Setup]:
    """Three-leg correction (A-B-C) where BC retraces a Fibonacci band of AB.

    Direction is the direction of the AB impulse, not of the C leg.
    """
    known = _known_swings(swings, i)
    if len(known) < 3:
        return None
    A, B, C = known[-3:]
    if A.kind == B.kind:
        return None

    ab = B.price - A.price
    bc = C.price - B.price
    if ab == 0:
        return None
    retr = abs(bc / ab)
    if not (ratio_lo <= retr <= ratio_hi):
        return None

    if ab < 0:  # AB leg was down -> bullish ABC (buy the dip)
        if C.kind != "low":
            return None
        direction = "call"
    else:
        if C.kind != "high":
            return None
        direction = "put"

    strength = max(0.2, min(1.0, 1.0 - abs(retr - 0.618) / 0.5))
    return Setup(
        name="abc_rejection",
        direction=direction,
        strength=strength,
        bar=i,
        time=bv.index[i],
        reasons=["ABC %s: BC retraced %.1f%% of AB (ideal zone 38.2-88.6%%)."
                 % ("up" if direction == "call" else "down", retr * 100)],
        context={"A": A.price, "B": B.price, "C": C.price, "retracement": retr},
        base_score=30.0 + strength * 8.0,
    )


# --------------------------------------------------------------------------

SWING_DETECTORS = (quasimodo, third_touch, breakout_retest, abc_pattern)
SIMPLE_DETECTORS = (doji_break, engulfing, rejection_wick)


def detect_all(
    bv: BarView, i: int, swings: Sequence[Swing]
) -> List[Setup]:
    """Run every detector at bar `i`. Multiple hits are normal and good --
    confluence is what the scoring rewards."""
    out: List[Setup] = []
    for fn in SIMPLE_DETECTORS:
        try:
            s = fn(bv, i)
        except (IndexError, KeyError, TypeError, ValueError):
            s = None
        if s is not None:
            out.append(s)
    for fn in SWING_DETECTORS:
        try:
            s = fn(bv, swings, i)
        except (IndexError, KeyError, TypeError, ValueError):
            s = None
        if s is not None:
            out.append(s)
    return out


def scan(
    df: pd.DataFrame,
    warmup: int = 60,
    swings: Optional[List[Swing]] = None,
    only: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Scan every bar for setups.

    Swings are pushed in incrementally as their `confirmed_at` passes, so the
    per-bar cost is O(window) rather than O(all swings) -- the difference
    between a scan that finishes in a second and one that takes four minutes.
    """
    bv = BarView(df)
    if swings is None:
        from .structure import find_swings

        swings = find_swings(df, 2, 2)

    order = sorted(range(len(swings)), key=lambda k: swings[k].confirmed_at)
    ptr = 0
    known: List[Swing] = []
    n = len(bv.index)
    start = max(warmup, 1)

    rows: List[dict] = []
    for i in range(start, n):
        while ptr < len(order) and swings[order[ptr]].confirmed_at <= i:
            known.append(swings[order[ptr]])
            ptr += 1
        if len(known) > 32:
            known = known[-32:]
        for s in detect_all(bv, i, known):
            if only and s.name not in only:
                continue
            rows.append(
                {
                    "bar": i,
                    "time": bv.index[i],
                    "name": s.name,
                    "direction": s.direction,
                    "strength": s.strength,
                    "base_score": s.base_score,
                    "reasons": " | ".join(s.reasons),
                }
            )

    cols = ["bar", "time", "name", "direction", "strength", "base_score", "reasons"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).reset_index(drop=True)


def detect_latest(
    df: pd.DataFrame, swings: Optional[List[Swing]] = None
) -> List[Setup]:
    """Setups firing on the most recent bar only -- the live-scanner path."""
    bv = BarView(df)
    i = bv.n - 1
    known = _known_swings(swings or [], i, window=32)
    return detect_all(bv, i, known)
