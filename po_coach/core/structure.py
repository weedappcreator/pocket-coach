"""Market structure: swings, trend classification, support/resistance zones.

The trader's method is explicitly price-action first ("market structure HH/HL,
LH/LL plus the 50 EMA on 5-min; S/R zones from 5-min+ with 2+ reactions; order
blocks and trendlines"). So this module turns candles into the vocabulary his
rules are written in, and every threshold is a named constant that the YAML
config can override.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Swing:
    index: int
    time: pd.Timestamp
    price: float
    kind: str  # "high" or "low"
    confirmed_at: int = -1  # bar index at which this swing became knowable


@dataclass
class Zone:
    """A support/resistance band, not a line.

    Real levels have width, and 'distance to zone' is one of the scoring
    components, so zones are stored as (low, high, strength, last_reaction).
    """
    low: float
    high: float
    kind: str            # "support" | "resistance" | "order_block"
    touches: int = 1
    last_touch: int = -1
    strength: float = 0.0
    label: str = ""


@dataclass
class StructureRead:
    trend: str = "unclear"           # up | down | range | unclear
    bias: float = 0.0                # -1..+1
    last_high: Optional[Swing] = None
    last_low: Optional[Swing] = None
    swings: List[Swing] = field(default_factory=list)
    zones: List[Zone] = field(default_factory=list)
    ema_above: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trend": self.trend,
            "bias": self.bias,
            "ema_above": self.ema_above,
            "last_high": None if self.last_high is None else {
                "price": self.last_high.price, "time": str(self.last_high.time)
            },
            "last_low": None if self.last_low is None else {
                "price": self.last_low.price, "time": str(self.last_low.time)
            },
            "n_swings": len(self.swings),
            "n_zones": len(self.zones),
            "zones": [
                {
                    "low": z.low, "high": z.high, "kind": z.kind,
                    "touches": z.touches, "strength": z.strength, "label": z.label,
                }
                for z in self.zones
            ],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------

def find_swings(
    df: pd.DataFrame, left: int = 2, right: int = 2
) -> List[Swing]:
    """Fractal-style swing points.

    A pivot high needs `right` bars after it to exist, so `confirmed_at` is
    what matters for backtesting: acting on a swing before it was knowable is
    lookahead bias, and at 1-5 minute expiries it would inflate every backtest
    result we produce.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df.index
    n = len(df)
    out: List[Swing] = []

    for i in range(left, n - right):
        h = highs[i]
        win_h = highs[i - left: i + right + 1]
        if np.isnan(h) or len(win_h) == 0:
            continue
        if h >= win_h.max() and (win_h == h).sum() == 1:
            out.append(Swing(i, times[i], float(h), "high", i + right))
        l = lows[i]
        win_l = lows[i - left: i + right + 1]
        if np.isnan(l) or len(win_l) == 0:
            continue
        if l <= win_l.min() and (win_l == l).sum() == 1:
            out.append(Swing(i, times[i], float(l), "low", i + right))

    out.sort(key=lambda s: s.index)
    return out


def classify_trend(
    swings: List[Swing], lookback: int = 4
) -> Tuple[str, float, List[str]]:
    """Label HH/HL (up), LH/LL (down) or range, using the last `lookback` swings."""
    notes: List[str] = []
    highs = [s for s in swings if s.kind == "high"][-lookback:]
    lows = [s for s in swings if s.kind == "low"][-lookback:]

    if len(highs) < 2 or len(lows) < 2:
        return "unclear", 0.0, ["Not enough confirmed swings to judge structure."]

    hh = sum(1 for a, b in zip(highs, highs[1:]) if b.price > a.price)
    lh = len(highs) - 1 - hh
    hl = sum(1 for a, b in zip(lows, lows[1:]) if b.price > a.price)
    ll = len(lows) - 1 - hl

    total = hh + lh + hl + ll
    bias = ((hh + hl) - (lh + ll)) / float(total) if total else 0.0

    if hh >= 2 and hl >= 2 and lh == 0:
        trend, notes = "up", ["Clean HH/HL sequence."]
    elif lh >= 2 and ll >= 2 and hh == 0:
        trend, notes = "down", ["Clean LH/LL sequence."]
    elif abs(bias) < 0.25:
        trend = "range"
        notes.append("Swing highs and lows are interleaved -- no clean structure.")
    else:
        trend = "up" if bias > 0 else "down"
        notes.append("Mixed structure: %d HH, %d HL, %d LH, %d LL." % (hh, hl, lh, ll))

    return trend, bias, notes


def build_zones(
    df: pd.DataFrame,
    swings: List[Swing],
    tolerance_atr: float = 0.35,
    min_touches: int = 2,
    max_zones: int = 8,
) -> List[Zone]:
    """Cluster swing highs and lows into S/R bands by ATR-relative tolerance.

    ATR-relative rather than fixed-percent so the same code works on EUR/USD and
    on Bitcoin without per-asset tuning.
    """
    if not len(df):
        return []
    from .indicators import atr

    a = atr(df, 14)
    last_atr = float(a.iloc[-1]) if not np.isnan(a.iloc[-1]) else float(df["close"].iloc[-1]) * 0.001
    tol = last_atr * tolerance_atr
    if not np.isfinite(tol) or tol <= 0:
        tol = float(df["close"].iloc[-1]) * 1e-4

    raw: List[Tuple[float, int, str]] = []
    for s in swings:
        raw.append((s.price, s.index, "resistance" if s.kind == "high" else "support"))
    raw.sort(key=lambda t: t[0])

    clusters: List[List[Tuple[float, int, str]]] = []
    for price, idx, kind in raw:
        if clusters and abs(price - clusters[-1][-1][0]) <= tol:
            clusters[-1].append((price, idx, kind))
        else:
            clusters.append([(price, idx, kind)])

    zones: List[Zone] = []
    n = len(df)
    for cl in clusters:
        prices = [p for p, _, _ in cl]
        lo, hi = min(prices), max(prices)
        last_idx = max(i for _, i, _ in cl)
        age_bars = n - 1 - last_idx
        recency = 1.0 / (1.0 + age_bars / 50.0)
        strength = len(cl) * recency
        kind = cl[0][2]
        zones.append(
            Zone(
                low=lo - tol * 0.5,
                high=hi + tol * 0.5,
                kind=kind,
                touches=len(cl),
                last_touch=last_idx,
                strength=strength,
                label="%s x%d" % (kind, len(cl)),
            )
        )

    zones = [z for z in zones if z.touches >= min_touches]
    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones[:max_zones]


def find_order_blocks(
    df: pd.DataFrame, swings: List[Swing], displacement_atr: float = 1.0
) -> List[Zone]:
    """Order blocks: the last opposing candle before a strong displacement leg."""
    from .indicators import atr

    if not len(df) or not swings:
        return []
    a = atr(df, 14)
    n = len(df)
    out: List[Zone] = []
    for s in swings:
        if s.index < 3 or s.confirmed_at >= n - 1:
            continue
        a_now = a.iloc[s.confirmed_at]
        if np.isnan(a_now) or a_now <= 0:
            continue
        # Walk back from the swing to the last candle in the opposite colour.
        for j in range(s.index - 1, max(s.index - 8, 0), -1):
            row = df.iloc[j]
            if (row["close"] < row["open"] and s.kind == "high") or (
                row["close"] > row["open"] and s.kind == "low"
            ):
                leg = abs(df["close"].iloc[s.index] - df["open"].iloc[j])
                bar_body = abs(row["close"] - row["open"]) / max(row["high"] - row["low"], 1e-12)
                if leg >= displacement_atr * float(a_now) and bar_body > 0.5:
                    kind = "supply" if s.kind == "high" else "demand"
                    out.append(
                        Zone(
                            low=float(min(row["open"], row["close"])),
                            high=float(max(row["open"], row["close"])),
                            kind=kind,
                            touches=1,
                            last_touch=j,
                            strength=leg / float(a_now),
                            label="order block (%s)" % kind,
                        )
                    )
                break
    out.sort(key=lambda z: z.last_touch, reverse=True)
    return out[:5]


def nearest_zone(zones: List[Zone], price: float) -> Optional[Zone]:
    if not zones:
        return None
    return min(zones, key=lambda z: 0.0 if z.low <= price <= z.high else min(abs(z.low - price), abs(z.high - price)))


def zone_distance_pct(zones: List[Zone], price: float) -> float:
    """Distance from price to the nearest zone edge, in percent.

    The blueprint scores 'room to move before expiry'; a price sitting inside a
    zone has nowhere to go, which is why this returns 0 there.
    """
    z = nearest_zone(zones, price)
    if z is None:
        return float("inf")
    if z.low <= price <= z.high:
        return 0.0
    d = (z.low - price) if price < z.low else (price - z.high)
    return abs(d) / max(abs(price), 1e-12) * 100.0


def zone_room_pct(zones: List[Zone], price: float, direction: str) -> float:
    """Clear space in the trade's direction, as a percent of price.

    `zone_distance_pct` answers "how far am I from the nearest level in *any*
    direction", which is the wrong question for a directional trade. Sitting in
    the middle of a wide support zone gives it 0.0 -- "no room" -- when in fact
    there is a great deal of room downward.

    On a 5-minute FX chart the current price is inside some recent swing zone
    almost all of the time, so the undirected measure returned 0.0 for
    effectively every bar and the blueprint's room-to-move filter vetoed 100% of
    setups. The scanner could never fire, regardless of market conditions.

    So: measure the distance to the next obstacle *ahead* of the trade, and when
    price sits inside a zone, use the far edge -- the side the trade is heading
    toward.
    """
    if not zones or not np.isfinite(price) or price <= 0:
        return float("inf")
    if direction == "call":
        # nearest resistance above, or the top of the zone we are sitting in
        tops = [z.high for z in zones if z.high > price]
        room = min(tops) - price if tops else float("inf")
    else:
        # nearest support below, or the bottom of the zone we are sitting in
        bots = [z.low for z in zones if z.low < price]
        room = price - max(bots) if bots else float("inf")
    if not np.isfinite(room):
        return float("inf")
    return max(0.0, room) / price * 100.0


def analyse(df: pd.DataFrame, swing_left: int = 2, swing_right: int = 2) -> StructureRead:
    """Full structure read on a featured DataFrame (needs `ema_slow` present)."""
    read = StructureRead()
    if not len(df):
        return read

    swings = find_swings(df, swing_left, swing_right)
    read.swings = swings
    read.last_high = next((s for s in reversed(swings) if s.kind == "high"), None)
    read.last_low = next((s for s in reversed(swings) if s.kind == "low"), None)

    trend, bias, notes = classify_trend(swings)
    read.trend = trend
    read.bias = bias
    read.notes.extend(notes)

    read.zones = build_zones(df, swings)
    read.zones.extend(find_order_blocks(df, swings))

    if "ema_slow" in df.columns and not np.isnan(df["ema_slow"].iloc[-1]):
        read.ema_above = bool(df["close"].iloc[-1] > df["ema_slow"].iloc[-1])
        read.notes.append(
            "Price is %s the 50 EMA." % ("above" if read.ema_above else "below")
        )

    if read.ema_above is not None:
        ema_bias = 1.0 if read.ema_above else -1.0
        if (trend == "up" and ema_bias < 0) or (trend == "down" and ema_bias > 0):
            read.notes.append("Structure and the 50 EMA disagree -- reduced conviction.")
            read.bias = bias * 0.5

    return read


def timeframe_alignment(
    trends: Dict[str, StructureRead], weights: Optional[Dict[str, float]] = None
) -> Tuple[float, bool]:
    """Weighted agreement across timeframes.

    Returns (score in -1..1, all_agreeing). The blueprint's 'timeframe
    alignment' component is worth 20% of the setup score, and disagreement
    between 5/10/15-min is an automatic no-trade.
    """
    if not trends:
        return 0.0, True
    w = weights or {tf: 1.0 for tf in trends}
    total_w = 0.0
    acc = 0.0
    signs = []
    for tf, read in trends.items():
        if read.trend == "up":
            s = 1.0
        elif read.trend == "down":
            s = -1.0
        else:
            s = 0.0
        acc += s * w.get(tf, 1.0)
        total_w += w.get(tf, 1.0)
        signs.append(s)
    score = acc / total_w if total_w else 0.0
    directional = [s for s in signs if s != 0.0]
    all_agree = len(directional) > 0 and all(s == directional[0] for s in directional)
    return score, all_agree


def draw_trendline(swings: List[Swing], kind: str) -> Optional[Tuple[float, float]]:
    """Least-squares line through the last 3 same-kind swings -> (m, b)."""
    pts = [s for s in swings if s.kind == kind][-3:]
    if len(pts) < 2:
        return None
    xs = np.array([s.index for s in pts], dtype=float)
    ys = np.array([s.price for s in pts], dtype=float)
    if np.ptp(xs) == 0:
        return None
    m, b = np.polyfit(xs, ys, 1)
    return float(m), float(b)
