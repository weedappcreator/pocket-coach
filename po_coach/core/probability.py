"""Turning a chart read into an actual probability of winning.

A binary trade is a bet that the price will be above (or below) a strike at
expiry. Given spot, strike, remaining time and volatility, that probability is
a closed-form function. Comparing it to the break-even win rate implied by the
payout is the whole game: the tool's job is not to predict direction, it is to
find the handful of setups where our probability estimate is honestly higher
than what the payout is paying for.

Two honesty rules are baked in:

* The drift is estimated, not assumed to be helpful. Assuming a trend will
  "continue" with a positive drift at the 1-5 minute horizon is precisely how
  you manufacture a fake 60% win rate, so drift is shrunk hard toward zero and
  its contribution is reported separately from the volatility term.
* Volatility is what actually determines the probability. A 92% payout on a
  dead market can be a much better bet than an 80% payout on a volatile one,
  because the payout is fixed but the vol is not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .payoff import break_even_win_rate, ev_pct


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_arr(x: np.ndarray) -> np.ndarray:
    from scipy.special import ndtr

    return ndtr(x)


BARS_PER_YEAR = {
    "1m": 24 * 60 * 365,
    # 2m bars are 12 per hour (not 30); 5m bars are 12 per hour. Using the
    # wrong divisor here rescales every implied probability on that timeframe.
    "2m": 24 * 12 * 365,
    "5m": 24 * 12 * 365,
    "15m": 24 * 4 * 365,
    "30m": 24 * 2 * 365,
    "1h": 24 * 365,
    "4h": 6 * 365,
    "1d": 252,
}


def bars_to_years(bars: int, timeframe: str) -> float:
    per_year = BARS_PER_YEAR.get(timeframe, BARS_PER_YEAR["5m"])
    return float(bars) / float(per_year)


def expiry_probability(
    spot: float,
    strike: float,
    sigma_annual: float,
    t_years: float,
    drift_annual: float = 0.0,
) -> float:
    """P(S_T > strike) under a lognormal with given vol and drift.

    `sigma_annual` and `drift_annual` are annualised; `t_years` is the time to
    expiry. For a 1-minute expiry and typical FX vol this sits within a few
    percent of 0.5, which is the honest answer.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if t_years <= 0:
        return 1.0 if spot > strike else 0.0
    vol_t = sigma_annual * math.sqrt(t_years)
    if vol_t <= 1e-12:
        return 1.0 if spot > strike else 0.0
    d = (math.log(spot / strike) + (drift_annual - 0.5 * sigma_annual ** 2) * t_years) / vol_t
    return _phi(d)


def atm_coinflip_probability(
    sigma_annual: float, t_years: float, drift_annual: float = 0.0
) -> float:
    """Win probability for an at-the-money trade.

    This is the number that should humble anyone. At 5 minutes with 8% annualised
    vol, an ATM trade wins about 50.2% of the time -- and a 92% payout needs
    52.08%. So an ATM trade with no directional view is, structurally, a losing
    bet, no matter how confident the trader feels.
    """
    return expiry_probability(1.0, 1.0, sigma_annual, t_years, drift_annual)


def strike_distance_pct(spot: float, strike: float) -> float:
    return (strike - spot) / spot * 100.0


def vol_from_atr(atr_value: float, price: float, timeframe: str = "5m") -> float:
    """Convert an ATR reading into an annualised volatility estimate.

    Rough, but it is the right order of magnitude and it adapts per asset,
    which matters more than precision here. True range is ~1.3x the per-bar
    sigma for most instruments.
    """
    if price <= 0 or atr_value <= 0:
        return 0.0
    per_bar_sigma = (atr_value / price) / 1.3
    per_year = BARS_PER_YEAR.get(timeframe, BARS_PER_YEAR["5m"])
    return per_bar_sigma * math.sqrt(per_year)


@dataclass
class EdgeEstimate:
    """Full picture for one candidate trade."""

    spot: float
    strike: float
    direction: str
    payout: float
    timeframe: str
    expiry_bars: int
    sigma_annual: float
    t_years: float
    p_market: float            # vol-implied, no directional opinion
    p_technical: float         # after the technical tilt
    drift_applied: float
    breakeven: float
    edge_pp_market: float
    edge_pp_technical: float
    ev_pct_market: float
    ev_pct_technical: float
    verdict: str
    notes: Tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d


def estimate_edge(
    spot: float,
    payout: float,
    direction: str,
    sigma_annual: float,
    expiry_bars: int,
    timeframe: str = "5m",
    technical_bias: float = 0.0,
    strike: Optional[float] = None,
    drift_annual: float = 0.0,
    drift_shrink: float = 0.15,
    min_edge_pp: float = 3.0,
) -> EdgeEstimate:
    """Full edge evaluation for one trade.

    `technical_bias` in -1..+1 is the net directional opinion from the scoring
    model. It is converted into a *tiny* drift adjustment on purpose: over a
    1-5 minute horizon, a real directional edge is worth a fraction of a
    percent, and pretending otherwise is how backtests lie.

    The market probability is reported alongside the technical one so the trader
    can see exactly how much of the claimed edge came from the chart and how
    much from volatility alone.
    """
    k = spot if strike is None else float(strike)
    t_years = bars_to_years(expiry_bars, timeframe)
    be = break_even_win_rate(payout)

    p_market = expiry_probability(spot, k, sigma_annual, t_years, 0.0)

    # Convert bias into an annualised drift, shrunk hard.
    per_bar_ret = sigma_annual / math.sqrt(BARS_PER_YEAR.get(timeframe, BARS_PER_YEAR["5m"]))
    raw_drift = technical_bias * per_bar_ret * 12.0
    applied = raw_drift * drift_shrink
    p_tech_raw = expiry_probability(spot, k, sigma_annual, t_years, applied)
    p_technical = p_tech_raw if direction == "call" else 1.0 - p_tech_raw
    p_market = p_market if direction == "call" else 1.0 - p_market

    edge_m = (p_market - be) * 100.0
    edge_t = (p_technical - be) * 100.0
    ev_m = ev_pct(payout, p_market)
    ev_t = ev_pct(payout, p_technical)

    notes = []
    if abs(technical_bias) < 0.15:
        notes.append("No meaningful directional bias -- this is a coin flip with a fee.")
    if t_years < 1.0 / (BARS_PER_YEAR.get(timeframe, 252) * 6):
        notes.append("Expiry is very short; strike distance dominates the outcome.")
    if sigma_annual <= 0.02:
        notes.append("Volatility is very low (%0.2f%% annualised) -- treat p as ~50%%." % (sigma_annual * 100))

    if edge_t < min_edge_pp:
        verdict = "REJECT"
        notes.append(
            "Edge %+.2f pp is under the %.0f pp floor. This is a negative-EV trade."
            % (edge_t, min_edge_pp)
        )
    elif edge_t < min_edge_pp * 2:
        verdict = "MARGINAL"
        notes.append("Edge %+.2f pp is thin. Size down or skip." % edge_t)
    else:
        verdict = "TRADEABLE"
        notes.append("Edge %+.2f pp over the %.2f%% break-even." % (edge_t, be * 100))

    if abs(edge_t - edge_m) < 0.25 and abs(edge_m) >= min_edge_pp:
        notes.append("The edge comes from volatility/strike, not from the chart read.")

    return EdgeEstimate(
        spot=spot,
        strike=k,
        direction=direction,
        payout=payout,
        timeframe=timeframe,
        expiry_bars=expiry_bars,
        sigma_annual=sigma_annual,
        t_years=t_years,
        p_market=p_market,
        p_technical=p_technical,
        drift_applied=applied,
        breakeven=be,
        edge_pp_market=edge_m,
        edge_pp_technical=edge_t,
        ev_pct_market=ev_m,
        ev_pct_technical=ev_t,
        verdict=verdict,
        notes=tuple(notes),
    )


def implied_payout_probability(payout: float, breakeven: Optional[float] = None) -> float:
    """What win probability the payout is paying for -- i.e. the broker's own
    implied estimate. If your model says 54% and the payout implies 52.1%, the
    payout is the market's price for that 52.1%."""
    return breakeven if breakeven is not None else break_even_win_rate(payout)


def sigma_from_candles(df: pd.DataFrame, n: int = 200, timeframe: str = "5m") -> float:
    """Annualised sigma from log returns, for the timeframe of `df`.

    Uses an EWMA (RiskMetrics lambda=0.94) blended with a long simple window
    rather than a short rolling window alone. A 20-bar rolling std is noisy
    enough to swing the implied edge by several points, and it trends low on
    quiet stretches -- which is exactly when a trader is most eager to trade.

    `timeframe` must match the bars actually in `df`. It previously defaulted to
    the 1-minute constant regardless of input, which inflated every 5-minute
    volatility estimate by sqrt(5) ~= 2.24x: EUR/USD was reported at 17%/yr
    when the real figure was nearer 7.6%, and an inflated sigma makes short
    expiries look far more likely to finish in the money.
    """
    if len(df) < 30:
        return 0.0
    close = df["close"].dropna()
    lr = np.log(close / close.shift(1)).dropna()
    if len(lr) < 20:
        return 0.0

    simple = float(lr.iloc[-n:].std(ddof=1))
    ewma = float(lr.ewm(span=60).mean().std(ddof=1))
    # Take the more conservative (higher) of the two: underestimating vol makes
    # every trade look better than it is.
    per_bar = max(simple, ewma * 0.98) if simple > 0 else ewma
    per_year = BARS_PER_YEAR.get(timeframe, BARS_PER_YEAR["5m"])
    return per_bar * math.sqrt(per_year)


def vol_plausibility(sigma_annual: float, asset_class: str = "fx") -> dict:
    """Sanity-band a sigma estimate so a broken feed cannot fake an edge.

    Typical annualised vols: majors FX 6-12%, crypto 40-120%, gold 12-25%,
    index CFDs 12-30%. Outside the band the number is not trustworthy, and any
    edge computed from it should be treated as unavailable.
    """
    bands = {
        "fx": (0.04, 0.20),
        "fx_otc": (0.02, 0.35),
        "crypto": (0.25, 1.50),
        "metals": (0.08, 0.40),
        "index": (0.06, 0.45),
        "unknown": (0.01, 2.00),
    }
    lo, hi = bands.get(asset_class, bands["unknown"])
    if not (lo <= sigma_annual <= hi):
        return {
            "ok": False,
            "sigma_annual": sigma_annual,
            "expected_range": [lo, hi],
            "asset_class": asset_class,
            "warning": (
                "Implied volatility %.1f%% is outside the plausible %.0f-%.0f%% band "
                "for %s. Check the data feed before trusting any edge estimate."
                % (sigma_annual * 100, lo * 100, hi * 100, asset_class)
            ),
        }
    return {"ok": True, "sigma_annual": sigma_annual, "expected_range": [lo, hi], "asset_class": asset_class}


def probability_ladder(
    spot: float, payout: float, sigma_annual: float, expiry_bars: int,
    timeframe: str = "5m", direction: str = "call", biases: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Probability and edge across a grid of technical biases.

    Useful in the dashboard: it shows how much directional conviction you need
    before a given payout becomes worth taking, which is a much more actionable
    question than "what is the win rate".
    """
    b = biases if biases is not None else np.linspace(-1.0, 1.0, 41)
    rows = []
    for bias in b:
        e = estimate_edge(
            spot, payout, direction, sigma_annual, expiry_bars, timeframe, float(bias)
        )
        rows.append(
            {
                "bias": float(bias),
                "p_win": e.p_technical,
                "breakeven": e.breakeven,
                "edge_pp": e.edge_pp_technical,
                "ev_pct": e.ev_pct_technical,
                "verdict": e.verdict,
            }
        )
    return pd.DataFrame(rows)


def required_bias_for_payout(
    spot: float, payout: float, sigma_annual: float, expiry_bars: int,
    timeframe: str = "5m", direction: str = "call", min_edge_pp: float = 3.0
) -> Optional[float]:
    """Smallest directional bias that clears the payout's break-even.

    If the answer is near 1.0, the trade needs a call you do not have, and the
    correct action is to skip it. This is the single most useful output of the
    module for a 1-5 minute trader.
    """
    target = break_even_win_rate(payout) + min_edge_pp / 100.0
    ladder = probability_ladder(spot, payout, sigma_annual, expiry_bars, timeframe, direction)
    ok = ladder[ladder["p_win"] >= target]
    if len(ok) == 0:
        return None
    if direction == "call":
        return float(ok["bias"].min())
    return float(ok["bias"].max())
