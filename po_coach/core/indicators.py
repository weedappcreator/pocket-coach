"""Technical indicators, implemented on numpy/pandas only.

Deliberately no TA-Lib: it needs a C library, and the blueprint's trader uses a
specific, small set of tools (RSI, EMA, Stochastic, Fractals, ATR). Owning the
formulas means the scoring weights and the backtester agree exactly, and there
is no version drift between what is charted and what is tested.

Every function takes and returns a pandas Series/array aligned to the input
index, with warm-up periods left as NaN rather than back-filled.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing -- the average RSI and ATR are actually built on."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return rma(true_range(df), n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = rma(gain, n)
    avg_loss = rma(loss, n)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # All-gain windows have zero average loss -> RSI 100 by definition.
    out[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    out[(avg_gain == 0) & (avg_loss > 0)] = 0.0
    return out


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Slow stochastic %K / %D, the settings most retail charts ship with."""
    low_n = df["low"].rolling(k_period, min_periods=k_period).min()
    high_n = df["high"].rolling(k_period, min_periods=k_period).max()
    rng = (high_n - low_n).replace(0.0, np.nan)
    raw_k = 100.0 * (df["close"] - low_n) / rng
    k = raw_k.rolling(smooth, min_periods=smooth).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(s, n)
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return mid - k * sd, mid, mid + k * sd


def ema_slope(s: pd.Series, n: int = 5) -> pd.Series:
    """Signed slope of the EMA, normalised to % per bar so assets are comparable."""
    e = ema(s, n)
    return (e.diff() / e.replace(0.0, np.nan)) * 100.0


def williams_r(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high_n = df["high"].rolling(n, min_periods=n).max()
    low_n = df["low"].rolling(n, min_periods=n).min()
    rng = (high_n - low_n).replace(0.0, np.nan)
    return -100.0 * (high_n - df["close"]) / rng


def momentum(s: pd.Series, n: int) -> pd.Series:
    return s.pct_change(n) * 100.0


def realized_vol(s: pd.Series, n: int = 20, annualize: int = 0) -> pd.Series:
    """Rolling standard deviation of log returns.

    `annualize` is the number of bars per year (0 = leave per-bar). This is the
    input the expiry-probability model needs.
    """
    lr = np.log(s / s.shift(1))
    v = lr.rolling(n, min_periods=n).std(ddof=0)
    if annualize:
        v = v * np.sqrt(annualize)
    return v


def body_pct(df: pd.DataFrame) -> pd.Series:
    """Candle body as a fraction of total range -- 'strength' of the candle."""
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    return (df["close"] - df["open"]).abs() / rng


def wick_ratio(df: pd.DataFrame, side: str = "upper") -> pd.Series:
    body_top = df[["open", "close"]].max(axis=1)
    body_bot = df[["open", "close"]].min(axis=1)
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    if side == "upper":
        return (df["high"] - body_top) / rng
    return (body_bot - df["low"]) / rng


def is_doji(df: pd.DataFrame, body_frac: float = 0.10) -> pd.Series:
    return body_pct(df) <= body_frac


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average Directional Index -- used here only to classify 'trending' vs
    'choppy', which is a hard no-trade condition in the blueprint."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(df), n)
    atr_ = tr.replace(0.0, np.nan)
    plus_di = 100.0 * rma(pd.Series(plus_dm, index=df.index), n) / atr_
    minus_di = 100.0 * rma(pd.Series(minus_dm, index=df.index), n) / atr_
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return rma(dx, n)


def chop_index(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Choppiness index: 100 = pure chop, 0 = pure trend.

    Above ~61.8 is conventionally 'no trend'; this is the mechanical version of
    the trader's "skip choppy markets" rule.
    """
    atr_ = atr(df, n)
    prev_close = df["close"].shift(1)
    hl_range = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    ratio = atr_.rolling(n, min_periods=n).sum() / hl_range.rolling(n, min_periods=n).sum().replace(0.0, np.nan)
    val = 100.0 * np.log10(ratio) / np.log10(n)
    return val.clip(0.0, 100.0)


def keltner_width(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """ATR-normalised channel width -- a volatility-regime gate that works
    across assets with wildly different price scales."""
    mid = ema(df["close"], n)
    a = atr(df, n)
    return (4.0 * a) / mid.replace(0.0, np.nan) * 100.0


def supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """SuperTrend direction, used as a tie-breaker when timeframes disagree."""
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a
    close = df["close"]
    direction = pd.Series(np.where(close > upper, 1, -1), index=df.index, dtype=float)
    for i in range(1, len(df)):
        if pd.isna(a.iloc[i]):
            direction.iloc[i] = np.nan
        elif close.iloc[i] > upper.iloc[i - 1]:
            direction.iloc[i] = 1.0
        elif close.iloc[i] < lower.iloc[i - 1]:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = direction.iloc[i - 1]
    return pd.DataFrame({"direction": direction, "upper": upper, "lower": lower})


def compute_features(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    rsi_n: int = 14,
    atr_n: int = 14,
    bb_n: int = 20,
    stoch_n: int = 14,
) -> pd.DataFrame:
    """One-shot feature bundle. This is what the rule engine and the ML model
    both consume, so they can never disagree about what the chart looks like."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], ema_fast)
    out["ema_slow"] = ema(out["close"], ema_slow)
    out["ema_slope"] = ema_slope(out["close"], ema_slow)
    out["rsi"] = rsi(out["close"], rsi_n)
    out["atr"] = atr(out, atr_n)
    out["atr_pct"] = out["atr"] / out["close"] * 100.0
    bb_lo, bb_mid, bb_hi = bollinger(out["close"], bb_n)
    out["bb_low"], out["bb_mid"], out["bb_high"] = bb_lo, bb_mid, bb_hi
    out["bb_width"] = (bb_hi - bb_lo) / bb_mid.replace(0.0, np.nan) * 100.0
    out["bb_pctb"] = (out["close"] - bb_lo) / (bb_hi - bb_lo).replace(0.0, np.nan)
    k, d = stochastic(out, stoch_n)
    out["stoch_k"], out["stoch_d"] = k, d
    out["adx"] = adx(out, atr_n)
    out["chop"] = chop_index(out, atr_n)
    out["keltner_width"] = keltner_width(out)
    out["st_direction"] = supertrend(out)["direction"]
    out["ret_1"] = out["close"].pct_change(1) * 100.0
    out["ret_3"] = out["close"].pct_change(3) * 100.0
    out["ret_10"] = out["close"].pct_change(10) * 100.0
    out["body_pct"] = body_pct(out)
    out["upper_wick"] = wick_ratio(out, "upper")
    out["lower_wick"] = wick_ratio(out, "lower")
    out["vol_20"] = realized_vol(out["close"], 20)
    out["vol_ratio"] = out["vol_20"] / out["vol_20"].rolling(100, min_periods=40).median()
    out["hour"] = pd.to_datetime(out.index).hour
    out["dow"] = pd.to_datetime(out.index).dayofweek
    return out
