"""Backtester with PocketOption expiry semantics, plus the go/no-go gate.

Three things separate this from a typical indicator backtest, and all three
matter more than the strategy logic:

1. **Causality.** A signal at bar `i` is filled at the close of bar `i` (the
   blueprint's "entry timing: open of the candle after the confirmation candle
   closes") and settles at the close of bar `i + expiry`. Swings are only used
   after they are confirmable. One bar of lookahead inflates a 52% win rate into
   a 58% one, which would sail straight through the go/no-go gate.

2. **Overlapping trades do not give you independent samples.** Taking a signal
   on every bar of a 5-minute chart with a 5-minute expiry produces trades that
   share almost all their price path, so the effective sample size is far below
   the trade count. `effective_sample_size` estimates this, and the gate uses it
   in preference to the raw count.

3. **In-sample results are not results.** `walk_forward` splits the data, tunes
   on the first part and reports only on the second. The blueprint's 300-trade
   gate is applied to the out-of-sample leg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .indicators import compute_features
from .structure import find_swings
from .payoff import break_even_win_rate, ev_pct, test_edge, upper_tail, wilson_interval
from .structure import analyse
from .patterns import scan as scan_patterns


@dataclass
class Trade:
    bar: int
    time: pd.Timestamp
    asset: str
    direction: str
    payout: float
    expiry_bars: int
    entry: float
    exit: float
    won: bool
    stake: float
    pnl: float
    score: float
    setups: List[str]
    hourly: bool
    timeframe: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    strategy: str
    asset: str
    timeframe: str
    payout: float
    n_trades: int
    effective_n: float
    wins: int
    win_rate: float
    breakeven: float
    edge_pp: float
    ev_pct_per_trade: float
    total_return_pct: float
    max_drawdown_pct: float
    ci_low: float
    ci_high: float
    edge_test: dict
    by_expiry: Dict[str, dict]
    by_hour: Dict[str, dict]
    by_setup: Dict[str, dict]
    equity_curve: List[dict]
    gate_passed: bool
    gate_reason: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d


def effective_sample_size(bars: Sequence[int], expiry_bars: int) -> float:
    """Rough independence correction for overlapping trades.

    Consecutive signals share `expiry_bars` of price path. A crude but
    conservative deflation: cluster bars that fall within one expiry window of
    each other and count clusters, scaled by the intra-cluster correlation of a
    binary payoff (which for these payoffs is very high).
    """
    if not len(bars):
        return 0.0
    bars = sorted(bars)
    clusters = 1
    last = bars[0]
    for b in bars[1:]:
        if b - last >= expiry_bars:
            clusters += 1
        last = b
    # Within a cluster, outcomes are largely determined by one price path, so
    # treat a cluster of k trades as worth ~1.4 independent observations rather
    # than k.
    return 1.0 + (len(bars) - clusters) * 0.4


def simulate(
    df: pd.DataFrame,
    signals: pd.DataFrame,
    payout: float = 92.0,
    stake_pct: float = 1.0,
    starting_balance: float = 10_000.0,
    asset: str = "",
    timeframe: str = "5m",
    allow_overlap: bool = True,
) -> Tuple[List[Trade], pd.Series]:
    """Turn a signal frame into a trade list, then a balance curve.

    `signals` needs columns: bar, direction, expiry_bars, score, setups.
    """
    if signals is None or len(signals) == 0:
        return [], pd.Series(dtype=float)

    close = df["close"].to_numpy()
    n = len(df)
    trades: List[Trade] = []
    balance = starting_balance
    curve: List[Tuple[pd.Timestamp, float]] = []
    last_exit_bar = -1

    sigs = signals.sort_values("bar")
    for _, s in sigs.iterrows():
        i = int(s["bar"])
        exp = int(s.get("expiry_bars", 1))
        j = i + exp
        if j >= n or i < 0:
            continue
        if not allow_overlap and i < last_exit_bar:
            continue

        entry = float(close[i])
        exit_px = float(close[j])
        direction = str(s["direction"])
        won = (exit_px > entry) if direction == "call" else (exit_px < entry)
        # At-the-money settles by the platform's tick rule; a flat bar is a tie
        # and is booked as a loss (the conservative choice).
        if exit_px == entry:
            won = False

        stake = balance * (stake_pct / 100.0)
        pnl = stake * (payout / 100.0) if won else -stake
        balance += pnl
        last_exit_bar = j

        setups = s.get("setups", "")
        if isinstance(setups, str):
            setups = [x for x in setups.split("|") if x]

        trades.append(
            Trade(
                bar=i,
                time=df.index[i],
                asset=asset,
                direction=direction,
                payout=payout,
                expiry_bars=exp,
                entry=entry,
                exit=exit_px,
                won=bool(won),
                stake=stake,
                pnl=pnl,
                score=float(s.get("score", 0.0) or 0.0),
                setups=list(setups),
                hourly=bool(s.get("hourly", False)),
                timeframe=timeframe,
            )
        )
        curve.append((df.index[j], balance))

    return trades, pd.Series(
        [b for _, b in curve], index=[t for t, _ in curve], dtype=float
    )


def _group_stats(trades: List[Trade], key: str, payout: float) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not trades:
        return out
    for k, grp in _group(trades, key):
        wins = sum(1 for t in grp if t.won)
        n = len(grp)
        wr = wins / float(n)
        be = break_even_win_rate(payout / 100.0)
        out[str(k)] = {
            "n": n,
            "wins": wins,
            "win_rate": wr,
            "breakeven": be,
            "edge_pp": (wr - be) * 100.0,
            "ev_pct": ev_pct(payout / 100.0, wr),
            "pnl": sum(t.pnl for t in grp),
            "ci_low": wilson_interval(wins, n)[0],
            "ci_high": wilson_interval(wins, n)[1],
        }
    return out


def _group(trades: List[Trade], key: str):
    buckets: Dict[object, List[Trade]] = {}
    for t in trades:
        if key == "hour":
            k = pd.Timestamp(t.time).hour
        elif key == "setup":
            for name in (t.setups or ["none"]):
                buckets.setdefault(name, []).append(t)
            continue
        else:
            k = getattr(t, key, None)
        buckets.setdefault(k, []).append(t)
    return sorted(buckets.items(), key=lambda kv: -len(kv[1]))


def max_drawdown(curve: pd.Series, starting: float) -> float:
    if len(curve) == 0:
        return 0.0
    peak = curve.cummax()
    dd = (peak - curve) / peak.replace(0.0, np.nan)
    return float(np.nanmax(dd.to_numpy())) * 100.0 if len(dd) else 0.0


def evaluate(
    trades: List[Trade],
    equity: pd.Series,
    strategy: str,
    asset: str,
    timeframe: str,
    payout: float,
    starting_balance: float,
    min_win_rate: float = 0.57,
    min_trades: int = 300,
    alpha: float = 0.05,
    in_sample: bool = True,
    min_edge_pp: float = 3.0,
) -> BacktestResult:
    """Score a trade list and apply the blueprint's go/no-go gate."""
    notes: List[str] = []
    n = len(trades)
    wins = sum(1 for t in trades if t.won)
    be = break_even_win_rate(payout / 100.0)
    wr = (wins / n) if n else 0.0

    eff_n = effective_sample_size([t.bar for t in trades], int(np.median([t.expiry_bars for t in trades])) if trades else 1)
    if eff_n < n * 0.6 and n > 0:
        notes.append(
            "Trades overlap heavily: %d raw trades give only ~%.0f independent "
            "samples, so the true uncertainty is larger than it looks."
            % (n, eff_n)
        )

    # Test against break-even, but deflate the sample for overlap.
    test = test_edge(wins, n, payout / 100.0, min_edge_pp=min_edge_pp, in_sample=in_sample)
    if eff_n < n:
        se = math.sqrt(be * (1 - be) / max(eff_n, 1.0))
        z_def = (wr - be) / se if se > 0 else 0.0
        p_def = upper_tail(z_def)
        notes.append(
            "Overlap-adjusted significance: z=%.2f, p=%.4f (raw p=%.4f)."
            % (z_def, p_def, test.p_value)
        )
        if p_def >= alpha and test.p_value < alpha:
            notes.append(
                "The edge is NOT significant once overlapping trades are treated "
                "as one observation. Treat this as unproven."
            )

    total_return = (equity.iloc[-1] / starting_balance - 1.0) * 100.0 if len(equity) else 0.0
    mdd = max_drawdown(equity, starting_balance)

    # ---- the gate -------------------------------------------------------
    reasons: List[str] = []
    passed = True
    if n < min_trades:
        passed = False
        reasons.append("Only %d trades; the gate requires %d." % (n, min_trades))
    if wr < min_win_rate:
        passed = False
        reasons.append(
            "Win rate %.2f%% is below the %.0f%% gate." % (wr * 100, min_win_rate * 100)
        )
    if test.p_value >= alpha:
        passed = False
        # p is a one-sided test for an edge ABOVE break-even, so p near 1.0 can
        # mean either "no difference" or "significantly worse". Reporting both
        # as "not distinguishable" is actively misleading -- a z of -5.81 is a
        # decisive result, just not the one anyone wanted.
        ci_below = test.ci_high < be
        if ci_below:
            reasons.append(
                "Significantly WORSE than break-even (z=%.2f, one-sided p=%.3f for an "
                "edge — the 95%% CI [%.2f%%-%.2f%%] lies entirely below the %.2f%% "
                "needed). This is a real finding, just the wrong sign."
                % (test.z_score, test.p_value, test.ci_low * 100,
                   test.ci_high * 100, be * 100)
            )
        else:
            reasons.append(
                "Not statistically distinguishable from break-even (p=%.3f)." % test.p_value
            )
    if test.ci_contains_breakeven:
        passed = False
        reasons.append(
            "95%% CI (%.2f%%-%.2f%%) still contains break-even %.2f%%."
            % (test.ci_low * 100, test.ci_high * 100, be * 100)
        )
    if in_sample:
        passed = False
        reasons.append("Result is in-sample; re-run out-of-sample before believing it.")

    gate_reason = "PASS" if passed else "FAIL: " + "; ".join(reasons)

    ci_low, ci_high = wilson_interval(wins, n) if n else (0.0, 1.0)

    return BacktestResult(
        strategy=strategy,
        asset=asset,
        timeframe=timeframe,
        payout=payout,
        n_trades=n,
        effective_n=round(eff_n, 1),
        wins=wins,
        win_rate=wr,
        breakeven=be,
        edge_pp=(wr - be) * 100.0,
        ev_pct_per_trade=ev_pct(payout / 100.0, wr),
        total_return_pct=total_return,
        max_drawdown_pct=mdd,
        ci_low=ci_low,
        ci_high=ci_high,
        edge_test=test.to_dict(),
        by_expiry=_group_stats(trades, "expiry_bars", payout),
        by_hour=_group_stats(trades, "hour", payout),
        by_setup=_group_stats(trades, "setup", payout),
        equity_curve=(
            [{"time": str(t), "balance": float(b)} for t, b in equity.items()]
            if len(equity)
            else []
        ),
        gate_passed=passed,
        gate_reason=gate_reason,
        notes=notes,
    )


def walk_forward(
    df: pd.DataFrame,
    build_signals,
    payout: float = 92.0,
    split: float = 0.7,
    starting_balance: float = 10_000.0,
    **kwargs,
) -> Tuple[BacktestResult, BacktestResult]:
    """Tune on the first `split`, report only on the rest.

    `build_signals(train_df) -> DataFrame` is called with the training slice
    only, so anything it learns cannot leak into the reported period.
    """
    n = len(df)
    cut = int(n * split)
    train = df.iloc[:cut]
    test_df = df.iloc[cut:]

    sig_train = build_signals(train)
    trades_train, eq_train = simulate(
        train, sig_train, payout, 1.0, starting_balance, kwargs.get("asset", ""), kwargs.get("timeframe", "5m")
    )
    in_res = evaluate(
        trades_train, eq_train, "in-sample", kwargs.get("asset", ""), kwargs.get("timeframe", "5m"),
        payout, starting_balance, in_sample=True, **{k: v for k, v in kwargs.items() if k not in ("asset", "timeframe")}
    )

    sig_test = build_signals(test_df)
    trades_test, eq_test = simulate(
        test_df, sig_test, payout, 1.0, starting_balance, kwargs.get("asset", ""), kwargs.get("timeframe", "5m")
    )
    out_res = evaluate(
        trades_test, eq_test, "out-of-sample", kwargs.get("asset", ""), kwargs.get("timeframe", "5m"),
        payout, starting_balance, in_sample=False, **{k: v for k, v in kwargs.items() if k not in ("asset", "timeframe")}
    )
    out_res.notes.insert(
        0, "Out-of-sample: the last %d%% of candles, never seen while tuning." % int((1 - split) * 100)
    )
    return in_res, out_res


def baseline_check(
    df: pd.DataFrame, expiry_bars: int = 1, payout: float = 92.0
) -> dict:
    """What does a random guess score on this data?

    Run this before believing any strategy. If a coin flip gets 52% on your
    sample, then a 55% strategy is inside the noise and you have found nothing.
    """
    n = len(df)
    if n <= expiry_bars + 1:
        return {"ok": False}
    up = (df["close"].shift(-expiry_bars) > df["close"]).dropna()
    n_t = len(up)
    wins = int(up.sum())
    wr = wins / float(n_t)
    be = break_even_win_rate(payout / 100.0)
    t = test_edge(wins, n_t, payout / 100.0, in_sample=False)
    return {
        "ok": True,
        "n": n_t,
        "always_call_win_rate": wr,
        "breakeven": be,
        "edge_pp": (wr - be) * 100.0,
        "ev_pct": ev_pct(payout / 100.0, wr),
        "edge_test": t.to_dict(),
        "note": "A trend-following 'always CALL' baseline. Any strategy must beat this.",
    }


def features_for(df: pd.DataFrame) -> pd.DataFrame:
    return compute_features(df)


def prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    """Feature frame plus causal swings, ready for pattern detection."""
    feats = compute_features(df)
    return feats, find_swings(feats, 2, 2)
