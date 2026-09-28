"""Trade journal and the analytics that make it worth keeping.

The journal's job is to answer one question the platform never will: **is this
trader actually profitable, and if not, which of their rules is costing them
money?**

PocketOption's own history view reports win rate. Win rate alone is a trap --
it says nothing about payout, and a 50% win rate at 92% payout is a losing
business that feels successful. Every statistic here is therefore reported
*against the break-even rate for the payout actually paid*, and grouped so the
worst-performing slice is findable.

The most important function is :func:`diagnose`, which maps observed behaviour
onto the blueprint's failure modes (revenge sizing, martingale, over-trading,
cutting winners early). A trader who is at 48% win rate at 92% payout is
probably not bad at patterns -- they are probably doubling down on losses.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.assets import resolve
from .payoff import break_even_win_rate, ev_pct, test_edge, wilson_interval

ROOT = Path(__file__).resolve().parents[2]
JOURNAL_DIR = ROOT / "data" / "journal"

# Column aliases seen across platform exports and hand-typed CSVs.
ALIASES: Dict[str, Sequence[str]] = {
    "time": ("time", "timestamp", "datetime", "date", "opened_at", "open_time", "entry_time"),
    "asset": ("asset", "symbol", "pair", "instrument", "currency_pair", "market"),
    "direction": ("direction", "side", "type", "trade_type", "action"),
    "payout": ("payout", "payout_pct", "payout_percent", "return_pct", "reward", "profitability"),
    "stake": ("stake", "amount", "investment", "invested", "size", "deal"),
    "pnl": ("pnl", "profit", "result", "net", "p_and_l", "return", "outcome", "gain"),
    "result": ("result", "outcome", "status", "won", "win"),
    "setup": ("setup", "strategy", "signal", "pattern", "setup_name", "signal_type"),
    "score": ("score", "rating", "confidence", "quality"),
    "expiry": ("expiry", "expiry_minutes", "duration", "expiration", "timeframe"),
    "notes": ("notes", "note", "comment", "comments", "reason"),
}

# A trade's stake is the money at risk; a win pays stake * payout. That is the
# definition this module uses, so `stake` and `pnl` are optional as long as one
# of them plus `result` is present.
REQUIRED = ("time", "asset", "direction")


@dataclass
class JournalStats:
    payout: float
    n: int
    wins: int
    losses: int
    win_rate: float
    breakeven: float
    edge_pp: float
    ev_per_trade: float
    total_staked: float
    total_pnl: float
    roi: float
    max_drawdown: float
    avg_stake: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    ci: tuple
    p_value: float
    verdict: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ci"] = [round(self.ci[0], 4), round(self.ci[1], 4)]
        return d


def _sniff_columns(df: pd.DataFrame) -> Dict[str, str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    out: Dict[str, str] = {}
    for target, names in ALIASES.items():
        for n in names:
            if n in lowered:
                out[target] = lowered[n]
                break
    return out


def normalise(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Map an arbitrary export onto the journal's canonical schema.

    Anything that cannot be mapped is reported rather than guessed, because a
    mis-parsed journal produces confident nonsense.
    """
    if df is None or len(df) == 0:
        raise ValueError("journal import is empty")
    cols = _sniff_columns(df)
    missing = [c for c in REQUIRED if c not in cols]
    if missing:
        raise ValueError(
            f"missing required column(s) {missing}; found {list(df.columns)}. "
            f"Recognised aliases: { {k: list(v)[:4] for k, v in ALIASES.items()} }"
        )
    out = pd.DataFrame()
    for canon, src in cols.items():
        out[canon] = df[src]

    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    out["asset"] = out["asset"].astype(str).str.strip()
    out["direction"] = (
        out["direction"].astype(str).str.strip().str.lower()
        .replace({"up": "call", "buy": "call", "higher": "call", "c": "call"})
        .replace({"down": "put", "sell": "put", "lower": "put", "p": "put"})
    )

    for c in ("payout", "stake", "pnl", "score", "expiry"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Payout is stored as a percent (92 == 92%). Accept the fraction form too,
    # because exports are inconsistent about it.
    if "payout" in out.columns:
        med = out["payout"].median(skipna=True)
        if pd.notna(med) and 0 < med <= 1.0:
            out["payout"] = out["payout"] * 100.0

    # Derive `won` from whatever is available. Prefer explicit pnl, then result
    # text, then stake sign. A positive pnl is the only unambiguous signal.
    if "result" in out.columns and "won" not in out.columns:
        text = out["result"].astype(str).str.strip().str.lower()
        out["won"] = text.isin(
            ("win", "won", "w", "1", "true", "yes", "profit", "+", "tp", "take profit", "successful")
        )
    if "pnl" in out.columns and "won" not in out.columns:
        out["won"] = out["pnl"] > 0
    if "won" not in out.columns:
        raise ValueError(
            "cannot determine outcome: need a 'result'/'outcome' column or a "
            "signed 'pnl' column"
        )
    out["won"] = out["won"].astype(bool)

    if "stake" not in out.columns and "pnl" in out.columns:
        # pnl = stake * (payout/100) on a win, -stake on a loss
        payout = out.get("payout", pd.Series(92.0, index=out.index)).fillna(92.0)
        implied = np.where(out["won"], out["pnl"] / (payout / 100.0), -out["pnl"])
        out["stake"] = pd.Series(implied, index=out.index)

    for c in ("payout", "stake"):
        if c not in out.columns:
            out[c] = np.nan
    out["payout"] = out["payout"].fillna(92.0)
    out["stake"] = out["stake"].fillna(1.0)

    out["pnl"] = np.where(
        out["won"], out["stake"] * out["payout"] / 100.0, -out["stake"]
    )
    if "setup" not in out.columns:
        out["setup"] = "unspecified"
    out["setup"] = out["setup"].astype(str).str.strip().replace("", "unspecified")
    if "source" not in out.columns:
        out["source"] = source or "unknown"
    return out


CANONICAL = ["time", "asset", "direction", "payout", "stake", "pnl", "won",
             "setup", "score", "expiry", "notes", "source"]


def add_trades(df: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append trades, skipping exact duplicates.

    Re-importing the same export after adding a row is the normal workflow, and
    silently double-counting trades would corrupt every downstream statistic.
    """
    if new is None or len(new) == 0:
        return df
    key = ["time", "asset", "direction", "stake"]
    before = len(df)
    combined = pd.concat([df, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=key, keep="last").sort_values("time")
    return combined.reset_index(drop=True)


def _stats_from(df: pd.DataFrame) -> JournalStats:
    n = len(df)
    if n == 0:
        raise ValueError("no trades to summarise")
    wins = int(df["won"].sum())
    wr = wins / n
    payout = float(df["payout"].median())
    be = break_even_win_rate(payout / 100.0)
    edge_pp = (wr - be) * 100.0
    ev = ev_pct(payout / 100.0, wr)

    equity = df["pnl"].cumsum()
    peak = equity.cummax()
    staked_total = float(df["stake"].sum())
    # Drawdown in stake units, then expressed against total capital at risk.
    # (peak-equity)/peak is undefined once equity goes negative, which is exactly
    # when it matters most, so the currency form is the primary one.
    max_dd_cash = float((peak - equity).max())
    max_dd = (max_dd_cash / staked_total) if staked_total > 0 else 0.0

    gains = float(df.loc[df["won"], "pnl"].sum())
    losses = abs(float(df.loc[~df["won"], "pnl"].sum()))
    pf = gains / losses if losses > 0 else float("inf")

    staked = float(df["stake"].sum())
    pnl_total = float(df["pnl"].sum())
    # 3rd arg of test_edge is the payout FRACTION. Passing the break-even
    # win rate here silently re-derives a 65.8% break-even and makes every
    # real edge look like noise (p=1.0).
    test = test_edge(wins, n, payout / 100.0)
    ci = wilson_interval(wins, n)

    notes: List[str] = []
    if edge_pp <= 0:
        notes.append(
            f"Win rate {wr*100:.2f}% is below the {be*100:.2f}% break-even for a "
            f"{payout:.0f}% payout. More trades will not fix this."
        )
    if staked > 0 and pnl_total < 0:
        notes.append(f"Net P&L is -{abs(pnl_total):.2f} on {staked:.2f} staked.")
    significant = test.p_value < 0.05
    if not significant:
        notes.append(
            f"p={test.p_value:.3f}: not statistically distinguishable from break-even."
        )
    # Requires BOTH a positive edge over break-even AND significance. Labelling a
    # 55%-at-92% sample "PROFITABLE" while p=1.0 is precisely the false confidence
    # this project exists to prevent.
    verdict = "PROFITABLE" if (edge_pp > 0 and significant) else "NOT PROFITABLE"

    return JournalStats(
        payout=payout,
        n=n, wins=wins, losses=n - wins, win_rate=wr, breakeven=be,
        edge_pp=edge_pp, ev_per_trade=ev, total_staked=staked, total_pnl=pnl_total,
        roi=(pnl_total / staked * 100.0) if staked else 0.0, max_drawdown=max_dd,
        avg_stake=float(df["stake"].mean()), largest_win=float(df["pnl"].max()),
        largest_loss=float(df["pnl"].min()), profit_factor=pf, ci=ci,
        p_value=test.p_value, verdict=verdict, notes=notes,
    )


def stats(df: pd.DataFrame) -> JournalStats:
    return _stats_from(df)


def _slice(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per-slice stats with enough trades to mean something.

    The 10-trade minimum is not arbitrary: below that a 90% win rate is a coin
    flip that happened to land well, and reporting it invites overfitting to
    noise.
    """
    rows = []
    for key, grp in df.groupby(by):
        if len(grp) < 10:
            continue
        s = _stats_from(grp)
        d = s.to_dict()
        d[by] = key
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values("edge_pp", ascending=False).reset_index(drop=True)


def by_asset(df: pd.DataFrame) -> pd.DataFrame:
    return _slice(df, "asset")


def by_setup(df: pd.DataFrame) -> pd.DataFrame:
    return _slice(df, "setup")


def by_payout(df: pd.DataFrame) -> pd.DataFrame:
    """Grouped into payout buckets -- the point is to see the edge *rise* with
    payout, which is the signature of a real, if small, edge."""
    d = df.copy()
    bins = [0, 80, 85, 88, 90, 95, 100]
    labels = ["<80", "80-85", "85-88", "88-90", "90-95", "95+"]
    d["payout_bucket"] = pd.cut(d["payout"], bins=bins, labels=labels)
    return _slice(d, "payout_bucket")


def by_hour(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["hour_utc"] = d["time"].dt.hour
    return _slice(d, "hour_utc")


def by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["weekday"] = d["time"].dt.day_name()
    return _slice(d, "weekday")


def diagnose(df: pd.DataFrame) -> List[dict]:
    """Map observed behaviour onto the blueprint's named failure modes.

    Each finding cites the evidence that triggered it, because a rule-based
    diagnosis the trader cannot check is not actionable.
    """
    findings: List[dict] = []
    if len(df) < 20:
        return [{"severity": "info", "issue": "Not enough trades to diagnose",
                 "evidence": f"{len(df)} trades; 20 is the minimum for a useful read."}]

    s = _stats_from(df)

    if s.edge_pp <= 0:
        findings.append({
            "severity": "critical",
            "issue": f"Below break-even by {abs(s.edge_pp):.1f}pp",
            "evidence": f"win rate {s.win_rate*100:.2f}% vs {s.breakeven*100:.2f}% "
                        f"required at a {s.payout:.0f}% payout; p={s.p_value:.4f}.",
            "action": "Fix the edge before trading more. Volume amplifies loss.",
        })

    # Martingale / revenge sizing: stake rises immediately after a loss.
    stakes = df["stake"].to_numpy()
    won = df["won"].to_numpy(dtype=bool)
    after_loss = stakes[1:][~won[:-1]]
    after_win = stakes[1:][won[:-1]]
    if len(after_loss) >= 5 and len(after_win) >= 5:
        ml, mw = float(np.mean(after_loss)), float(np.mean(after_win))
        if ml > mw * 1.25:
            findings.append({
                "severity": "critical", "issue": "Martingale / revenge sizing",
                "evidence": f"average stake after a loss is {ml:.2f} vs {mw:.2f} "
                            f"after a win ({ml/mw:.2f}x).",
                "action": "Cap stake at the risk engine's flat %; never increase to recover.",
            })

    # Cutting winners early: win rate high but average win much smaller than loss.
    wins = df.loc[df["won"], "pnl"]
    losses = -df.loc[~df["won"], "pnl"]
    if len(wins) >= 10 and len(losses) >= 10:
        aw, al = float(wins.mean()), float(losses.mean())
        if aw < al * 0.8:
            findings.append({
                "severity": "high", "issue": "Winners cut short",
                "evidence": f"average win {aw:.2f} vs average loss {al:.2f}. "
                            f"{s.win_rate*100:.0f}% win rate is being wasted.",
                "action": "Hold to expiry; set a fixed exit rule and stop overriding it.",
            })

    # Over-trading: too many trades per day relative to the risk model.
    per_day = df.groupby(df["time"].dt.date).size()
    if len(per_day) >= 5:
        med = float(per_day.median())
        if med > 12:
            findings.append({
                "severity": "high", "issue": f"Over-trading ({med:.0f} trades/day median)",
                "evidence": f"median {med:.0f} trades/day, max {int(per_day.max())}.",
                "action": "Cap at the risk limit (default 10/day) and require score >= 70.",
            })

    # Concentration: is everything on one asset?
    top = df["asset"].value_counts(normalize=True)
    if len(top) > 1 and float(top.iloc[0]) > 0.6:
        findings.append({
            "severity": "medium", "issue": "Concentrated in one instrument",
            "evidence": f"{top.index[0]} is {top.iloc[0]*100:.0f}% of trades.",
            "action": "Spread risk or deliberately accept the correlation.",
        })

    # Trading outside the liquid window.
    hours = df["time"].dt.hour
    off = int(((hours < 7) | (hours > 21)).sum())
    if off > 0.15 * len(df):
        findings.append({
            "severity": "medium", "issue": "Trading outside the liquid session",
            "evidence": f"{off} trades ({off/len(df)*100:.0f}%) between 21:00 and 07:00 UTC.",
            "action": "FX OTC is dead outside the London/NY overlap.",
        })

    if not findings:
        findings.append({
            "severity": "info", "issue": "No behavioural red flags detected",
            "evidence": f"{s.n} trades, edge {s.edge_pp:+.1f}pp.",
        })
    return findings


def to_csv(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Write the journal to disk.

    Raises ``OSError`` when the destination is not writable (a serverless
    bundle is read-only). Callers that cannot persist must say so rather than
    report a successful import -- an import the user cannot retrieve is data
    they believe they saved and did not.
    """
    p = Path(path) if path else JOURNAL_DIR / "trades.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in CANONICAL if c in df.columns]
    df[cols].to_csv(p, index=False)
    return p


def from_csv(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=CANONICAL)
    raw = pd.read_csv(p)
    try:
        return normalise(raw, source=p.name)
    except ValueError:
        return pd.DataFrame(columns=CANONICAL)


def report(df: pd.DataFrame) -> dict:
    """Everything the dashboard needs, in one call."""
    if len(df) == 0:
        return {"empty": True}
    return {
        "empty": False,
        "overall": stats(df).to_dict(),
        "by_asset": by_asset(df).to_dict("records"),
        "by_setup": by_setup(df).to_dict("records"),
        "by_payout": by_payout(df).to_dict("records"),
        "by_hour": by_hour(df).to_dict("records"),
        "by_weekday": by_weekday(df).to_dict("records"),
        "diagnosis": diagnose(df),
        "equity": df.assign(equity=df["pnl"].cumsum())[
            ["time", "asset", "setup", "pnl", "equity"]
        ].assign(time=lambda d: d["time"].astype(str)).to_dict("records"),
    }
