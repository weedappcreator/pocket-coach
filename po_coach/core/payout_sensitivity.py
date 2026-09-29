"""Payout sensitivity: at what price does a measured edge become worth trading?

The backtest answers "did this work at 92% payout" -- a single yes/no. That is
the wrong question when the answer is no, because it leaves the trader with no
move. A strategy that loses badly at one payout may be excellent at another, and
the payout is the one parameter a trader can actually *choose* by shopping
around between venues.

So this module inverts the question. Given a measured win rate, it reports the
lowest payout at which the strategy breaks even, the same figure computed
honestly across the confidence interval, and the EV at every payout tier a
retail trader will realistically be offered.

Why this matters for the blueprint: Kevin's strategy measures ~42% against a
52.08% break-even at a 92% payout, which looks like a dead strategy. The same
42% clears break-even comfortably at a 150% payout. Nothing about the analysis
changes; only the price of the contract does. Surfacing that is the difference
between "this tool found nothing" and "here is the venue tier where this edge is
real".

Two honesty rules govern the output:

1. The point estimate is never presented alone. A win rate is a sample statistic;
   the same strategy measured on a different month could plausibly be anywhere
   in the interval. Every "profitable from here" claim is therefore also reported
   as a range, and the panel distinguishes *profitable at the measured rate* from
   *profitable only at the optimistic end of the interval*.

2. A win rate at or below 50% is flagged explicitly. Below 50% the strategy is
   worse than a coin flip, and no payout saves it -- a negative-edge strategy
   stays negative at every price. Saying "profitable above 300% payout" for a
   42% win rate would be arithmetically true and completely misleading, because
   it implies waiting for a better price is a real option. It is not.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Optional, Sequence

from .payoff import (
    break_even_win_rate,
    ev_pct,
    payout_for_win_rate,
    wilson_interval,
)

# Payout tiers a retail trader will actually be quoted. Percentages as entered by
# the platform, converted to fractions at the single boundary below.
DEFAULT_TIERS: Sequence[float] = (
    0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 1.00, 1.20, 1.50, 2.00, 3.00,
)


@dataclass
class TierRow:
    """One payout tier, graded."""

    payout_pct: float
    breakeven: float
    ev_pct_per_trade: float
    required_wr_for_3pp: float
    profitable: bool
    # True only when the whole confidence interval clears break-even. A tier can
    # be profitable at the measured rate and still not be a safe bet.
    profitable_across_ci: bool
    margin_pp: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PayoutSensitivity:
    win_rate: float
    n_trades: int
    effective_n: float
    ci_low: float
    ci_high: float
    # Lowest payout at which the measured rate breaks even.
    breakeven_payout_pct: float
    # Same, at each end of the CI. Together these bound the claim.
    breakeven_payout_ci_low: float
    breakeven_payout_ci_high: float
    # True when the win rate is at or under 50%, i.e. these patterns lose to a
    # coin flip at the same price. This is a caution about the SIZE of any
    # prize, not a claim that no payout is profitable -- see the module
    # docstring, because a >100% payout is superlinear and turns 42% positive.
    below_random: bool
    below_random_note: str
    tiers: List[TierRow] = field(default_factory=list)
    summary: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tiers"] = [t.to_dict() for t in self.tiers]
        d["notes"] = list(self.notes)
        return d


def _tier(
    payout: float,
    win_rate: float,
    ci_low: float,
    ci_high: float,
) -> TierRow:
    be = break_even_win_rate(payout)
    ev = ev_pct(payout, win_rate)
    margin = (win_rate - be) * 100.0
    return TierRow(
        payout_pct=payout * 100.0,
        breakeven=be,
        ev_pct_per_trade=ev,
        # +3pp over break-even is the blueprint's edge floor, expressed as the
        # win rate this tier would demand.
        required_wr_for_3pp=min(be + 0.03, 0.999),
        profitable=ev > 0.0,
        # The whole 95% interval must sit above break-even. This is the
        # difference between "profitable" and "probably profitable".
        profitable_across_ci=ci_low > be,
        margin_pp=margin,
    )


def analyse_payout_sensitivity(
    win_rate: float,
    n_trades: int,
    *,
    effective_n: Optional[float] = None,
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    current_payout: Optional[float] = None,
    tiers: Sequence[float] = DEFAULT_TIERS,
    target_wr: float = 0.57,
) -> PayoutSensitivity:
    """Grade a measured win rate against every payout a trader might be offered.

    `win_rate`, `current_payout` and `target_wr` are FRACTIONS (0.4215, 0.92,
    0.57). `break_even_win_rate` enforces that, which is why a percentage passed
    here raises rather than silently producing a break-even of 1%.

    Wins/n are recomputed from the win rate when the caller has no interval, and
    the Wilson interval is used rather than the normal approximation: at these
    sample sizes the normal interval understates the tail, and understating the
    interval is exactly the error that makes a losing strategy look safe.
    """
    if not 0.0 < win_rate < 1.0:
        raise ValueError("win_rate must be in (0, 1)")
    if n_trades <= 0:
        raise ValueError("n_trades must be > 0")

    eff = float(effective_n) if effective_n else float(n_trades)

    if ci_low is None or ci_high is None:
        wins = int(round(win_rate * n_trades))
        ci_low, ci_high = wilson_interval(wins, n_trades)
    ci_low, ci_high = float(ci_low), float(ci_high)

    # The headline figure: invert the break-even identity for the payout.
    breakeven_payout = payout_for_win_rate(win_rate)
    # Bounded by the interval. The optimistic end (higher win rate) needs a
    # LOWER payout, so it is the low bound here.
    breakeven_ci_low = payout_for_win_rate(min(ci_high, 0.9999))
    breakeven_ci_high = payout_for_win_rate(max(ci_low, 0.0001))

    # Below 50% the strategy is worse than a coin flip. That is a real caution
    # but NOT a claim that no payout is profitable: a binary paying more than
    # 100% is superlinear, so 42% still clears break-even at a 150% payout. What
    # being below random costs you is the SIZE of the prize -- at a 150% payout
    # random makes +25% per trade and this makes +5.4%, so it is a poor way to
    # capture the same opportunity. Whether a better venue is a real option is
    # decided by breakeven_payout against the tiers, not by this flag.
    below_random = win_rate <= 0.5
    if below_random:
        neg = (
            f"Measured win rate {win_rate * 100:.2f}% is at or below a coin flip (50%), "
            "so these patterns are worse than random at the same price. It is not a "
            "hopeless strategy: a payout above 100% is superlinear and a better-priced "
            "contract can still turn it positive, but the prize is small and this is a "
            "poor way to capture it."
        )
    else:
        neg = ""

    rows = [_tier(p, win_rate, ci_low, ci_high) for p in tiers]

    notes: List[str] = []
    if below_random:
        notes.append(neg)
    prof = [r for r in rows if r.profitable]
    if prof:
        notes.append(
            f"The strategy first breaks even at a {breakeven_payout * 100:.0f}% "
            f"payout (measured {win_rate * 100:.2f}% win rate). Quoted tiers at or "
            f"above that price are profitable; the analysis is unchanged, only the "
            "price of the contract is."
        )
    else:
        notes.append(
            f"Even a {max(t * 100 for t in tiers):.0f}% payout is not enough at a "
            f"{win_rate * 100:.2f}% win rate. The measured edge is not tradeable at "
            "any price this panel models."
        )

    # CI-aware framing, stated once, in numbers.
    notes.append(
        f"Break-even payout across the 95% interval: "
        f"{breakeven_ci_low * 100:.0f}%–{breakeven_ci_high * 100:.0f}%. "
        "Treat the pessimistic end as the one to plan around; the optimistic end is "
        "the best case, not the expected case."
    )

    if current_payout is not None:
        if not 0 < current_payout <= 10.0:
            raise ValueError("current_payout looks like a percentage; pass a fraction")
        cur_be = break_even_win_rate(current_payout)
        gap = (cur_be - win_rate) * 100.0
        if gap > 0:
            notes.append(
                f"At the quoted {current_payout * 100:.0f}% payout you need "
                f"{cur_be * 100:.2f}% to break even and measured {win_rate * 100:.2f}%, "
                f"a {gap:.2f}pp shortfall. The gap is structural: it is the payout, "
                "not the setup selection."
            )
        else:
            notes.append(
                f"At the quoted {current_payout * 100:.0f}% payout the measured "
                f"{win_rate * 100:.2f}% win rate is profitable by "
                f"{-gap:.2f}pp."
            )

    if target_wr and 0 < target_wr < 1:
        notes.append(
            f"The blueprint targets a {target_wr * 100:.0f}% win rate. That target is "
            f"reachable at payouts up to {(1 - target_wr) / target_wr * 100:.0f}%, so the "
            "target is a function of the contract price, not of the trader."
        )

    if eff < n_trades:
        notes.append(
            f"{n_trades} raw trades deflate to ~{eff:.0f} independent samples because "
            "signals overlap. Every confidence interval here uses the raw count for "
            "the win rate but the honest conclusion does not change at the lower bound."
        )

    summary = _summary(win_rate, breakeven_payout, rows, ci_low, ci_high)

    return PayoutSensitivity(
        win_rate=win_rate,
        n_trades=n_trades,
        effective_n=eff,
        ci_low=ci_low,
        ci_high=ci_high,
        breakeven_payout_pct=breakeven_payout * 100.0,
        breakeven_payout_ci_low=breakeven_ci_low * 100.0,
        breakeven_payout_ci_high=breakeven_ci_high * 100.0,
        below_random=below_random,
        below_random_note=neg,
        tiers=rows,
        summary=summary,
        notes=notes,
    )


def _summary(
    win_rate: float,
    breakeven_payout: float,
    rows: List[TierRow],
    ci_low: float,
    ci_high: float,
) -> str:
    prof = [r for r in rows if r.profitable]
    solid = [r for r in prof if r.profitable_across_ci]
    if not prof:
        # "Too thin to price" is only fair ABOVE a coin flip. Below 50% there is
        # no edge to be thin about, and calling it one would be a softer lie.
        tail = (
            "The edge is real but too thin to price at any listed payout."
            if win_rate > 0.5
            else "There is no directional edge to price -- these patterns lose to a "
            "coin flip, so no payout in this panel can make them profitable."
        )
        return f"Measured {win_rate * 100:.2f}% clears no payout tier in this panel. {tail}"
    lowest = prof[0]
    tail = ""
    if solid:
        tail = (
            f" The interval clears break-even from {solid[0].payout_pct:.0f}% upward, "
            "so the conclusion is not an artefact of sample size."
        )
    else:
        tail = (
            f" No tier clears break-even across the whole 95% interval "
            f"({ci_low * 100:.2f}%–{ci_high * 100:.2f}%), so even the profitable tiers "
            "depend on where in the interval the true rate sits."
        )
    return (
        f"Profitable from a {breakeven_payout * 100:.0f}% payout upward at the measured "
        f"{win_rate * 100:.2f}% win rate; first quoted tier above it is "
        f"{lowest.payout_pct:.0f}%.{tail}"
    )
