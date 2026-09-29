"""Payout mathematics, edge measurement and statistical significance.

This is the most important module in the project. A binary-options trade is a
fixed-payoff bet: win `payout` * stake, lose the whole stake. Every question a
trader asks -- "is this trade worth taking?", "does my strategy actually have
an edge?", "how much should I stake?" -- reduces to the maths in this file.

The central identity is the break-even win rate:

    p_breakeven = 1 / (1 + payout)

At a 92% payout you need 52.08% to sit exactly still. You are not "beating the
market" by winning 55% -- you are beating it by 2.9 points, and the dispersion
around that is enormous. So the tool's job is mostly to stop you from taking
trades whose real probability is below `p_breakeven`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Tuple

# Highest payout fraction we treat as real. Crypto and CFD brands quote up to
# ~300%; nothing reaches 1000%. This also separates plausible fractions from
# percentages passed unconverted (92.0), which is what `break_even_win_rate`
# uses to reject unit mistakes.
_MAX_PLAUSIBLE_PAYOUT = 10.0

try:
    from statistics import NormalDist

    _ND = NormalDist()

    def _phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _ppf(p: float) -> float:
        return _ND.inv_cdf(p)

    def _sf(z: float) -> float:
        """Upper-tail probability for a standard normal."""
        return 1.0 - _phi(z)

except Exception:  # pragma: no cover
    def _phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _ppf(p: float) -> float:
        lo, hi = -8.0, 8.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if _phi(mid) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def _sf(z: float) -> float:
        return 1.0 - _phi(z)


# --------------------------------------------------------------------------
# Basic payoff functions
# --------------------------------------------------------------------------

def break_even_win_rate(payout: float) -> float:
    """Win rate needed to hold the bankroll flat at a given payout.

    `payout` is a DECIMAL FRACTION: 0.92 means a 92% payout.

    The guard below is not decoration. Passing a percentage (92) instead of a
    fraction (0.92) silently yields a break-even of 1.07% instead of 52.08%,
    which turns every risk check into a no-op and every edge estimate into
    nonsense. That bug shipped three times while this module was being written,
    so it is now a hard error.

    The threshold is deliberately NOT ``> 1.0``. Payouts above 100% are real --
    crypto and CFD brands quote 150-300% -- and ``1.5`` is a legitimate 150%
    payout, not a mistyped percentage. The only unambiguous tell is a value big
    enough that no platform would ever quote it as a fraction, so the cut sits
    far above any real payout and well below any real percentage.
    """
    if payout <= 0:
        raise ValueError("payout must be > 0")
    if payout > _MAX_PLAUSIBLE_PAYOUT:
        raise ValueError(
            "payout looks like a percentage (%.4f). This function takes a decimal "
            "fraction, e.g. 0.92 for 92%%. Use `payout / 100.0` at the boundary "
            "where the UI/platform percentage enters." % payout
        )
    return 1.0 / (1.0 + payout)


def ev_per_unit(payout: float, win_rate: float) -> float:
    """Expected profit per 1.0 staked. Negative means the bet has no edge."""
    return win_rate * payout - (1.0 - win_rate)


def ev_pct(payout: float, win_rate: float) -> float:
    """Expected profit as a percentage of amount staked."""
    return 100.0 * ev_per_unit(payout, win_rate)


def required_win_rate(payout: float, target_roi: float) -> float:
    """Win rate needed to earn `target_roi` percent of the stake per trade."""
    if target_roi <= 0:
        return break_even_win_rate(payout)
    t = target_roi / 100.0
    return (1.0 + t) / (1.0 + payout)


def payout_for_win_rate(win_rate: float) -> float:
    """Lowest payout at which `win_rate` is profitable. 55% WR -> 0.818 payout."""
    if not 0.0 < win_rate < 1.0:
        raise ValueError("win_rate must be in (0, 1)")
    return (1.0 - win_rate) / win_rate


def max_breakeven_for_target(win_rate: float, target_roi: float) -> float:
    """Inverse of `required_win_rate`: highest acceptable payout."""
    if target_roi <= 0:
        return payout_for_win_rate(win_rate)
    t = target_roi / 100.0
    return (1.0 - win_rate) / (win_rate - t)


def kelly_fraction(payout: float, win_rate: float) -> float:
    """Full-Kelly stake as a fraction of bankroll.

    For a binary paying `payout`, Kelly reduces to:

        f* = (payout * win_rate - (1 - win_rate)) / payout

    which is just the edge divided by the payout. Returns a negative number
    when the bet has no edge; callers should treat negative as "do not bet".
    """
    return (payout * win_rate - (1.0 - win_rate)) / payout


def kelly_odds_form(b_win: float, loss_ratio: float) -> float:
    """Kelly for a bet paying `loss_ratio`x on a loss (1.0 == even money)."""
    if loss_ratio <= 0:
        raise ValueError("loss_ratio must be > 0")
    return (b_win - (1.0 - b_win) / loss_ratio) / b_win if b_win > 0 else 0.0


# --------------------------------------------------------------------------
# Uncertainty: is the observed win rate real, or did we get lucky?
# --------------------------------------------------------------------------

@dataclass
class EdgeTest:
    """Result of testing an observed win rate against break-even."""

    n: int
    wins: int
    win_rate: float
    breakeven: float
    payout: float
    edge_pp: float                 # win_rate - breakeven, in percentage points
    z_score: float
    p_value: float                 # one-tailed
    significant_at_05: bool
    ci_low: float                  # Wilson 95%
    ci_high: float
    ci_contains_breakeven: bool
    ev_pct_per_trade: float
    trades_to_breakeven_sample: Optional[int]
    required_n_for_power: Optional[int]
    power_at_n: float
    verdict: str
    caveats: Tuple[str, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["caveats"] = list(self.caveats)
        return d


def wilson_interval(wins: int, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Far better behaved than the normal approximation when n is small or the
    win rate is near 0/1, which is exactly our regime.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def one_sample_z(wins: int, n: int, p0: float) -> Tuple[float, float]:
    """One-tailed z-test that the true win rate exceeds `p0`."""
    if n <= 0:
        return (0.0, 1.0)
    phat = wins / n
    se = math.sqrt(p0 * (1.0 - p0) / n)
    if se == 0:
        return (float("inf"), 0.0) if phat > p0 else (float("-inf"), 1.0)
    z = (phat - p0) / se
    return z, _sf(z)


def required_sample_size(p0: float, p1: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Trades needed to detect a true win rate of `p1` over `p0`."""
    if p1 <= p0:
        raise ValueError("p1 must exceed p0")
    z_a = _ppf(1.0 - alpha)
    z_b = _ppf(power)
    num = (z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(p1 * (1 - p1))) ** 2
    return int(math.ceil(num / ((p1 - p0) ** 2)))


def achieved_power(p0: float, p1: float, n: int, alpha: float = 0.05) -> float:
    """Probability a test at this n would have detected a real p1 edge."""
    if n <= 0 or p1 <= p0:
        return 0.0
    se_null = math.sqrt(p0 * (1 - p0) / n)
    crit = p0 + z_for_alpha(alpha) * se_null
    se_alt = math.sqrt(p1 * (1 - p1) / n)
    return 1.0 - _phi((crit - p1) / se_alt)


def z_for_alpha(alpha: float) -> float:
    return _ppf(1.0 - alpha)


def upper_tail(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return _sf(z)


def test_edge(
    wins: int,
    n: int,
    payout: float,
    min_edge_pp: float = 3.0,
    in_sample: bool = True,
) -> EdgeTest:
    """Judge whether a sample of trades demonstrates a real, tradable edge.

    `min_edge_pp` is the smallest gap over break-even you would actually trade
    for. Anything thinner than a few points is inside the noise, no matter how
    good the p-value looks.
    """
    caveats = []
    breakeven = break_even_win_rate(payout)
    win_rate = (wins / n) if n > 0 else 0.0
    ev = ev_pct(payout, win_rate)

    if n < 30:
        z, p = 0.0, 1.0
        ci_low, ci_high = wilson_interval(wins, max(n, 1))
    else:
        z, p = one_sample_z(wins, n, breakeven)
        ci_low, ci_high = wilson_interval(wins, n)

    ci_contains = ci_low <= breakeven <= ci_high
    required_n = required_sample_size(breakeven, win_rate) if win_rate > breakeven else None
    power = achieved_power(breakeven, win_rate, n) if win_rate > breakeven else 0.0

    if n < 100:
        verdict = "NOT ENOUGH DATA"
    elif p < 0.01 and not ci_contains and (win_rate - breakeven) * 100 >= min_edge_pp:
        verdict = "REAL EDGE"
    elif p < 0.05 and not ci_contains:
        verdict = "PROBABLE EDGE"
    elif p < 0.05:
        verdict = "MARGINAL - CI still crosses break-even"
    else:
        verdict = "NO EDGE DETECTED"

    if ci_contains:
        caveats.append(
            "The 95%% confidence interval still contains the break-even rate of "
            "%.2f%%, so the sample cannot rule out a losing strategy." % (breakeven * 100)
        )
    if in_sample:
        caveats.append(
            "Tested in-sample. If these rules were tuned on the same candles, the "
            "p-value is optimistic -- re-run on unseen data before believing it."
        )
    if required_n and n < required_n:
        caveats.append(
            "Only %d trades; %d needed to detect a %.1f%% win rate with 80%% power. "
            "This test can easily miss a real edge." % (n, required_n, win_rate * 100)
        )
    if power and power < 0.8:
        caveats.append("Statistical power of this sample is %.0f%%, below the 80%% bar." % (power * 100))
    if (win_rate - breakeven) * 100 < min_edge_pp:
        caveats.append(
            "Edge is only %+.2f pp over break-even, under the %.0f pp minimum you set. "
            "A bad run wipes it out." % ((win_rate - breakeven) * 100, min_edge_pp)
        )

    return EdgeTest(
        n=n,
        wins=wins,
        win_rate=win_rate,
        breakeven=breakeven,
        payout=payout,
        edge_pp=(win_rate - breakeven) * 100.0,
        z_score=z,
        p_value=p,
        significant_at_05=(p < 0.05),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_contains_breakeven=ci_contains,
        ev_pct_per_trade=ev,
        trades_to_breakeven_sample=n,
        required_n_for_power=required_n,
        power_at_n=power,
        verdict=verdict,
        caveats=tuple(caveats),
    )


# --------------------------------------------------------------------------
# Risk of ruin
# --------------------------------------------------------------------------

def ruin_probability(
    p_win: float,
    payout: float,
    stake_frac: float,
    n_trades: int,
    bankroll: float = 10_000.0,
    ruin_level: float = 0.5,
    n_runs: int = 2000,
    seed: int = 7,
) -> dict:
    """Monte-Carlo the chance of losing `ruin_level` of the bankroll.

    A strategy with a positive edge can still wipe you out if the stake is too
    large relative to the dispersion of the payoff. This is the number that
    decides how big the stake has to be.
    """
    import random

    rng = random.Random(seed)
    ruined = 0
    ends = []
    max_dd = 0.0
    for _ in range(n_runs):
        bal = bankroll
        floor = bankroll * (1.0 - ruin_level)
        peak = bal
        peak_dd = 0.0
        dead = False
        for _ in range(n_trades):
            stake = bal * stake_frac
            bal += stake * (payout if rng.random() < p_win else -1.0)
            if bal > peak:
                peak = bal
            if peak > 0:
                dd = (peak - bal) / peak
                if dd > peak_dd:
                    peak_dd = dd
            if bal <= floor:
                dead = True
                break
        if dead:
            ruined += 1
        ends.append(bal)
        if peak_dd > max_dd:
            max_dd = peak_dd

    ends.sort()
    return {
        "ruin_probability": ruined / float(n_runs),
        "median_end_bankroll": ends[len(ends) // 2],
        "p05_end_bankroll": ends[int(0.05 * len(ends))],
        "p95_end_bankroll": ends[int(0.95 * len(ends)) - 1],
        "worst_max_drawdown": max_dd,
        "n_runs": n_runs,
        "n_trades": n_trades,
        "stake_frac": stake_frac,
    }


def stake_table(
    bankroll: float,
    payout: float,
    win_rate: float,
    fractions: Sequence[float] = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10),
    n_trades: int = 500,
    ruin_level: float = 0.5,
    n_runs: int = 1000,
    seed: int = 11,
) -> list:
    """Risk/return curve across stake sizes -- pick the biggest size that
    survives the ruin budget, not the biggest size that looks profitable."""
    rows = []
    for f in fractions:
        r = ruin_probability(
            win_rate, payout, f, n_trades, bankroll, ruin_level, n_runs, seed
        )
        ev = ev_per_unit(payout, win_rate) * f * bankroll * n_trades
        rows.append(
            {
                "stake_pct": f * 100.0,
                "stake_amount": bankroll * f,
                "expected_profit": ev,
                "ruin_probability": r["ruin_probability"],
                "p05_bankroll": r["p05_end_bankroll"],
                "median_end": r["median_end_bankroll"],
                "worst_drawdown": r["worst_max_drawdown"],
            }
        )
    return rows


def po_ladder(payout: float, win_rate: float, bankroll: float) -> dict:
    """The martingale ladder you should *not* take, costed out explicitly.

    Doubling after a loss is the standard PocketOption pattern. This returns
    the number of steps the bankroll actually supports and what the ladder costs
    at the given win rate, so the risk module can refuse to size into it.
    """
    base = bankroll * 0.01
    steps = 0
    total_risk = 0.0
    exposure = base
    while True:
        total_risk += exposure
        if total_risk > bankroll * 0.5:
            break
        steps += 1
        exposure *= 2.0
        if steps > 20:
            break
    recovery = (2.0 ** steps - 1.0) * base
    return {
        "base_stake": base,
        "supported_steps": steps,
        "capital_at_risk": total_risk,
        "capital_at_risk_pct": total_risk / bankroll * 100.0,
        "needed_to_recover": recovery,
        "all_in_required_win_rate": break_even_win_rate(payout),
        "p_win_all_steps": win_rate ** steps,
    }
