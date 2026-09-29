"""Tests for payout sensitivity.

Script-style, matching the other test files in this directory: run it directly
(`python tests/test_payout_sensitivity.py`) rather than via pytest, which cannot
collect modules that call sys.exit() at import.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from po_coach.core.payoff import break_even_win_rate
from po_coach.core.payout_sensitivity import analyse_payout_sensitivity

FAILURES: list[str] = []


def check(label: str, got, want=None, predicate=None) -> None:
    try:
        if predicate is not None:
            assert predicate(got), f"predicate failed on {got!r}"
        else:
            assert got == want, f"got {got!r}, want {want!r}"
        print(f"  PASS  {label}")
    except AssertionError as exc:
        FAILURES.append(f"{label}: {exc}")
        print(f"  FAIL  {label}: {exc}")


def raises(label: str, fn, *a, **k) -> None:
    try:
        fn(*a, **k)
    except ValueError:
        print(f"  PASS  {label}")
        return
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{label}: raised {type(exc).__name__}, want ValueError")
        print(f"  FAIL  {label}: raised {type(exc).__name__}, want ValueError")
        return
    FAILURES.append(f"{label}: did not raise")
    print(f"  FAIL  {label}: did not raise")


print("\n== guard: legitimate high payouts must NOT be mistaken for percentages ==")
# Regression: the guard was `payout > 1.0`, which rejected a real 150% payout.
# 150%-300% payouts exist on crypto/CFD venues, so the guard has to separate a
# high payout from a mistyped percentage without rejecting either.
check("150% payout accepted", round(break_even_win_rate(1.50), 4), 0.4)
check("200% payout accepted", round(break_even_win_rate(2.00), 4), 0.3333)
check("300% payout accepted", round(break_even_win_rate(3.00), 4), 0.25)
check("92% payout unchanged", round(break_even_win_rate(0.92), 4), 0.5208)
check("85% payout accepted", round(break_even_win_rate(0.85), 4), 0.5405)
raises("92.0 still rejected as a percentage", break_even_win_rate, 92.0)
raises("85.0 still rejected as a percentage", break_even_win_rate, 85.0)

print("\n== the measured case: 42.15% at a 92% payout ==")
r = analyse_payout_sensitivity(
    0.4215, 1694, effective_n=884, current_payout=0.92
)
check("below_random flagged", r.below_random, True)
check("breakeven payout 137%", round(r.breakeven_payout_pct, 1), 137.2)

by_payout = {round(t.payout_pct): t for t in r.tiers}
check("92% tier loses", by_payout[92].profitable, False)
check("92% tier EV is -19.07%", round(by_payout[92].ev_pct_per_trade, 2), -19.07)
check("92% breakeven 52.08%", round(by_payout[92].breakeven * 100, 2), 52.08)
# The whole point of the panel: a >100% payout is superlinear, so 42% IS positive
# at 150%. An earlier version of this module claimed no payout could rescue a
# sub-50% win rate, which is false and contradicted its own table.
check("150% tier is profitable", by_payout[150].profitable, True)
check("150% tier EV is +5.37%", round(by_payout[150].ev_pct_per_trade, 2), 5.37)
check("200% tier clears CI too", by_payout[200].profitable_across_ci, True)
check("150% does NOT clear CI", by_payout[150].profitable_across_ci, False)
check(
    "summary agrees the 150% tier is profitable",
    "150%" in r.summary,
    True,
)
check(
    "summary does not claim hopeless",
    "No payout makes this tradeable" in r.summary,
    False,
)

print("\n== a genuinely positive edge prices normally ==")
good = analyse_payout_sensitivity(0.60, 5000, current_payout=0.92)
check("60% WR beats 92% payout", good.below_random, False)
check("breakeven payout 66.7%", round(good.breakeven_payout_pct, 1), 66.7)
g_by = {round(t.payout_pct): t for t in good.tiers}
check("92% profitable at 60% WR", g_by[92].profitable, True)
check("60% payout not profitable", g_by[60].profitable, False)
check("CI is ordered", good.ci_low < good.ci_high, True)

print("\n== summary stays honest when no tier works ==")
# 20% is worse than random AND too thin to price at any tier in DEFAULT_TIERS:
# EV at a 300% payout is 0.20*3 - 0.80 = -0.20.
hopeless = analyse_payout_sensitivity(0.20, 1694)
check("no tier profitable at 20% WR", [t.profitable for t in hopeless.tiers], [False]*13)
check("summary says no edge to price", "no directional edge to price" in hopeless.summary, True)
check("summary does not name a tier", "first quoted tier" in hopeless.summary, False)
check("20% is not called a real edge", "edge is real" in hopeless.summary, False)

print("\n== input validation ==")
raises("win_rate 0 rejected", analyse_payout_sensitivity, 0.0, 100)
raises("win_rate 1 rejected", analyse_payout_sensitivity, 1.0, 100)
raises("win_rate >1 rejected", analyse_payout_sensitivity, 1.5, 100)
raises("n_trades 0 rejected", analyse_payout_sensitivity, 0.5, 0)
raises(
    "current_payout as percent rejected",
    analyse_payout_sensitivity,
    0.5,
    100,
    current_payout=92.0,
)

print("\n== CI bounds bracket the headline correctly ==")
# Inverting break-even flips the direction: a HIGHER win rate needs a LOWER
# payout. So the payout interval derived from the WR interval must bracket the
# point estimate from both sides, with the pessimistic (higher) payout on top.
check(
    "payout CI brackets the point estimate",
    r.breakeven_payout_ci_low < r.breakeven_payout_pct < r.breakeven_payout_ci_high,
    True,
)
check(
    "pessimistic payout is the upper bound",
    round(r.breakeven_payout_ci_high, 1),
    151.1,
)
check(
    "optimistic payout is the lower bound",
    round(r.breakeven_payout_ci_low, 1),
    124.6,
)
check(
    "pessimistic bound is above every profitable tier's breakeven story",
    r.breakeven_payout_ci_high > r.breakeven_payout_pct > r.breakeven_payout_ci_low,
    True,
)

print("\n" + "=" * 60)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S)")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
