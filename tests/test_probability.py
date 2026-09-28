"""Verify the closed-form expiry probability against Monte-Carlo simulation."""
import math
import random
import sys

sys.path.insert(0, "/Users/macbookpro/Downloads/pocket-coach")

from po_coach.core.probability import (
    atm_coinflip_probability, bars_to_years, estimate_edge,
    expiry_probability, required_bias_for_payout, vol_from_atr,
)
from po_coach.core.payoff import break_even_win_rate

print("=== 1. closed form vs Monte Carlo (100k paths) ===")
rng = random.Random(3)
sigma, t, S, K = 0.08, bars_to_years(5, "5m"), 1.0, 1.0005  # tiny OTM put-like strike
theo = expiry_probability(S, K, sigma, t, 0.0)
dt = math.sqrt(t)
wins = 0
N = 200_000
for _ in range(N):
    z = rng.gauss(0, 1)
    st = S * math.exp(-0.5 * sigma * sigma * t + sigma * math.sqrt(t) * z)
    if st > K:
        wins += 1
mc = wins / N
print(f"  theory={theo:.4f}  montecarlo={mc:.4f}  diff={abs(theo-mc)*100:.3f}pp  (mc se~{100*math.sqrt(0.25/N):.3f}pp)")

print("\n=== 2. ATM coin-flip reality check (this is the important one) ===")
for tf, bars, expo in (("1m", 1, 1), ("1m", 3, 3), ("5m", 1, 1), ("5m", 3, 3), ("15m", 1, 1)):
    t = bars_to_years(bars, tf)
    p = atm_coinflip_probability(0.08, t)
    be = break_even_win_rate(0.92)
    print(f"  {tf} expiry {expo} bar : P(win)={p*100:6.3f}%   break-even@92% payout={be*100:.2f}%   "
          f"edge={(p-be)*100:+.2f}pp")

print("\n=== 3. how far out of the money must you be? (92% payout, 5m) ===")
t = bars_to_years(1, "5m")
be = break_even_win_rate(0.92)
for dist_pct in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
    strike = 1.0 * (1 - dist_pct / 100.0)  # OTM call
    p = expiry_probability(1.0, strike, 0.08, t)
    print(f"  strike {dist_pct:5.2f}% OTM : P(win)={p*100:6.2f}%  edge={(p-be)*100:+.2f}pp")

print("\n=== 4. realistic FX vol: EUR/USD ~7% annualised ===")
for tf, bars in (("1m", 1), ("5m", 1), ("5m", 3), ("15m", 1), ("15m", 3)):
    t = bars_to_years(bars, tf)
    p = atm_coinflip_probability(0.07, t)
    print(f"  {tf} x {bars}: ATM P(win)={p*100:6.3f}%  edge vs 92% payout = {(p-be)*100:+.3f}pp")

print("\n=== 5. technical bias contribution (deliberately tiny) ===")
for bias in (0.0, 0.3, 0.6, 1.0):
    e = estimate_edge(1.0, 92.0, "call", 0.08, 1, "5m", technical_bias=bias)
    print(f"  bias={bias:.1f}: P(win)={e.p_technical*100:6.3f}%  edge={e.edge_pp_technical:+.2f}pp  "
          f"verdict={e.verdict}")

print("\n=== 6. required bias to clear a 92% payout ===")
for tf, bars in (("1m", 1), ("5m", 1), ("5m", 3), ("15m", 1)):
    rb = required_bias_for_payout(1.0, 92.0, 0.08, bars, tf, "call", 3.0)
    print(f"  {tf} x {bars}: need bias {'IMPOSSIBLE (>1.0)' if rb is None else f'{rb:+.3f}'}")

print("\n=== 7. volatility regime changes everything ===")
for sigma in (0.04, 0.07, 0.15, 0.40, 0.80):
    t = bars_to_years(1, "5m")
    p = atm_coinflip_probability(sigma, t)
    print(f"  sigma={sigma*100:5.1f}%: ATM P(win)={p*100:6.3f}%  edge vs 92% = {(p-be)*100:+.2f}pp")

print("\n=== 8. ATR -> sigma conversion sanity (EUR/USD 1.0850, ATR=0.00040) ===")
s = vol_from_atr(0.00040, 1.0850, "5m")
print(f"  implied annualised sigma = {s*100:.2f}%  (plausible for a liquid FX pair)")
s2 = vol_from_atr(450.0, 65000.0, "5m")
print(f"  BTC 65000, ATR=450: sigma = {s2*100:.1f}%  (plausible for crypto)")
