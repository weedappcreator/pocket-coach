"""End-to-end smoke test of the core engine on real Yahoo Finance data.

Checks the numbers that must be right before any strategy work:
closed-form probability vs Monte Carlo, indicator sanity, pattern detection
firing at all, and the backtester + significance gate running end to end.
"""
import math
import random
import sys
import time

sys.path.insert(0, "/Users/macbookpro/Downloads/pocket-coach")

import numpy as np
import pandas as pd

from po_coach.core.probability import (
    atm_coinflip_probability, bars_to_years, estimate_edge,
    expiry_probability, required_bias_for_payout, sigma_from_candles,
    vol_from_atr, vol_plausibility,
)
from po_coach.core.payoff import break_even_win_rate, kelly_fraction, po_ladder, ruin_probability, test_edge
from po_coach.core.risk import RiskConfig, RiskEngine, diagnose_behaviour
from po_coach.core.indicators import compute_features
from po_coach.core.structure import analyse
from po_coach.core.patterns import scan
from po_coach.core.scoring import ScoringConfig, score_setup, rank_setups
from po_coach.core import backtest as bt

ok = lambda c: "\033[32mOK\033[0m" if c else "\033[31mFAIL\033[0m"
fails = []


def check(label, cond, extra=""):
    print(f"  [{ok(cond)}] {label} {extra}")
    if not cond:
        fails.append(label)


print("=" * 72)
print("1. CLOSED-FORM PROBABILITY vs MONTE CARLO")
print("=" * 72)
rng = random.Random(3)
sigma, t, S, K = 0.08, bars_to_years(5, "5m"), 1.0, 1.0005
theo = expiry_probability(S, K, sigma, t, 0.0)
N = 40_000
wins = sum(
    1 for _ in range(N)
    if 1.0 * math.exp(-0.5 * sigma * sigma * t + sigma * math.sqrt(t) * rng.gauss(0, 1)) > K
)
mc = wins / N
se = 100 * math.sqrt(0.25 / N)
print(f"  theory={theo:.4f}  mc={mc:.4f}  diff={abs(theo-mc)*100:.3f}pp (mc 95% band +-{1.96*se:.3f}pp)")
check("closed form matches MC within 2 se", abs(theo - mc) * 100 < 2 * se)

print()
print("=" * 72)
print("2. THE ATM COIN-FLIP PROBLEM  (the number that should humble you)")
print("=" * 72)
be92 = break_even_win_rate(0.92)
print(f"  break-even at 92% payout = {be92*100:.2f}%")
for tf, bars in (("1m", 1), ("1m", 3), ("5m", 1), ("5m", 3), ("15m", 1), ("15m", 3), ("1h", 1)):
    t_ = bars_to_years(bars, tf)
    p = atm_coinflip_probability(0.08, t_)
    print(f"  {tf:>3} x {bars} bar : ATM P(win) = {p*100:6.3f}%   edge = {(p-be92)*100:+.3f}pp")
check("all ATM 1-5min trades are negative EV at 92% payout",
      all(atm_coinflip_probability(0.08, bars_to_years(b, tf)) < be92
          for tf, b in (("1m", 1), ("1m", 3), ("5m", 1), ("5m", 3))))

print()
print("=" * 72)
print("3. KELLY / LADDER / RUIN")
print("=" * 72)
for wr in (0.50, 0.52, 0.55, 0.60):
    print(f"  WR {wr*100:.0f}% @92% payout: kelly={kelly_fraction(0.92, wr)*100:+.2f}%  ev={100*(wr*0.92-(1-wr)):+.2f}%")
check("kelly is negative below breakeven", kelly_fraction(0.92, 0.50) < 0)
check("kelly is positive above breakeven", kelly_fraction(0.92, 0.55) > 0)

lad = po_ladder(0.92, 0.55, 10_000)
print(f"  martingale ladder on $10k: {lad['supported_steps']} steps, "
      f"{lad['capital_at_risk_pct']:.1f}% of bankroll at risk, "
      f"all-steps win prob = {lad['p_win_all_steps']*100:.2f}%")
# A ${:,.0f} ladder that risks {:.0f}% of the bankroll needs {} wins IN A ROW.
# At a 55% win rate that is a {}-in-{} shot, i.e. a {}-in-1,000 event.
check("martingale all-steps win probability is tiny", lad["p_win_all_steps"] < 0.10,
      "= 1 in {:.0f}".format(1.0 / lad["p_win_all_steps"]))

r2 = ruin_probability(0.55, 0.92, 0.02, 1000, 10_000, 0.5, 300, seed=5)
r10 = ruin_probability(0.55, 0.92, 0.10, 1000, 10_000, 0.5, 300, seed=5)
print(f"  2% stake: ruin={r2['ruin_probability']*100:.1f}%  median end=${r2['median_end_bankroll']:,.0f}  worst DD={r2['worst_max_drawdown']*100:.0f}%")
print(f" 10% stake: ruin={r10['ruin_probability']*100:.1f}%  median end=${r10['median_end_bankroll']:,.0f}  worst DD={r10['worst_max_drawdown']*100:.0f}%")
check("10% stake is materially riskier than 2%", r10["ruin_probability"] > r2["ruin_probability"])

print()
print("=" * 72)
print("4. RISK ENGINE ENFORCEMENT")
print("=" * 72)
eng = RiskEngine(RiskConfig(), bankroll=10_000)
d = eng.evaluate(score=85, payout=92, win_rate=0.58, conviction=1.0)
print(f"  good setup 85 score, 58% WR (+5.9pp), 92% payout -> approved={d.approved} stake=${d.stake} ({d.stake_pct}%)")
check("approves a genuine edge", d.approved and d.stake > 0)

# 55% at 92% is only +2.92pp: below the 3pp floor, so it must be refused.
d_55 = eng.evaluate(score=85, payout=92, win_rate=0.55)
check("refuses 55% at 92% (only +2.9pp)", not d_55.approved,
      f"-> {d_55.vetoes[0] if d_55.vetoes else 'APPROVED (wrong)'}")

d2 = eng.evaluate(score=85, payout=78, win_rate=0.55)
check("vetoes sub-80% payout", not d2.approved and any("Payout" in v for v in d2.vetoes))

d3 = eng.evaluate(score=65, payout=92, win_rate=0.55)
check("vetoes sub-70 score", not d3.approved)

d4 = eng.evaluate(score=85, payout=92, win_rate=0.53)
check("vetoes thin 0.9pp edge", not d4.approved and any("Edge" in v for v in d4.vetoes))

for _ in range(3):
    eng.record_result(won=False, stake=150.0, payout=92)
d5 = eng.evaluate(score=90, payout=92, win_rate=0.58)
check("hard stop after 3 losses", not d5.approved and any("consecutive" in v for v in d5.vetoes))
print(f"  status: {eng.status()['paused']} paused, {eng.status()['consecutive_losses']} losses, "
      f"bankroll=${eng.status()['bankroll']:,.2f}")

eng2 = RiskEngine(RiskConfig(), bankroll=10_000)
eng2.record_result(won=False, stake=100, payout=92)
d6 = eng2.evaluate(score=90, payout=92, win_rate=0.58, conviction=3.0)
check("does not size up after a loss", d6.stake <= 100.0 + 1e-6, f"(stake={d6.stake})")
print(f"  post-loss stake clamped to ${d6.stake} (was asked for 3x conviction)")
for w in d6.warnings:
    print(f"    warn: {w}")

print()
print("=" * 72)
print("5. BEHAVIOUR DIAGNOSIS ON A SIMULATED KEVIN-STYLE HISTORY")
print("=" * 72)
import random as _r
rg = _r.Random(1)
hist, bal = [], 10_000.0
stake = 150.0
for k in range(600):
    base_wr = 0.50
    if k > 0 and not hist[-1]["won"]:
        base_wr = 0.47
    won = rg.random() < base_wr
    hist.append({"won": won, "stake": stake, "payout": 92.0})
    stake = 150.0 if won else stake * 2.0
    if stake > 3000:
        stake = 150.0
dg = diagnose_behaviour(hist, 92.0)
print(f"  n={dg['n_trades']}  win_rate={dg['win_rate']*100:.2f}%  breakeven={dg['breakeven']*100:.2f}%  "
      f"edge={dg['edge_pp']:+.2f}pp  pnl=${dg['total_pnl']:,.0f}")
print(f"  stake after loss is {dg['stake_after_loss_pct_higher']:+.0f}% higher; longest losing run={dg['longest_losing_run']}")
for f in dg["findings"]:
    print(f"  - {f}")
check("diagnoses negative expectancy", dg["edge_pp"] < 0)
check("detects the martingale", dg["stake_after_loss_pct_higher"] > 20)

print()
print("=" * 72)
print("6. REAL MARKET DATA + INDICATORS + PATTERNS")
print("=" * 72)
t0 = time.time()
try:
    import yfinance as yf

    raw = yf.download("EURUSD=X", period="60d", interval="5m", progress=False, auto_adjust=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError("no data")
    df = raw.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.lower)
    print(f"  fetched EURUSD=X 5m: {len(df)} bars  {df.index[0]} -> {df.index[-1]}")
    check("got real candles", len(df) > 500, f"({len(df)} bars)")

    feats = compute_features(df)
    last = feats.iloc[-1]
    sig_ann = sigma_from_candles(feats, 200, "5m")
    print(f"  robust sigma estimate: {sig_ann*100:.1f}%/yr  "
          f"plausible_for_fx={vol_plausibility(sig_ann, 'fx')['ok']}")
    print(f"  RSI={last['rsi']:.1f}  ATR%={last['atr_pct']:.4f}  chop={last['chop']:.1f}  "
          f"ADX={last['adx']:.1f}  vol20={last['vol_20']*100:.4f}%/bar")
    check("indicators are in valid ranges",
          0 <= last["rsi"] <= 100 and 0 <= last["chop"] <= 100 and last["atr_pct"] > 0)
    check("annualized vol is plausible for EUR/USD", vol_plausibility(sig_ann, "fx")["ok"],
          f"({sig_ann*100:.1f}%/yr)")

    st = analyse(feats)
    print(f"  structure: trend={st.trend} bias={st.bias:+.2f} swings={len(st.swings)} zones={len(st.zones)} ema_above={st.ema_above}")
    check("structure analysis produces a read", st.trend != "" and len(st.swings) > 0)
    print(f"    note: {st.notes[0] if st.notes else '-'}")

    tsc = time.time()
    found = scan(feats, warmup=60)
    dt = time.time() - tsc
    print(f"  pattern scan: {len(found)} setups found across {len(feats)} bars in {dt:.1f}s")
    if len(found):
        vc = found["name"].value_counts().to_dict()
        print(f"    by type: {vc}")
    check("pattern detectors fire on real data", len(found) > 0)

    print()
    print("=" * 72)
    print("7. SCORING + RANKING")
    print("=" * 72)
    rows = found.tail(40)
    scored = []
    for _, r in rows.iterrows():
        sub = feats.iloc[max(0, int(r["bar"]) - 120): int(r["bar"]) + 1]
        if len(sub) < 40:
            continue
        sst = analyse(sub)
        sc = score_setup(
            asset="EUR/USD OTC", feats=sub, structure=sst,
            direction=r["direction"], setups_found=[r["name"]],
            required_setups=None, alignment=sst.bias, all_agree=True,
            payout=92.0, expiry_bars=1, timeframe="5m",
            sigma_annual=sig_ann, cfg=ScoringConfig(in_sample=False),
        )
        scored.append(sc)
    scored.sort(key=lambda s: s.score, reverse=True)
    top = scored[:3]
    for s in top:
        ev = f"P={s.edge.p_technical*100:.2f}% edge={s.edge.edge_pp_technical:+.2f}pp" if s.edge else "no edge"
        print(f"  {s.asset} {s.direction.upper():4} score={s.score:5.1f}  {ev}")
        print(f"     setups={[c.name+':'+str(round(c.points,1)) for c in s.components]}")
        if s.vetoes:
            print(f"     VETO: {s.vetoes[0]}")
    check("scoring produces a ranked list", len(top) > 0)
    ranked = rank_setups(scored, top_n=5)
    print(f"  rank_setups returned {len(ranked)} of {len(scored)} (vetoes removed)")
    check("most scored setups are vetoed as negative EV",
          len(ranked) < len(scored) or len(ranked) == 0,
          f"({len(ranked)}/{len(scored)} survived)")

    print()
    print("=" * 72)
    print("8. BACKTEST + THE GO/NO-GO GATE")
    print("=" * 72)
    bl = bt.baseline_check(feats, expiry_bars=1, payout=92.0)
    print(f"  baseline 'always CALL': n={bl['n']} win_rate={bl['always_call_win_rate']*100:.2f}% "
          f"edge={bl['edge_pp']:+.2f}pp  verdict={bl['edge_test']['verdict']}")
    check("baseline is computed", bl["ok"])

    sig = found.copy()
    sig["expiry_bars"] = 1
    sig["score"] = 70.0
    sig["hourly"] = False
    sig["setups"] = sig["name"]
    trades, eq = bt.simulate(feats, sig, payout=92.0, stake_pct=1.0, asset="EUR/USD OTC", timeframe="5m")
    print(f"  simulated {len(trades)} trades from {len(sig)} signals")
    res = bt.evaluate(trades, eq, "all-setups", "EUR/USD OTC", "5m", 92.0, 10_000.0, in_sample=True)
    print(f"  win_rate={res.win_rate*100:.2f}%  n={res.n_trades}  effective_n={res.effective_n}")
    print(f"  edge={res.edge_pp:+.2f}pp  ev={res.ev_pct_per_trade:+.2f}%/trade  return={res.total_return_pct:+.2f}%  maxDD={res.max_drawdown_pct:.1f}%")
    print(f"  95% CI: [{res.ci_low*100:.2f}%, {res.ci_high*100:.2f}%]  (breakeven {res.breakeven*100:.2f}%)")
    print(f"  GATE: {res.gate_reason[:150]}")
    for nt in res.notes:
        print(f"    - {nt}")
    check("gate fails (correctly, in-sample)", not res.gate_passed)
    check("effective n is deflated for overlap", res.effective_n < res.n_trades)
    check("by_expiry stats produced", len(res.by_expiry) >= 1)
    check("by_setup stats produced", len(res.by_setup) >= 1)
    check("equity curve produced", len(res.equity_curve) > 0)

    print()
    print("=" * 72)
    print("9. WALK-FORWARD")
    print("=" * 72)
    def build(train):
        f2 = compute_features(train)
        from po_coach.core.structure import find_swings as _fs
        sw = _fs(f2, 2, 2)
        out = scan(f2, warmup=60, swings=sw)
        if len(out) == 0:
            return out
        out["expiry_bars"] = 1
        out["score"] = 70.0
        out["hourly"] = False
        out["setups"] = out["name"]
        return out

    inr, outr = bt.walk_forward(df, build, payout=92.0, asset="EUR/USD OTC", timeframe="5m")
    print(f"  in-sample : n={inr.n_trades:4d} wr={inr.win_rate*100:.2f}%  {inr.gate_reason[:60]}")
    print(f"  out-sample: n={outr.n_trades:4d} wr={outr.win_rate*100:.2f}%  edge={outr.edge_pp:+.2f}pp")
    print(f"    {outr.notes[0]}")
    check("walk-forward produces both legs", inr.n_trades > 0 and outr.n_trades > 0)

except ImportError as e:
    print(f"  yfinance unavailable: {e}")
except Exception as e:
    import traceback
    traceback.print_exc()
    fails.append(f"data stage crashed: {e}")

print()
print("=" * 72)
print(f"RESULT: {'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILURE(S): ' + ', '.join(fails)}")
print("=" * 72)
sys.exit(1 if fails else 0)
