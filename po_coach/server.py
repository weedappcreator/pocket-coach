"""FastAPI server for the dashboard.

Deploy notes, because these are the parts that break in production:

* **No local disk.** The decision ledger and the candle cache both write to
  disk, which is read-only or ephemeral on serverless. Both are wrapped so a
  failed write degrades to "no history" instead of a 500.
* **Cold starts are real.** The first scan pulls candles from Yahoo and takes
  several seconds. Results are cached in-process with a short TTL so a demo
  clicking around does not re-pay the fetch each time.
* **This is decision support, not an order router.** There is deliberately no
  endpoint that places a trade.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import Orchestrator, Strategy
from .core import decisions as ledger
from .core import journal as journal_mod
from .core import backtest as backtest_mod
from .core import payout_sensitivity as payout_mod
from .core import scoring as scoring_mod
from .core.indicators import compute_features
from .core.patterns import scan as pattern_scan
from .core.structure import analyse
from .data import assets as assets_mod
from .data import feed as feed_mod

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "public" if (ROOT / "public").exists() else ROOT / "web"
JOURNAL_CSV = ROOT / "data" / "journal" / "trades.csv"

app = FastAPI(title="Pocket Coach", version="1.0")

# One orchestrator, reused. Rebuilding per request would re-read the YAML, rebuild
# the risk engine (and so lose the running daily-loss and streak state) on every
# click -- which would silently reset the risk limits the dashboard is displaying.
_ORCH: Optional[Orchestrator] = None
_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}
_CACHE_TTL = 60.0


def orch() -> Orchestrator:
    global _ORCH
    if _ORCH is None:
        with _LOCK:
            if _ORCH is None:
                _ORCH = Orchestrator(Strategy.load(), bankroll=10_000.0)
    return _ORCH


def cached(key: str, ttl: float, producer):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = producer()
    _CACHE[key] = (now, val)
    return val


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "pocket-coach", "ts": time.time()}


@app.get("/api/assets")
def list_assets() -> dict:
    return {
        "assets": assets_mod.list_assets(),
        "note": (
            "OTC instruments are synthetic platform instruments. Where a public "
            "proxy is used it is labelled is_proxy=true and its statistics are "
            "indicative only."
        ),
    }


@app.get("/api/scan")
def scan(
    assets: Optional[str] = Query(None, description="comma-separated symbols"),
    period: str = Query("5d"),
    force_killzone: bool = Query(False, description="ignore the session filter (demo)"),
    refresh: bool = Query(False),
    debug: bool = Query(False, description="return the traceback instead of a 500"),
) -> dict:
    o = orch()
    if force_killzone:
        o.strategy.raw.setdefault("markets", {})["killzone_only"] = False
    syms = [s.strip() for s in assets.split(",")] if assets else None

    def produce() -> dict:
        report = o.scan(syms, period=period, write_ledger=True)
        try:
            ledger.sweep_stale()
        except OSError:
            pass
        d = report.to_dict()
        d["force_killzone"] = force_killzone
        return d

    key = f"scan:{syms}:{period}:{force_killzone}"
    if refresh:
        _CACHE.pop(key, None)
    try:
        return cached(key, _CACHE_TTL, produce)
    except Exception as exc:
        # A bare 500 on a serverless function is close to useless: the stack
        # goes to a log stream nobody is watching. Surface it instead.
        if debug:
            import traceback

            return JSONResponse(
                status_code=500,
                content={
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc().splitlines()[-25:],
                    "root": str(ROOT),
                    "snapshot_dir_exists": (ROOT / "data" / "snapshot").exists(),
                    "files_here": sorted(p.name for p in ROOT.iterdir())[:40]
                    if ROOT.exists()
                    else None,
                },
            )
        raise


@app.get("/api/ledger")
def ledger_view() -> dict:
    try:
        return {"review": ledger.review(), "pending": [d.__dict__ for d in ledger.pending()]}
    except OSError as exc:
        return {"error": f"ledger unavailable: {exc}", "review": None, "pending": []}


class TradeIn(BaseModel):
    time: str
    asset: str
    direction: str
    result: Optional[str] = None
    pnl: Optional[float] = None
    payout: Optional[float] = None
    stake: Optional[float] = None
    setup: Optional[str] = None
    score: Optional[float] = None
    expiry: Optional[float] = None
    notes: Optional[str] = None


@app.post("/api/journal/import")
def import_journal(rows: List[TradeIn]) -> dict:
    """Accept manual entries or a pasted CSV and fold them into the journal.

    Duplicate counting is derived, not guessed: rows already present are the
    difference between what arrived and what the merged frame actually gained.
    The previous arithmetic reported 375 "duplicates" for a 3-row import.
    """
    if not rows:
        raise HTTPException(400, "no rows supplied")
    df = pd.DataFrame([r.dict(exclude_none=True) for r in rows])
    try:
        new = journal_mod.normalise(df, source="api")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    existing = journal_mod.from_csv(JOURNAL_CSV)
    merged = journal_mod.add_trades(existing, new)
    added = len(merged) - len(existing)
    try:
        journal_mod.to_csv(merged)
    except OSError as exc:
        # The rows parsed fine; there is simply nowhere to put them. Say that
        # plainly instead of returning a success the user cannot retrieve.
        raise HTTPException(
            503,
            f"Parsed {len(new)} row(s) but could not save the journal: {exc}. "
            "This deployment has a read-only filesystem, so trades cannot be "
            "persisted here. Run locally (./run.sh) or import the CSV directly "
            "into data/journal/trades.csv.",
        )
    _CACHE.pop("journal", None)
    return {
        "imported": len(new),
        "added": added,
        "duplicates_skipped": len(new) - added,
        "total": len(merged),
    }


@app.get("/api/journal")
def journal_report() -> dict:
    def produce() -> dict:
        df = journal_mod.from_csv(JOURNAL_CSV)
        if len(df) == 0:
            return {
                "empty": True,
                "message": (
                    "No trades yet. The journal is what makes this tool honest: "
                    "without it there is no way to know whether the rules are "
                    "actually profitable, only whether they feel profitable."
                ),
            }
        return journal_mod.report(df)
    return cached("journal", 30.0, produce)


@app.get("/api/backtest")
def backtest(
    symbol: str = Query("EUR/USD OTC"),
    period: str = Query("5d"),
    warmup: int = Query(60),
    bars: int = Query(4000),
    min_win_rate: float = Query(0.57),
    min_trades: int = Query(300),
) -> dict:
    """Score every historical bar and grade the result honestly.

    Defaults to a slice rather than full history because the blueprint budgets
    20 seconds end to end, and an honest 'no edge' serves a trader better than a
    slow one. The coin-flip baseline is included deliberately: if 'always CALL'
    scores 52% on the sample, then a 55% strategy is inside the noise.
    """
    def produce() -> dict:
        res = feed_mod.get_candles(symbol, "5m", period)
        if not res.usable:
            return {"error": res.error or "no data", "warnings": res.warnings}

        feats = compute_features(res.df)
        if len(feats) > bars:
            feats = feats.iloc[-bars:]

        found = pattern_scan(feats, warmup=warmup)
        if len(found) == 0:
            return {"error": "no setups found in this window", "warnings": res.warnings}

        payout = assets_mod.payout_for(symbol)
        expiry = orch().strategy.expiry_bars
        sst = analyse(feats)
        req = orch().strategy.required_setups()

        # Turn the raw setup frame into the signal frame `simulate` expects,
        # scoring every bar. Scoring each bar individually is what the live path
        # does too, so the backtest cannot flatter the scanner by using
        # information the live scanner would not have.
        rows = []
        for r in found.itertuples():
            sc = scoring_mod.score_setup(
                asset=symbol, feats=feats, structure=sst, direction=r.direction,
                setups_found=[r.name], required_setups=req, alignment=1.0,
                all_agree=False, payout=payout, expiry_bars=expiry, timeframe="5m",
                sigma_annual=None, is_visual=True,
            )
            rows.append({
                "bar": int(r.bar), "direction": r.direction,
                "expiry_bars": expiry, "score": float(sc.score), "setups": [r.name],
            })
        signals = pd.DataFrame(rows)

        trades, equity = backtest_mod.simulate(
            feats, signals, payout=payout, asset=symbol, timeframe="5m",
        )
        if not trades:
            return {"error": "no trades survived simulation", "warnings": res.warnings}

        result = backtest_mod.evaluate(
            trades, equity, strategy="all-setups", asset=symbol, timeframe="5m",
            payout=payout, starting_balance=10_000.0,
            min_win_rate=min_win_rate, min_trades=min_trades, in_sample=True,
        )
        base = backtest_mod.baseline_check(feats, expiry_bars=expiry, payout=payout)

        # Folded into the backtest response rather than exposed as a second
        # endpoint. A standalone route would re-run the whole pipeline for one
        # number, and the panel is only meaningful next to the win rate that
        # produced it -- the two must never come from different runs.
        sens = payout_mod.analyse_payout_sensitivity(
            win_rate=result.win_rate,
            n_trades=result.n_trades,
            effective_n=result.effective_n,
            ci_low=result.ci_low,
            ci_high=result.ci_high,
            # `assets.payout_for` returns a PERCENT (92.0); the sensitivity maths
            # is in fractions. `break_even_win_rate` raises on a percentage, which
            # is how this mismatch gets caught instead of silently returning a
            # 1% break-even.
            current_payout=payout / 100.0,
            target_wr=min_win_rate,
        )

        curve = equity.reset_index()
        curve.columns = ["time", "balance"]
        return {
            "symbol": symbol,
            "bars": len(feats),
            "setups": int(len(found)),
            "setups_by_type": {k: int(v) for k, v in found["name"].value_counts().items()},
            "trades": len(trades),
            "mean_score": float(signals["score"].mean()),
            "payout": payout,
            "result": result.to_dict(),
            "baseline": base,
            "payout_sensitivity": sens.to_dict(),
            "equity": [
                {"time": str(t), "balance": float(b)}
                for t, b in curve.itertuples(index=False)
            ][:: max(1, len(curve) // 300)],
            "warnings": res.warnings,
        }

    return cached(f"bt:{symbol}:{period}:{bars}", 300.0, produce)


@app.get("/")
def index() -> Any:
    f = WEB / "index.html"
    if not f.exists():
        return JSONResponse({"error": "web/index.html missing"}, status_code=500)
    return FileResponse(f)


if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")
