"""Market data adapter: one interface, several sources, and a disk cache.

The design rule here is that a feed must be allowed to be **absent**. PocketOption
OTC instruments have no public historical feed, so a scanner that insists on
real candles either lies or stalls. :func:`get_candles` returns a `FeedResult`
that carries a provenance tier and a `usable` flag, and every consumer in this
package is required to check it before treating the frame as tradable truth.

Cache layout::

    data/cache/<symbol>_<timeframe>.parquet   last successful fetch
    data/cache/<symbol>_<timeframe>.meta.json fetch time, rows, source
    data/snapshot/<symbol>_<timeframe>.csv.gz baked-in copy, committed to git

Parquet is preferred because the bar counts here (tens of thousands per asset)
re-read badly from CSV on every poll.

The snapshot directory is not a cache. It is a committed copy of real candles
kept so a deployed demo still has something real to compute on when the live
provider is unreachable -- which is the normal case from a serverless IP, since
Yahoo rate-limits and blocks datacenter ranges. Anything served from the
snapshot is labelled with its age and a warning. The whole point of this module
is that the dashboard can always say where its numbers came from; a silent
fallback to stale data would defeat it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .assets import FEED_OTC, FEED_PUBLIC, FEED_UNKNOWN, Asset, resolve

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"
SNAPSHOT_DIR = ROOT / "data" / "snapshot"

OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass
class FeedResult:
    asset: Optional[Asset]
    df: Optional[pd.DataFrame]
    source: str
    feed_tier: str
    fetched_at: float
    stale: bool = False
    from_cache: bool = False
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Can this be used to justify a trade?

        Requires a non-empty frame AND a tier we can actually reason about. A
        frame synthesised locally is explicitly not tradable evidence.
        """
        return self.df is not None and len(self.df) > 0 and self.feed_tier in (
            FEED_PUBLIC,
            "user",
        )

    @property
    def is_proxy(self) -> bool:
        """True when the frame stands in for an OTC instrument."""
        return bool(self.asset and self.asset.otc) and self.feed_tier == FEED_PUBLIC

    def to_dict(self) -> dict:
        return {
            "asset": self.asset.symbol if self.asset else None,
            "source": self.source,
            "feed_tier": self.feed_tier,
            "rows": 0 if self.df is None else len(self.df),
            "stale": self.stale,
            "from_cache": self.from_cache,
            "usable": self.usable,
            "is_proxy": self.is_proxy,
            "error": self.error,
            "warnings": self.warnings,
        }


def _cache_paths(symbol: str, timeframe: str) -> tuple:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in symbol)
    stem = CACHE_DIR / f"{safe}_{timeframe}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".meta.json")


def _read_cache(symbol: str, timeframe: str, max_age_s: float) -> tuple:
    pq, meta = _cache_paths(symbol, timeframe)
    if not meta.exists():
        return None, None
    try:
        info = json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError):
        return None, None
    age = time.time() - float(info.get("fetched_at", 0))
    if age > max_age_s:
        return None, info
    engine = info.get("engine", "csv")
    try:
        if engine == "parquet" and pq.exists():
            return pd.read_parquet(pq), info
        if pq.with_suffix(".csv.gz").exists():
            return pd.read_csv(pq.with_suffix(".csv.gz"), index_col=0, parse_dates=True), info
    except Exception:
        return None, info
    return None, info


def _write_cache(symbol: str, timeframe: str, df: pd.DataFrame, source: str) -> None:
    """Best-effort disk cache. Never raises.

    The whole point of a cache is that losing it is survivable, so every step is
    guarded -- including the ``mkdir``. An unguarded ``mkdir`` here is what
    turned a read-only serverless filesystem into a 500 on every scan, which is
    precisely the failure this function exists to prevent.
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    pq, meta = _cache_paths(symbol, timeframe)
    # Parquet is preferred but pyarrow/fastparquet are optional extras. Without
    # one, to_parquet raises and -- worse -- used to fail silently here, so the
    # cache appeared to work while never being written. Fall back to gzipped
    # CSV and record which engine was used so the choice is inspectable.
    engine = "none"
    try:
        df.to_parquet(pq)
        engine = "parquet"
    except Exception:
        try:
            df.to_csv(pq.with_suffix(".csv.gz"), compression="gzip")
            engine = "csv"
        except Exception:
            return
    try:
        meta.write_text(
            json.dumps(
                {
                    "fetched_at": time.time(),
                    "rows": len(df),
                    "source": source,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "engine": engine,
                },
                indent=2,
            )
        )
    except Exception:
        pass


def _snapshot_path(symbol: str, timeframe: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in symbol)
    return SNAPSHOT_DIR / f"{safe}_{timeframe}.csv.gz"


def read_snapshot(symbol: str, timeframe: str = "5m") -> Optional[FeedResult]:
    """Return the committed candle snapshot, or None if there isn't one."""
    path = _snapshot_path(symbol, timeframe)
    if not path.exists():
        return None
    try:
        df = _normalise(pd.read_csv(path, index_col=0, parse_dates=True))
    except Exception:
        return None
    if df.empty:
        return None
    meta_path = path.with_suffix("").with_suffix(".json")
    fetched_at = time.time()
    if meta_path.exists():
        try:
            fetched_at = float(json.loads(meta_path.read_text()).get("fetched_at", fetched_at))
        except Exception:
            pass
    age_h = (time.time() - fetched_at) / 3600.0
    return FeedResult(
        asset=None,
        df=df,
        source="yahoo:snapshot",
        feed_tier=FEED_PUBLIC,
        fetched_at=fetched_at,
        from_cache=True,
        warnings=[
            f"LIVE FEED UNREACHABLE - serving the committed snapshot from "
            f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(fetched_at))} "
            f"({age_h:.0f}h old, {len(df)} bars). These are real historical "
            f"candles, not live quotes, and the statistics below are therefore "
            f"indicative rather than current."
        ],
    )


def write_snapshot(symbol: str, timeframe: str, df: pd.DataFrame, fetched_at: float) -> Path:
    """Freeze the current cache into the committed snapshot directory."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(symbol, timeframe)
    df.to_csv(path, compression="gzip")
    path.with_suffix("").with_suffix(".json").write_text(
        json.dumps({"fetched_at": fetched_at, "rows": len(df), "symbol": symbol,
                    "timeframe": timeframe, "engine": "csv"}, indent=2)
    )
    return path


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce whatever the provider returned into a clean OHLCV frame."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=OHLCV)
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for c in OHLCV:
        if c not in df.columns:
            df[c] = np.nan
    if "adj close" in df.columns:
        df["adj_close"] = df["adj close"]
    if not isinstance(df.index, pd.DatetimeIndex):
        idx = None
        for cand in ("datetime", "date", "time", "timestamp"):
            if cand in df.columns:
                idx = pd.to_datetime(df[cand], utc=True, errors="coerce")
                break
        df = df.set_index(idx) if idx is not None else df
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    for c in OHLCV:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[OHLCV + (["adj_close"] if "adj_close" in df.columns else [])]


def fetch_public(
    yahoo_symbol: str,
    timeframe: str = "5m",
    period: str = "5d",
    force: bool = False,
) -> FeedResult:
    """Pull candles for a public market instrument.

    Yahoo is the default because it needs no key and covers FX, which is what
    the blueprint trades. Swap this function for another provider without
    touching callers -- the contract is a normalised frame plus provenance.
    """
    max_age = 120.0 if timeframe in ("1m", "2m", "5m") else 3600.0
    if not force:
        cached, info = _read_cache(yahoo_symbol, timeframe, max_age)
        if cached is not None and not cached.empty:
            return FeedResult(
                asset=None,
                df=cached,
                source="yahoo:cache",
                feed_tier=FEED_PUBLIC,
                fetched_at=float((info or {}).get("fetched_at", 0.0)),
                from_cache=True,
            )

    try:
        import yfinance as yf
    except ImportError as exc:
        snap = read_snapshot(yahoo_symbol, timeframe)
        if snap is not None:
            snap.warnings.append(f"yfinance unavailable ({exc}).")
            return snap
        return FeedResult(None, None, "none", FEED_UNKNOWN, time.time(),
                           error=f"yfinance unavailable: {exc}")

    try:
        raw = yf.download(
            yahoo_symbol,
            period=period,
            interval=timeframe,
            progress=False,
            auto_adjust=False,
            prepost=False,
        )
    except Exception as exc:
        snap = read_snapshot(yahoo_symbol, timeframe)
        if snap is not None:
            snap.warnings.append(f"Live fetch failed: {exc}.")
            return snap
        return FeedResult(None, None, "yahoo", FEED_UNKNOWN, time.time(),
                           error=f"fetch failed: {exc}")

    df = _normalise(raw)
    if df.empty:
        snap = read_snapshot(yahoo_symbol, timeframe)
        if snap is not None:
            snap.warnings.append(
                f"Live fetch returned no rows for {yahoo_symbol} {timeframe} {period}."
            )
            return snap
        return FeedResult(
            None, None, "yahoo", FEED_UNKNOWN, time.time(),
            error=f"no rows returned for {yahoo_symbol} {timeframe} {period}",
        )
    _write_cache(yahoo_symbol, timeframe, df, "yahoo")
    return FeedResult(None, df, "yahoo", FEED_PUBLIC, time.time())


def get_candles(
    symbol: str,
    timeframe: str = "5m",
    period: str = "5d",
    force: bool = False,
) -> FeedResult:
    """Resolve `symbol`, then fetch whatever feed we can honestly get for it.

    If the instrument is OTC and we only have a public proxy, the result comes
    back with ``is_proxy=True`` and a warning. Callers are expected to surface
    that rather than quietly score it.
    """
    asset = resolve(symbol)
    if asset is None or not asset.yahoo:
        return FeedResult(
            asset, None, "none", FEED_UNKNOWN, time.time(),
            error=f"unknown symbol {symbol!r}; no public feed mapped",
        )

    res = fetch_public(asset.yahoo, timeframe, period, force)
    res.asset = asset
    if res.df is not None and res.df.empty and res.asset.otc:
        res.feed_tier = FEED_OTC
    if asset.otc and res.feed_tier == FEED_PUBLIC:
        res.warnings.append(
            f"{asset.symbol}: candles are REAL MARKET {asset.yahoo}, not "
            f"PocketOption synthetic OTC. Treat statistics as a proxy only."
        )
    if asset.otc:
        res.warnings.append(
            "OTC expiry and payout behaviour cannot be validated from public candles."
        )
    return res


def load_user_csv(path: str, timeframe: str = "5m") -> FeedResult:
    """Load the trader's own exported bars.

    Accepted column names are intentionally forgiving because platform exports
    differ; anything unmapped is dropped and named in the warnings so a silent
    mis-parse cannot pass as good data.
    """
    p = Path(path)
    if not p.exists():
        return FeedResult(None, None, "user", "user", time.time(),
                           error=f"file not found: {path}")
    try:
        raw = pd.read_csv(p)
    except Exception as exc:
        return FeedResult(None, None, "user", "user", time.time(),
                           error=f"could not parse {path}: {exc}")
    df = _normalise(raw)
    if df.empty:
        return FeedResult(None, None, "user", "user", time.time(),
                           error=f"no usable OHLC rows in {path}")
    return FeedResult(resolve(p.stem), df, f"user:{p.name}", "user", time.time())


def get_many(
    symbols: List[str],
    timeframe: str = "5m",
    period: str = "5d",
) -> Dict[str, FeedResult]:
    """Sequential by design.

    yfinance rate-limits aggressively, and a thread pool here reliably gets the
    whole batch throttled. Latency is a secondary concern; data integrity is
    the primary one.
    """
    out: Dict[str, FeedResult] = {}
    for s in symbols:
        out[s] = get_candles(s, timeframe, period)
    return out
