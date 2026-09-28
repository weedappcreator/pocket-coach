"""Freeze the current candle cache into data/snapshot/ so the demo survives
a blocked network.

Run this after a good local scan:

    ./.venv/bin/python -m scripts.build_snapshot
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from po_coach.data.assets import ASSETS
from po_coach.data.feed import CACHE_DIR, _read_cache, write_snapshot

ok = skipped = 0
for a in ASSETS:
    if not a.yahoo:
        continue
    df, info = _read_cache(a.yahoo, "5m", max_age_s=10**9)
    if df is None or df.empty:
        print(f"  skip  {a.symbol:18} no cache for {a.yahoo}")
        skipped += 1
        continue
    fetched = float((info or {}).get("fetched_at", 0)) or 0
    p = write_snapshot(a.yahoo, "5m", df, fetched)
    print(f"  ok    {a.symbol:18} {len(df):>6} bars  {p.stat().st_size/1024:.0f} KB")
    ok += 1
print(f"\nsnapshotted {ok}, skipped {skipped}")
