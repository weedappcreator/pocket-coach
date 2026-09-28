"""Decision ledger: record what the system said, then score it against reality.

This is the most valuable idea borrowed from TauricResearch/TradingAgents
(`tradingagents/decision_log.py`), adapted to a system that must run without an
LLM.

The journal on its own only contains trades the trader *took*. That is a biased
sample: it cannot answer "what did the scanner recommend and I skipped, and was
it right to skip?" -- which is the question that determines whether the scanner
has any value at all. A tool that only records its hits will look brilliant
forever while quietly missing the best setups in the sample.

So every decision is logged *before* its outcome is known, tagged PENDING with
the reasons and the score that produced it. When the trade resolves, or the
expiry window passes, the entry is rewritten in place with the outcome and a
one-line reflection. Rewriting rather than appending keeps a decision's
reasoning next to its result, which is the only way to audit whether the
reasoning or the outcome was wrong.

Storage is append-only JSON Lines. Append-only matters: a ledger that can be
silently rewritten cannot be trusted to review itself, and a corrupted line is
skipped rather than fatal.

That storage is **ephemeral by default**. A serverless bundle mounts the project
read-only, so every write path here is guarded. When the ledger cannot be
persisted, decisions fall back to a process-local buffer and
:func:`persistence` reports ``False`` so the API can tell the user their history
is not being kept. The alternative -- crashing the scan, or silently pretending
to have saved -- is worse than both, because the trader would then trust a
recommendation record that does not exist.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .verdicts import NO_TRADE, REVIEW, Verdict

ROOT = Path(__file__).resolve().parents[2]
LEDGER_DIR = ROOT / "data" / "decisions"
LEDGER_PATH = LEDGER_DIR / "ledger.jsonl"

PENDING = "PENDING"
RESOLVED = "RESOLVED"
EXPIRED = "EXPIRED"

# A decision nobody acted on expires rather than hanging around as PENDING
# forever. The trader is not obliged to take every suggestion, and an
# unresolvable suggestion is a sign the scanner asked for something the risk
# engine was never going to allow.
DEFAULT_TTL_S = 900.0  # 15 minutes covers a 5-minute expiry plus slippage

# Decisions recorded while the ledger file is unwritable. Scoped to the process,
# so on serverless this lasts exactly as long as the warm container.
_EPHEMERAL: List["Decision"] = []
_PERSISTENT: Optional[bool] = None


def persistence(path: Optional[Path] = None) -> bool:
    """True when decisions are actually reaching disk, not just memory."""
    global _PERSISTENT
    if _PERSISTENT is not None:
        return _PERSISTENT
    p = Path(path) if path else LEDGER_PATH
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a"):
            pass
        _PERSISTENT = True
    except OSError:
        _PERSISTENT = False
    return _PERSISTENT


@dataclass
class Decision:
    ts: float
    asset: str
    verdict: str
    score: float = 0.0
    direction: Optional[str] = None
    setups: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    payout: Optional[float] = None
    suggested_stake: Optional[float] = None
    decision_id: str = ""
    status: str = PENDING
    outcome: Optional[str] = None        # "win" | "loss" | "skipped" | "expired"
    pnl: Optional[float] = None
    resolved_at: Optional[float] = None
    reflection: str = ""

    @property
    def key(self) -> str:
        return self.decision_id or f"{self.asset}@{self.ts:.3f}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "Decision":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _next_id(ts: float, asset: str) -> str:
    return f"{asset.replace('/', '').replace(' ', '')}-{ts:.0f}"


def record(
    verdict: Verdict,
    asset: str,
    score: float = 0.0,
    setups: Optional[List[str]] = None,
    payout: Optional[float] = None,
    suggested_stake: Optional[float] = None,
    path: Optional[Path] = None,
) -> Decision:
    """Write a decision to the ledger immediately, before any outcome exists."""
    p = Path(path) if path else LEDGER_PATH
    ts = time.time()
    d = Decision(
        ts=ts,
        asset=asset,
        verdict=verdict.value,
        score=round(float(score), 2),
        direction=verdict.direction,
        setups=list(setups or []),
        reasons=list(verdict.reasons),
        vetoes=list(verdict.reasons) if verdict.value in (NO_TRADE, REVIEW) else [],
        payout=payout,
        suggested_stake=suggested_stake,
        decision_id=_next_id(ts, asset),
        status=PENDING,
    )
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(d.to_json() + "\n")
    except OSError:
        # Read-only bundle. Keep the decision where the API can still show it,
        # and remember that this instance is not persisting anything.
        global _PERSISTENT
        _PERSISTENT = False
        _EPHEMERAL.append(d)
    return d


def load(path: Optional[Path] = None) -> List[Decision]:
    """Read the ledger, plus anything recorded while disk was unavailable.

    A truncated final line is skipped, not fatal, and an unreadable file yields
    only the in-memory decisions rather than raising.
    """
    p = Path(path) if path else LEDGER_PATH
    if not p.exists():
        return list(_EPHEMERAL)
    out: List[Decision] = []
    try:
        text = p.read_text()
    except OSError:
        return list(_EPHEMERAL)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Decision.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    known = {d.decision_id for d in out}
    out.extend(d for d in _EPHEMERAL if d.decision_id not in known)
    return out


def _atomic_write(decisions: List[Decision], p: Path) -> None:
    """Rewrite via temp file + rename.

    Resolution rewrites the whole file, so a crash mid-write would otherwise
    destroy the entire ledger -- the one artefact whose loss is unrecoverable.

    Every step is inside the guard on purpose. `tempfile.mkstemp` raising
    `PermissionError` on a read-only bundle is an ``OSError`` like any other,
    and an unguarded call to it turned a sweep into a 500.
    """
    tmp: Optional[str] = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            for d in decisions:
                fh.write(d.to_json() + "\n")
        os.replace(tmp, p)
    except OSError:
        # Nothing changed on disk; keep the buffer so the decision is still
        # visible, and let the caller carry on with a degraded ledger.
        global _PERSISTENT
        _PERSISTENT = False
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def resolve(
    decision_id: str,
    outcome: str,
    pnl: Optional[float] = None,
    reflection: str = "",
    path: Optional[Path] = None,
) -> Optional[Decision]:
    """Mark one PENDING decision resolved, in place."""
    p = Path(path) if path else LEDGER_PATH
    decisions = load(p)
    hit: Optional[Decision] = None
    for d in decisions:
        if d.decision_id == decision_id and d.status == PENDING:
            d.status = RESOLVED
            d.outcome = outcome
            d.pnl = pnl
            d.reflection = reflection
            d.resolved_at = time.time()
            hit = d
            break
    if hit is not None:
        _atomic_write(decisions, p)
    return hit


def sweep_stale(ttl_s: float = DEFAULT_TTL_S, path: Optional[Path] = None) -> List[Decision]:
    """Expire PENDING decisions older than the TTL.

    These are setups the trader did not take. Recording them as `expired` rather
    than deleting them is what makes the recommendation record honest: the
    scanner's hit rate is then measurable, and a scanner that only ever records
    what was taken cannot be evaluated at all.
    """
    p = Path(path) if path else LEDGER_PATH
    decisions = load(p)
    now = time.time()
    changed: List[Decision] = []
    for d in decisions:
        if d.status == PENDING and now - d.ts > ttl_s:
            d.status = EXPIRED
            d.outcome = "expired"
            d.resolved_at = now
            d.reflection = "not actioned by the trader within the expiry window"
            changed.append(d)
    if changed:
        _atomic_write(decisions, p)
    return changed


def pending(path: Optional[Path] = None) -> List[Decision]:
    return [d for d in load(path) if d.status == PENDING]


def review(path: Optional[Path] = None) -> Dict[str, object]:
    """Score the scanner against reality.

    Split on whether the system was tradeable, because those are different
    failure modes: a tradeable call that lost is a prediction error, while a
    NO_TRADE is a decision that cost nothing. Conflating them makes a cautious
    system look skilful.
    """
    decisions = [d for d in load(path) if d.status in (RESOLVED, EXPIRED)]
    tradeable = [d for d in decisions if d.verdict in ("CALL", "PUT")]
    blocked = [d for d in decisions if d.verdict in (NO_TRADE, REVIEW)]

    acted = [d for d in tradeable if d.outcome in ("win", "loss")]
    wins = sum(1 for d in acted if d.outcome == "win")
    losses = len(acted) - wins

    missed = [d for d in tradeable if d.outcome == "expired"]

    return {
        "persistent": persistence(path),
        "total": len(decisions),
        "tradeable_signals": len(tradeable),
        "blocked": len(blocked),
        "acted": len(acted),
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / len(acted)) if acted else None,
        "not_actioned": len(missed),
        "not_actioned_rate": (len(missed) / len(tradeable)) if tradeable else None,
        "expired": sum(1 for d in decisions if d.status == EXPIRED),
        "reflection": sorted(
            (d.reflection for d in decisions if d.reflection),
            key=lambda s: s,
        )[:10],
    }
