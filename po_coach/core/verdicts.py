"""One vocabulary for every decision this system makes.

The bug this module exists to prevent: somewhere upstream, an unparseable or
absent input becomes a default, and the default looks like a decision. In
TradingAgents that bug was live enough to be filed as #1170 -- a decision nobody
could read silently degraded to "Hold", and the next run quoted back that Hold
as a call that was never made. This project has been bitten the same way four
times while the edge maths was being written (percent-vs-fraction payout
confusion), and once more in the journal (`test_edge` handed a break-even rate
where a payout was expected, so a genuine 5.5pp edge reported p=1.0).

So: every stage emits a `Verdict` from this one scale, and the *absence* of a
decision is represented explicitly as `REVIEW` or `NO_TRADE` -- never as a
directional call. Directional values are opt-in, never defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Ordered most-permissive to most-restrictive. `REVIEW` is not a position; it
# means "a human must look at this", and it must never be converted into one.
REVIEW = "REVIEW"
NO_TRADE = "NO_TRADE"
CALL = "CALL"
PUT = "PUT"

DIRECTIONAL = (CALL, PUT)
TRADEABLE = (CALL, PUT)

_RANK = {REVIEW: 0, NO_TRADE: 1, CALL: 2, PUT: 2}


@dataclass
class Verdict:
    """A decision plus the reason it is not tradeable, when it isn't."""

    value: str = REVIEW
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0        # 0..1, the model's own certainty
    source: str = "unknown"        # which component produced this

    def __post_init__(self) -> None:
        v = str(self.value).strip().upper()
        if v not in _RANK:
            raise ValueError(
                f"unknown verdict {self.value!r}; expected one of "
                f"{sorted(_RANK)}"
            )
        self.value = v

    @property
    def is_tradeable(self) -> bool:
        return self.value in TRADEABLE

    @property
    def direction(self) -> Optional[str]:
        return self.value.lower() if self.is_tradeable else None

    def downgrade(self, value: str, reason: str) -> "Verdict":
        """Restrict a verdict, never widen it.

        Downgrade-only is deliberate. A later stage may veto a trade but must not
        be able to promote a vetoed one back to directional -- that asymmetry is
        what makes composing these stages safe.
        """
        if _RANK[value] > _RANK[self.value]:
            return self
        return Verdict(value, self.reasons + [reason], self.confidence, self.source)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "direction": self.direction,
            "tradeable": self.is_tradeable,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "reasons": self.reasons,
        }


def from_score(score: float, threshold: float, direction: str,
               reasons: Optional[List[str]] = None, source: str = "scoring") -> Verdict:
    """Map a 0-100 score plus a threshold onto the vocabulary.

    A missing or non-finite score yields REVIEW, not a directional call. Scoring
    a setup is exactly the place where a NaN used to sail through.
    """
    import math

    if score is None or not isinstance(score, (int, float)) or not math.isfinite(score):
        return Verdict(REVIEW, ["score was missing or non-finite"], 0.0, source)
    if score < threshold:
        return Verdict(NO_TRADE, [f"score {score:.1f} < {threshold:.0f}"], 0.0, source)
    d = str(direction).strip().lower()
    if d not in ("call", "put"):
        return Verdict(REVIEW, [f"unusable direction {direction!r}"], 0.0, source)
    return Verdict(d.upper(), list(reasons or []), min(1.0, score / 100.0), source)


def compose(*verdicts: Verdict) -> Verdict:
    """Combine stages by strictest-wins.

    REVIEW outranks NO_TRADE: if any stage could not form an opinion, the honest
    result is "a human must look", not "there is no trade". Those are different
    failure modes and the dashboard shows them differently.
    """
    present = [v for v in verdicts if v is not None]
    if not present:
        return Verdict(REVIEW, ["no stage produced a verdict"], 0.0, "compose")
    strictest = min(present, key=lambda v: _RANK[v.value])
    reasons: List[str] = []
    for v in present:
        if v is strictest or _RANK[v.value] == _RANK[strictest.value]:
            reasons.extend(v.reasons)
    return Verdict(
        strictest.value,
        reasons,
        min(v.confidence for v in present),
        "+".join(sorted({v.source for v in present})),
    )


def format_verdict(v: Verdict) -> str:
    head = v.value
    if v.reasons:
        head += "  (" + "; ".join(v.reasons[:3]) + ")"
    return head
