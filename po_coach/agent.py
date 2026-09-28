"""The orchestrator: raw candles in, one auditable verdict out.

Deliberately **not** an LLM. Tauric's TradingAgents runs a multi-agent LangGraph
debate between bull and bear researchers; that is the right shape when a large
language model has to read news prose and form a qualitative view. It is the
wrong shape here, for three concrete reasons:

* The blueprint budgets 20 seconds per signal. Several rounds of LLM calls do not
  fit in that, and the result would not be reproducible run to run.
* Every threshold in this system is a number the trader already stated. An LLM
  adds variance to a decision that should be arithmetic.
* A verdict has to be auditable after the fact. "the model said so" is not an
  audit trail; "55.4% modelled win rate vs 52.1% break-even, gated out by a 3pp
  minimum" is.

The agent vocabulary is therefore a *division of labour* between deterministic
components, not a conversation between personas. The six role documents in
`po_coach/agents/` describe what each stage is responsible for; this module is
where those responsibilities are actually wired together.

The ordering below is the whole design in one function: every stage may refuse,
and no stage may promote. A later stage can only ever tighten the verdict, which
is what `verdicts.Verdict.downgrade` encodes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .core import decisions as ledger
from .core import patterns, probability, risk as risk_mod, scoring, verdicts
from .core.indicators import compute_features
from .core.structure import analyse, timeframe_alignment
from .data import assets as assets_mod
from .data import feed as feed_mod

ROOT = Path(__file__).resolve().parents[1]   # project root, not the package dir
CONFIG_DIR = ROOT / "config"

_STAGES = (
    ("data", "testing-evidence-collector"),
    ("structure", "finance-financial-analyst"),
    ("patterns", "finance-financial-analyst"),
    ("probability", "academic-statistician"),
    ("scoring", "academic-statistician"),
    ("risk", "testing-reality-checker"),
    ("verdict", "support-analytics-reporter"),
)


@dataclass
class Strategy:
    """Parsed strategy YAML, with the weight-sum check the blueprint demands."""

    raw: dict

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Strategy":
        import yaml

        p = Path(path) if path else CONFIG_DIR / "kevin.yaml"
        with p.open() as fh:
            data = yaml.safe_load(fh)
        s = cls(data or {})
        s.validate()
        return s

    def validate(self) -> None:
        w = self.raw.get("scoring", {}).get("weights", {})
        total = sum(float(v) for v in w.values()) if w else 0.0
        if not w:
            raise ValueError("strategy has no scoring weights")
        if abs(total - 1.0) > 1e-6:
            # A weight set that does not normalise to 1.0 silently rescales every
            # score, which moves the 70 threshold without anyone changing it.
            raise ValueError(f"scoring weights sum to {total:.4f}, must sum to 1.0")

    def get(self, *keys, default=None):
        node = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    @property
    def assets(self) -> List[str]:
        return list(self.get("markets", "assets", default=[]) or [])

    @property
    def timeframe(self) -> str:
        return self.get("markets", "timeframe", default="5m")

    @property
    def expiry_bars(self) -> int:
        return int(self.get("markets", "expiry_bars", default=5))

    def scoring_config(self) -> scoring.ScoringConfig:
        cfg = scoring.ScoringConfig()
        w = self.get("scoring", "weights", default={}) or {}
        for k, v in w.items():
            if hasattr(cfg, k):
                setattr(cfg, k, float(v))
        for k in (
            "min_score", "min_score_visual", "min_rsi_oversold", "max_rsi_overbought",
            "min_chop_below", "min_atr_pct", "max_atr_pct", "min_vol_ratio",
            "max_vol_ratio", "max_zone_distance_pct", "adx_trend_min",
        ):
            v = self.get("scoring", k)
            if v is not None:
                setattr(cfg, k, float(v))
        cfg.in_sample = bool(self.get("scoring", "in_sample", default=True))
        return cfg

    def filters(self) -> scoring.Filters:
        f = scoring.Filters()
        for k, v in (self.get("filters", default={}) or {}).items():
            if hasattr(f, k):
                setattr(f, k, type(getattr(f, k))(v))
        return f

    def risk_config(self) -> risk_mod.RiskConfig:
        rc = risk_mod.RiskConfig()
        bankroll = 10000.0
        for k, v in (self.get("risk", default={}) or {}).items():
            if k == "bankroll":
                bankroll = float(v)
            elif hasattr(rc, k):
                cur = getattr(rc, k)
                setattr(rc, k, type(cur)(v) if cur is not None else v)
        self._bankroll = bankroll
        return rc

    def required_setups(self) -> List[str]:
        return list(self.get("setups", "required_any", default=[]) or [])


@dataclass
class Candidate:
    """One scored setup plus the reasoning that produced its verdict."""

    asset: str
    direction: str
    score: float
    setups: List[str] = field(default_factory=list)
    verdict: verdicts.Verdict = field(default_factory=verdicts.Verdict)
    risk: Optional[dict] = None
    edge: Optional[dict] = None
    warnings: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    price: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "direction": self.direction,
            "score": round(self.score, 1),
            "setups": self.setups,
            "verdict": self.verdict.to_dict(),
            "risk": self.risk,
            "edge": self.edge,
            "warnings": self.warnings,
            "reasons": self.reasons,
            "vetoes": self.vetoes,
            "price": self.price,
        }


@dataclass
class ScanReport:
    generated_at: float
    duration_ms: int
    strategy: str
    assets: List[dict] = field(default_factory=list)
    candidates: List[dict] = field(default_factory=list)
    best: Optional[dict] = None
    stages: List[dict] = field(default_factory=list)
    provenance: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "duration_ms": self.duration_ms,
            "strategy": self.strategy,
            "assets": self.assets,
            "candidates": self.candidates,
            "best": self.best,
            "stages": self.stages,
            "provenance": self.provenance,
        }


class Orchestrator:
    def __init__(self, strategy: Optional[Strategy] = None,
                 bankroll: float = 10000.0, log_decisions: bool = True):
        self.strategy = strategy or Strategy.load()
        self.cfg = self.strategy.scoring_config()
        self.filters = self.strategy.filters()
        self.rcfg = self.strategy.risk_config()
        self.bankroll = bankroll
        self.risk = risk_mod.RiskEngine(self.rcfg, bankroll=bankroll)
        self.log_decisions = log_decisions
        self.stages = [{"stage": s, "agent": a} for s, a in _STAGES]

    # -- stage 1: data ---------------------------------------------------
    def _load(self, symbol: str, period: str = "5d") -> Dict:
        res = feed_mod.get_candles(symbol, self.strategy.timeframe, period)
        a = assets_mod.resolve(symbol)
        return {
            "symbol": symbol,
            "asset": a.symbol if a else symbol,
            "ok": res.usable,
            "rows": 0 if res.df is None else len(res.df),
            "feed_tier": res.feed_tier,
            "is_proxy": res.is_proxy,
            "source": res.source,
            "from_cache": res.from_cache,
            "payout": assets_mod.payout_for(symbol),
            "warnings": res.warnings,
            "error": res.error,
            "_res": res,
        }

    # -- stages 2-4: structure, patterns, probability --------------------
    def _analyse(self, meta: Dict) -> Optional[Dict]:
        df = meta["_res"].df
        if df is None or len(df) < 120:
            meta["error"] = meta["error"] or f"need >=120 bars, have {0 if df is None else len(df)}"
            return None
        feats = compute_features(df)
        sst = analyse(feats)
        setups = patterns.detect_latest(feats, sst.swings)
        sigma = probability.sigma_from_candles(feats, 200, self.strategy.timeframe)
        plaus = probability.vol_plausibility(sigma, "fx")
        meta.update({
            "price": float(feats["close"].iloc[-1]),
            "trend": sst.trend,
            "bias": sst.bias,
            "rsi": float(feats["rsi"].iloc[-1]),
            "atr_pct": float(feats["atr_pct"].iloc[-1]),
            "chop": float(feats["chop"].iloc[-1]),
            "adx": float(feats["adx"].iloc[-1]),
            "sigma_annual": sigma,
            "vol_plausible": plaus.get("ok"),
            "vol_warning": plaus.get("warning"),
            "setups": [s.name for s in setups],
            "setup_details": [s.to_dict() for s in setups],
        })
        meta["_feats"] = feats
        meta["_sst"] = sst
        if not plaus.get("ok", True):
            meta["warnings"].append(
                f"volatility {sigma*100:.1f}%/yr outside the FX range; "
                "statistics will be unreliable"
            )
        return meta

    # -- stage 5-6: score then risk --------------------------------------
    def _candidates(self, metas: List[Dict]) -> List[Candidate]:
        out: List[Candidate] = []
        required = self.strategy.required_setups()
        expiry = self.strategy.expiry_bars
        tf = self.strategy.timeframe

        for meta in metas:
            if meta.get("error") or "_feats" not in meta:
                continue
            if meta.get("session_ok") is False:
                # The blueprint's kill-zone rule is a hard filter, not advice.
                # The asset still appears in provenance with session_ok=False so
                # the dashboard can explain the silence instead of looking broken.
                continue
            found = meta.get("setups", [])
            if not found:
                continue
            for direction in ("call", "put"):
                scored = scoring.score_setup(
                    asset=meta["asset"],
                    feats=meta["_feats"],
                    structure=meta["_sst"],
                    direction=direction,
                    setups_found=found,
                    required_setups=required,
                    alignment=1.0,
                    all_agree=False,
                    payout=meta["payout"],
                    expiry_bars=expiry,
                    timeframe=tf,
                    cfg=self.cfg,
                    filters=self.filters,
                    sigma_annual=meta["sigma_annual"],
                    is_visual=True,
                    news_block=False,
                )
                if scored.vetoes:
                    v = verdicts.Verdict(
                        verdicts.NO_TRADE, list(scored.vetoes), 0.0, "scoring"
                    )
                else:
                    v = verdicts.from_score(
                        scored.score,
                        self.cfg.min_score_visual if scored.is_visual else self.cfg.min_score,
                        direction,
                        scored.reasons,
                        "scoring",
                    )
                cand = Candidate(
                    asset=meta["asset"], direction=direction, score=scored.score,
                    setups=found, verdict=v, warnings=list(scored.warnings),
                    reasons=list(scored.reasons), vetoes=list(scored.vetoes),
                    price=scored.price,
                    edge=scored.edge.to_dict() if scored.edge else None,
                )
                cand.risk = self._risk_for(cand, meta)
                cand.verdict = verdicts.compose(v, self._risk_verdict(cand, meta))
                out.append(cand)
        out.sort(key=lambda c: c.score, reverse=True)
        return out[: int(self.strategy.get("scan", "top_n", default=5))]

    def _risk_for(self, cand: Candidate, meta: Dict) -> Optional[dict]:
        p = cand.edge.get("p_technical") if cand.edge else None
        decision = self.risk.evaluate(
            score=cand.score,
            payout=meta["payout"],
            win_rate=p,
            conviction=min(1.0, cand.score / 100.0),
            is_visual=True,
        )
        return decision.to_dict()

    @staticmethod
    def _risk_verdict(cand: Candidate, meta: Dict) -> verdicts.Verdict:
        r = cand.risk
        if not r:
            return verdicts.Verdict(verdicts.REVIEW, ["risk engine did not run"], 0.0, "risk")
        if r.get("approved"):
            stake = r.get("stake_pct")
            reasons = [f"risk approved at {stake:.2f}% of bankroll"] if stake is not None else ["risk approved"]
            return verdicts.Verdict(
                cand.direction.upper(), reasons,
                min(1.0, cand.score / 100.0), "risk",
            )
        vetoes = r.get("vetoes") or [r.get("reason", "risk veto")]
        return verdicts.Verdict(verdicts.NO_TRADE, list(vetoes), 0.0, "risk")

    # -- entry point -----------------------------------------------------
    def scan(self, symbols: Optional[List[str]] = None, period: str = "5d",
             write_ledger: Optional[bool] = None) -> ScanReport:
        t0 = time.time()
        syms = symbols or self.strategy.assets
        tf = self.strategy.timeframe

        metas: List[Dict] = []
        provenance: List[dict] = []
        for s in syms:
            meta = self._load(s, period)
            asset = assets_mod.resolve(s)
            if asset and asset.otc and meta["ok"]:
                hour = pd.Timestamp.utcnow().hour
                kz = bool(self.strategy.get("markets", "killzone_only", default=True))
                meta["session_ok"] = assets_mod.session_filter_ok(s, hour, killzone_only=kz)
                if not meta["session_ok"]:
                    meta["warnings"].append(
                        f"outside the London/NY overlap ({hour:02d}:00 UTC); no signals ranked"
                    )
            self._analyse(meta)
            provenance.append({k: v for k, v in meta.items() if not k.startswith("_")})
            metas.append(meta)

        cands = self._candidates(metas)
        tradeable = [c for c in cands if c.verdict.is_tradeable]
        best = tradeable[0].to_dict() if tradeable else None

        should_log = self.log_decisions if write_ledger is None else write_ledger
        if should_log:
            for c in cands:
                meta = next((m for m in metas if m["asset"] == c.asset), {})
                try:
                    ledger.record(
                        verdict=c.verdict, asset=c.asset, score=c.score,
                        setups=c.setups, payout=meta.get("payout"),
                        suggested_stake=(c.risk or {}).get("stake"),
                    )
                except OSError:
                    # A read-only filesystem (serverless) must not break a scan.
                    pass

        return ScanReport(
            generated_at=time.time(),
            duration_ms=int((time.time() - t0) * 1000),
            strategy=str(self.strategy.raw.get("name", "unnamed")),
            assets=[{k: v for k, v in m.items() if not k.startswith("_")} for m in metas],
            candidates=[c.to_dict() for c in cands],
            best=best,
            stages=self.stages,
            provenance=provenance,
        )
