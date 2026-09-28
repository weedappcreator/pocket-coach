"""Setup scoring (0-100) and the hard no-trade filters.

The weights below are the blueprint's starting values, and the single most
important line in this file is the comment on `ScoringConfig.in_sample`: they
must be calibrated on one period and then verified on a later one, because the
blueprint's "recalibrate after the backtest" step, done naively, is exactly how
a 57% backtest becomes a 51% live result.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .probability import EdgeEstimate, estimate_edge, sigma_from_candles, vol_from_atr
from .structure import StructureRead, zone_distance_pct, zone_room_pct


@dataclass
class ScoringConfig:
    """Blueprint page 5, verbatim. Weights must sum to 1.0."""

    rules_match: float = 0.40
    timeframe_alignment: float = 0.20
    momentum_confirmation: float = 0.15
    key_level_context: float = 0.15
    volatility_fit: float = 0.10

    min_score: float = 70.0
    min_score_visual: float = 80.0

    # Component gates
    min_rsi_oversold: float = 30.0        # for calls
    max_rsi_overbought: float = 70.0      # for puts
    min_chop_below: float = 61.8          # below this = tradable trend
    min_atr_pct: float = 0.0
    max_atr_pct: float = 0.35             # outside = spiking or dead
    min_vol_ratio: float = 0.5
    max_vol_ratio: float = 2.5
    max_zone_distance_pct: float = 0.30   # "too close to next level" guard
    adx_trend_min: float = 20.0

    calibrated_on: Optional[str] = None
    in_sample: bool = True

    def weights_total(self) -> float:
        return (
            self.rules_match
            + self.timeframe_alignment
            + self.momentum_confirmation
            + self.key_level_context
            + self.volatility_fit
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoreComponent:
    name: str
    weight: float
    raw: float          # 0..1 quality
    points: float       # raw * weight * 100
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoredSetup:
    asset: str
    direction: str
    score: float
    components: List[ScoreComponent]
    edge: Optional[EdgeEstimate]
    setups_found: List[str]
    reasons: List[str]
    vetoes: List[str]
    warnings: List[str]
    timeframe: str
    expiry_bars: int
    price: float
    is_visual: bool = False

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "direction": self.direction,
            "score": round(self.score, 1),
            "components": [c.to_dict() for c in self.components],
            "edge": self.edge.to_dict() if self.edge else None,
            "setups_found": self.setups_found,
            "reasons": self.reasons,
            "vetoes": self.vetoes,
            "warnings": self.warnings,
            "timeframe": self.timeframe,
            "expiry_bars": self.expiry_bars,
            "price": self.price,
            "is_visual": self.is_visual,
        }


# --------------------------------------------------------------------------
# Component scorers -- each returns 0..1
# --------------------------------------------------------------------------

def score_rules_match(setups_found: List[str], required: List[str]) -> Tuple[float, str]:
    """40% weight. Confluence: how many of the trader's entry conditions are met."""
    if not required:
        return 0.5, "no specific setups required"
    hit = [s for s in required if s in setups_found]
    raw = len(hit) / float(len(required))
    if len(setups_found) > len(hit):
        raw = min(1.0, raw + 0.1 * (len(setups_found) - len(hit)))
    detail = "%d/%d setups matched%s" % (
        len(hit), len(required),
        (": " + ", ".join(setups_found)) if setups_found else "",
    )
    return raw, detail


def score_timeframe_alignment(alignment: float, all_agree: bool) -> Tuple[float, str]:
    raw = min(1.0, abs(alignment))
    if not all_agree:
        raw *= 0.4
    return raw, "alignment %+.2f%s" % (alignment, "" if all_agree else " (timeframes disagree)")


def score_momentum(
    rsi_value: float, ema_slope: float, stoch_k: float, direction: str, cfg: ScoringConfig
) -> Tuple[float, str]:
    """15% weight. RSI + EMA slope + Stochastic all leaning the same way."""
    if any(pd.isna(v) for v in (rsi_value, ema_slope, stoch_k)):
        return 0.5, "momentum indicators not warm yet"
    parts = []
    if direction == "call":
        rsi_q = min(1.0, max(0.0, (55.0 - rsi_value) / 25.0))
        slope_q = min(1.0, max(0.0, ema_slope / 0.05))
        stoch_q = min(1.0, max(0.0, (50.0 - stoch_k) / 40.0))
        parts = [rsi_q, slope_q, stoch_q]
        detail = "RSI %.0f, EMA slope %+.3f%%, Stoch %.0f" % (rsi_value, ema_slope, stoch_k)
    else:
        rsi_q = min(1.0, max(0.0, (rsi_value - 45.0) / 25.0))
        slope_q = min(1.0, max(0.0, -ema_slope / 0.05))
        stoch_q = min(1.0, max(0.0, (stoch_k - 50.0) / 40.0))
        parts = [rsi_q, slope_q, stoch_q]
        detail = "RSI %.0f, EMA slope %+.3f%%, Stoch %.0f" % (rsi_value, ema_slope, stoch_k)
    return float(np.mean(parts)), detail


def score_key_levels(
    distance_pct: float, direction: str, structure_bias: float, cfg: ScoringConfig
) -> Tuple[float, str]:
    """15% weight. Room to move before expiry, plus trend agreement."""
    if not np.isfinite(distance_pct):
        lvl_raw, lvl_detail = 0.5, "no nearby zone"
    elif distance_pct < 0.02:
        lvl_raw, lvl_detail = 0.15, "sitting on a level (%.3f%% away)" % distance_pct
    elif distance_pct < cfg.max_zone_distance_pct:
        lvl_raw = 1.0 - (distance_pct / cfg.max_zone_distance_pct) * 0.4
        lvl_detail = "%.3f%% of clear space" % distance_pct
    else:
        lvl_raw = 0.7
        lvl_detail = "%.3f%% to next zone" % distance_pct
    bias_agrees = (structure_bias > 0.1 and direction == "call") or (
        structure_bias < -0.1 and direction == "put"
    )
    if abs(structure_bias) < 0.1:
        bias_raw, bias_detail = 0.4, "structure gives no directional edge"
    else:
        bias_raw = 1.0 if bias_agrees else 0.2
        bias_detail = "structure %s the trade" % ("supports" if bias_agrees else "opposes")
    return 0.6 * lvl_raw + 0.4 * bias_raw, "%s; %s" % (lvl_detail, bias_detail)


def score_volatility_fit(
    atr_pct: float, vol_ratio: float, chop: float, adx_value: float, cfg: ScoringConfig
) -> Tuple[float, str]:
    """10% weight. ATR in a tradeable band, not dead and not spiking."""
    if pd.isna(atr_pct) or pd.isna(vol_ratio):
        return 0.5, "volatility not warm yet"
    if atr_pct < cfg.min_atr_pct:
        v = 0.0
    elif atr_pct > cfg.max_atr_pct:
        v = 0.1
    else:
        span = cfg.max_atr_pct - max(cfg.min_atr_pct, 1e-6)
        mid = max(cfg.min_atr_pct, 1e-6) + span * 0.35
        v = 1.0 - abs(atr_pct - mid) / max(span, 1e-9) * 0.5
        v = float(np.clip(v, 0.0, 1.0))
    vr = 1.0
    if not pd.isna(vol_ratio):
        if vol_ratio < cfg.min_vol_ratio:
            vr = 0.2
        elif vol_ratio > cfg.max_vol_ratio:
            vr = 0.2
        else:
            vr = 1.0 - abs(vol_ratio - 1.0) / 1.0 * 0.4
    ch = 1.0
    if not pd.isna(chop):
        ch = float(np.clip((cfg.min_chop_below - chop) / cfg.min_chop_below, 0.0, 1.0))
    ax = 0.5
    if not pd.isna(adx_value):
        ax = float(np.clip(adx_value / 40.0, 0.0, 1.0))
    raw = 0.35 * v + 0.25 * vr + 0.2 * ch + 0.2 * ax
    detail = "ATR %.3f%%, vol x%.2f, chop %.0f, ADX %.0f" % (atr_pct, vol_ratio, chop, adx_value)
    return float(np.clip(raw, 0.0, 1.0)), detail


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------

@dataclass
class Filters:
    min_payout: float = 80.0
    block_chop_above: float = 61.8
    block_atr_pct_above: float = 0.35
    block_vol_ratio_above: float = 2.5
    block_timeframe_conflict: bool = True
    block_zone_too_close: float = 0.05
    max_expiry_bars: int = 5
    news_blackout: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def apply_filters(
    feats: pd.DataFrame,
    structure: StructureRead,
    direction: str,
    alignment: float,
    all_agree: bool,
    payout: float,
    expiry_bars: int,
    f: Optional[Filters] = None,
    news_block: bool = False,
) -> List[str]:
    """Blueprint page 6, 'automatic no-trade conditions'. Any hit blocks the setup."""
    f = f or Filters()
    out: List[str] = []
    i = len(feats) - 1
    row = feats.iloc[i]

    if news_block:
        out.append("High-impact news inside the blackout window.")
    if payout < f.min_payout:
        out.append("Payout %.0f%% below the %.0f%% minimum." % (payout, f.min_payout))
    if f.block_timeframe_conflict and not all_agree and abs(alignment) < 0.34:
        out.append("Timeframes conflict (5/10/15-min disagree).")
    if not pd.isna(row.get("chop")) and row["chop"] > f.block_chop_above:
        out.append("Choppy market (chop index %.0f > %.0f)." % (row["chop"], f.block_chop_above))
    if not pd.isna(row.get("atr_pct")) and row["atr_pct"] > f.block_atr_pct_above:
        out.append("Volatility spike (ATR %.3f%% > %.2f%%)." % (row["atr_pct"], f.block_atr_pct_above))
    if not pd.isna(row.get("vol_ratio")) and row["vol_ratio"] > f.block_vol_ratio_above:
        out.append("Abnormal volume move (x%.2f)." % row["vol_ratio"])
    if expiry_bars > f.max_expiry_bars:
        out.append("Expiry %d bars exceeds the %d-bar cap." % (expiry_bars, f.max_expiry_bars))
    d = zone_room_pct(structure.zones, float(row["close"]), direction)
    if np.isfinite(d) and d < f.block_zone_too_close:
        out.append("Too close to a level (%.3f%%) -- no room." % d)
    return out


# --------------------------------------------------------------------------
# The scorer
# --------------------------------------------------------------------------

def score_setup(
    asset: str,
    feats: pd.DataFrame,
    structure: StructureRead,
    direction: str,
    setups_found: List[str],
    required_setups: Optional[List[str]],
    alignment: float,
    all_agree: bool,
    payout: float,
    expiry_bars: int,
    timeframe: str = "5m",
    cfg: Optional[ScoringConfig] = None,
    filters: Optional[Filters] = None,
    sigma_annual: Optional[float] = None,
    is_visual: bool = False,
    news_block: bool = False,
) -> ScoredSetup:
    """Score one candidate setup. `payout` is a PERCENTAGE (92 == 92%)."""
    cfg = cfg or ScoringConfig()
    row = feats.iloc[-1]
    price = float(row["close"])
    vetoes: List[str] = []
    warnings: List[str] = []

    # `payout` arrives in PERCENT (92 == 92%, matching the platform UI and
    # ScoringConfig/Filters thresholds). estimate_edge now takes a percent too,
    # so it converts internally -- see probability.payout_fraction.

    components: List[ScoreComponent] = []

    raw, detail = score_rules_match(setups_found, required_setups or [])
    components.append(ScoreComponent("trader rules match", cfg.rules_match, raw, raw * cfg.rules_match * 100, detail))

    raw, detail = score_timeframe_alignment(alignment, all_agree)
    components.append(ScoreComponent("timeframe alignment", cfg.timeframe_alignment, raw, raw * cfg.timeframe_alignment * 100, detail))

    raw, detail = score_momentum(
        float(row.get("rsi", np.nan)),
        float(row.get("ema_slope", np.nan)),
        float(row.get("stoch_k", np.nan)),
        direction, cfg,
    )
    components.append(ScoreComponent("momentum confirmation", cfg.momentum_confirmation, raw, raw * cfg.momentum_confirmation * 100, detail))

    dist = zone_room_pct(structure.zones, price, direction)
    raw, detail = score_key_levels(dist, direction, structure.bias, cfg)
    components.append(ScoreComponent("key-level context", cfg.key_level_context, raw, raw * cfg.key_level_context * 100, detail))

    raw, detail = score_volatility_fit(
        float(row.get("atr_pct", np.nan)),
        float(row.get("vol_ratio", np.nan)),
        float(row.get("chop", np.nan)),
        float(row.get("adx", np.nan)),
        cfg,
    )
    components.append(ScoreComponent("volatility fit", cfg.volatility_fit, raw, raw * cfg.volatility_fit * 100, detail))

    score = sum(c.points for c in components)

    if sigma_annual is None:
        # Prefer the log-return estimator; fall back to ATR if there is not
        # enough history for it.
        sigma_annual = sigma_from_candles(feats, 200, timeframe)
        if sigma_annual <= 0:
            sigma_annual = vol_from_atr(float(row.get("atr", np.nan)), price, timeframe)

    edge = estimate_edge(
        spot=price,
        payout=payout,
        direction=direction,
        sigma_annual=sigma_annual,
        expiry_bars=expiry_bars,
        timeframe=timeframe,
        technical_bias=structure.bias,
    )

    vetoes.extend(
        apply_filters(
            feats, structure, direction, alignment, all_agree, payout, expiry_bars, filters, news_block
        )
    )

    threshold = cfg.min_score_visual if is_visual else cfg.min_score
    if score < threshold:
        vetoes.append("Score %.0f below the %.0f threshold." % (score, threshold))

    if edge.verdict == "REJECT":
        vetoes.append("Negative EV: %s" % edge.notes[-1])
    elif edge.verdict == "MARGINAL":
        warnings.append("Thin edge, below the preferred floor.")
    if edge.verdict == "REJECT" and "coin flip" in " ".join(edge.notes):
        vetoes.append("No directional edge: this is a coin flip paying a spread.")

    if cfg.in_sample:
        warnings.append("Scored with in-sample weights; verify out-of-sample.")

    reasons = [c.detail for c in components] + list(edge.notes[:2])

    return ScoredSetup(
        asset=asset,
        direction=direction,
        score=score,
        components=components,
        edge=edge,
        setups_found=setups_found,
        reasons=reasons,
        vetoes=vetoes,
        warnings=warnings,
        timeframe=timeframe,
        expiry_bars=expiry_bars,
        price=price,
        is_visual=is_visual,
    )


def rank_setups(setups: List[ScoredSetup], top_n: int = 5) -> List[ScoredSetup]:
    """Blueprint: top 5, or fewer if weak. 'No trade' is a valid output."""
    tradable = [s for s in setups if not s.vetoes]
    tradable.sort(key=lambda s: (s.score, s.edge.p_technical if s.edge else 0.0), reverse=True)
    return tradable[:top_n]
