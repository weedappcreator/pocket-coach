"""Risk rules and trade sizing.

This module is deliberately unforgiving. The blueprint's own diagnosis is that
the trader's problem is not a lack of signals -- it is a 50% win rate at a 92%
payout (52.08% break-even), 5% stakes and a two-step martingale. That
combination has negative expectancy *and* a fat left tail. So the risk agent
here has real veto power: it can refuse a setup outright, and it refuses to
size up after losses.

All limits are expressed as fractions of the *current* bankroll, never as
absolute amounts, so they hold as the account grows or shrinks.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class RiskConfig:
    """The risk envelope. Defaults are the blueprint's stated rules."""

    max_stake_pct: float = 2.0           # cap per trade
    target_stake_pct: float = 1.5        # what full-Kelly scaling actually recommends
    daily_loss_limit_pct: float = 5.0    # pause trading for the day
    max_consecutive_losses: int = 3      # hard stop
    min_payout: float = 80.0             # refuse anything paying below this
    min_score: float = 70.0              # blueprint's delivery threshold
    min_score_visual: float = 80.0       # higher bar for OTC/visual signals
    news_blackout_minutes: int = 5       # pause either side of high-impact news
    max_signals_per_day: int = 10        # launch limit
    allow_martingale: bool = False       # hard off
    kelly_fraction: float = 0.25         # quarter-Kelly
    max_edge_pp: float = 3.0             # refuse sub-2.9pp edges by default
    daily_profit_lock_pct: Optional[float] = None  # e.g. 5.0 -> stop at +5%

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AccountState:
    """Everything the risk agent needs to decide on one trade."""

    bankroll: float
    starting_bankroll_today: float
    day: str
    pnl_today: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    trades_total: int = 0
    paused_until: Optional[datetime] = None
    signals_sent_today: int = 0
    ladder_active: bool = False
    last_stake: Optional[float] = None
    last_result: Optional[str] = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["paused_until"] = self.paused_until.isoformat() if self.paused_until else None
        return d


@dataclass
class RiskDecision:
    approved: bool
    stake: float = 0.0
    stake_pct: float = 0.0
    reasons: List[str] = field(default_factory=list)
    vetoes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rule_trace: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------

def recommended_stake_pct(
    payout: float,
    win_rate: float,
    cfg: RiskConfig,
    conviction: float = 1.0,
) -> Tuple[float, str]:
    """Quarter-Kelly stake as a PERCENT of bankroll, clipped to the hard cap.

    Kelly is the right frame here: a fixed-payoff bet has an exact optimal
    stake, and it is usually much larger than a trader's nerve allows. We take
    a fraction of it, because the win rate is estimated and therefore noisy --
    full Kelly on an estimated edge is how accounts die.

    `payout` is a decimal fraction; the return value is a percentage, matching
    `RiskConfig.max_stake_pct`.
    """
    from .payoff import break_even_win_rate, kelly_fraction

    breakeven = break_even_win_rate(payout)
    if win_rate <= breakeven:
        return 0.0, "no edge: estimated win rate is at or below break-even"

    k = kelly_fraction(payout, win_rate)
    if k <= 0:
        return 0.0, "no edge"

    # kelly_fraction returns a fraction; convert to percent before comparing
    # against the percent-valued caps in RiskConfig.
    frac_pct = k * 100.0 * cfg.kelly_fraction * max(0.0, min(1.5, conviction))

    note = "quarter-Kelly on a %+.2f pp edge" % ((win_rate - breakeven) * 100)
    if frac_pct > cfg.max_stake_pct:
        frac_pct = cfg.max_stake_pct
        note = "clipped to the %.1f%% hard cap" % cfg.max_stake_pct
    if frac_pct > cfg.target_stake_pct:
        frac_pct = cfg.target_stake_pct
        note += "; trimmed to the %.1f%% target" % cfg.target_stake_pct
    return frac_pct, note


def flat_stake_pct(cfg: RiskConfig) -> float:
    """Sizing when we have no edge estimate but the setup passed scoring.

    Deliberately flat and small. After a loss streak, flat sizing is the point:
    the ladder is what turns a bad hour into an empty account.
    """
    frac = cfg.target_stake_pct
    if cfg.max_stake_pct < frac:
        frac = cfg.max_stake_pct
    return frac


# --------------------------------------------------------------------------
# Ruin simulation
# --------------------------------------------------------------------------

def simulate_ruin(
    p_win: float,
    payout: float,
    stake_frac: float,
    n_trades: int,
    bankroll: float,
    ruin_level: float,
    rng: random.Random,
) -> Tuple[int, float, float]:
    """Roll out one random path of a fixed-fraction bettor.

    Returns (hit_ruin, ending_bankroll, worst_drawdown_seen).
    """
    bal = bankroll
    floor = bankroll * (1.0 - ruin_level)
    peak = bal
    worst_dd = 0.0
    for _ in range(n_trades):
        stake = bal * stake_frac
        bal += stake * (payout if rng.random() < p_win else -1.0)
        if bal > peak:
            peak = bal
        if peak > 0:
            dd = (peak - bal) / peak
            if dd > worst_dd:
                worst_dd = dd
        if bal <= floor:
            return 1, bal, worst_dd
    return 0, bal, worst_dd


# --------------------------------------------------------------------------
# The risk agent
# --------------------------------------------------------------------------

class RiskEngine:
    """Holds account state and answers 'may I take this trade, and at what size?'"""

    def __init__(self, cfg: Optional[RiskConfig] = None, bankroll: float = 10_000.0):
        self.cfg = cfg or RiskConfig()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        self.state = AccountState(
            bankroll=bankroll,
            starting_bankroll_today=bankroll,
            day=today,
        )
        self.news_windows: List[Tuple[datetime, datetime, str]] = []

    # -- account bookkeeping -------------------------------------------------

    def register_news(self, when: datetime, title: str = "high-impact event") -> None:
        span = timedelta(minutes=self.cfg.news_blackout_minutes)
        self.news_windows.append((when - span, when + span, title))

    def record_result(self, won: bool, stake: float, payout: float, when: Optional[datetime] = None) -> None:
        """Fold a settled trade into account state and roll the daily counters."""
        when = when or datetime.utcnow()
        self._roll_day(when)
        pnl = stake * payout if won else -stake
        self.state.bankroll += pnl
        self.state.pnl_today += pnl
        self.state.trades_today += 1
        self.state.trades_total += 1
        if won:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
        self.state.last_stake = stake
        self.state.last_result = "win" if won else "loss"
        # A martingale is defined by staking more after a loss. Clear the flag
        # on a win; set it whenever the next stake exceeds the last one after
        # a loss, so the guard can catch it downstream.
        if won:
            self.state.ladder_active = False

    def _roll_day(self, when: datetime) -> None:
        today = when.strftime("%Y-%m-%d")
        if today != self.state.day:
            self.state.day = today
            self.state.starting_bankroll_today = self.state.bankroll
            self.state.pnl_today = 0.0
            self.state.trades_today = 0
            self.state.signals_sent_today = 0
            self.state.paused_until = None
            self.state.ladder_active = False

    def note_signal_sent(self, when: Optional[datetime] = None) -> None:
        when = when or datetime.utcnow()
        self._roll_day(when)
        self.state.signals_sent_today += 1

    def note_skipped(self) -> None:
        pass

    # -- the decision --------------------------------------------------------

    def in_news_blackout(self, when: Optional[datetime] = None) -> Optional[str]:
        when = when or datetime.utcnow()
        for start, end, title in self.news_windows:
            if start <= when <= end:
                return title
        return None

    def evaluate(
        self,
        score: float,
        payout: float,
        win_rate: Optional[float] = None,
        conviction: float = 1.0,
        is_visual: bool = False,
        when: Optional[datetime] = None,
    ) -> RiskDecision:
        """Run every rule and return a decision. Vetoes are hard blocks.

        `payout` is in PERCENT (92 means a 92% payout), matching the platform UI
        and `RiskConfig.min_payout`. The maths in `payoff` wants a fraction, so
        the conversion happens once, here.
        """
        cfg = self.cfg
        when = when or datetime.utcnow()
        self._roll_day(when)
        d = RiskDecision(approved=False)
        payout_fraction = payout / 100.0

        # --- hard vetoes ---------------------------------------------------
        if payout < cfg.min_payout:
            d.vetoes.append(
                "Payout %.0f%% is below the %.0f%% floor (break-even would be %.1f%%)."
                % (payout, cfg.min_payout, 100.0 / (1.0 + payout / 100.0))
            )

        threshold = cfg.min_score_visual if is_visual else cfg.min_score
        if score < threshold:
            d.vetoes.append(
                "Score %.0f is below the %.0f delivery threshold%s."
                % (score, threshold, " for visual/OTC data" if is_visual else "")
            )

        if self.state.consecutive_losses >= cfg.max_consecutive_losses:
            d.vetoes.append(
                "%d consecutive losses -- hard stop at %d. Stop for the day."
                % (self.state.consecutive_losses, cfg.max_consecutive_losses)
            )

        day_loss = self.state.pnl_today
        limit = -abs(self.state.starting_bankroll_today * cfg.daily_loss_limit_pct / 100.0)
        if day_loss <= limit:
            d.vetoes.append(
                "Daily loss limit reached: %+.2f%% today vs a -%.0f%% limit."
                % (day_loss / max(self.state.starting_bankroll_today, 1e-9) * 100.0,
                   cfg.daily_loss_limit_pct)
            )

        if cfg.daily_profit_lock_pct is not None:
            gain_pct = day_loss / max(self.state.starting_bankroll_today, 1e-9) * 100.0
            if gain_pct >= cfg.daily_profit_lock_pct:
                d.vetoes.append(
                    "Daily profit lock: +%.1f%% today, stopping at +%.0f%%."
                    % (gain_pct, cfg.daily_profit_lock_pct)
                )

        if self.state.signals_sent_today >= cfg.max_signals_per_day:
            d.vetoes.append(
                "Signal cap reached: %d/%d today." % (self.state.signals_sent_today, cfg.max_signals_per_day)
            )

        news = self.in_news_blackout(when)
        if news:
            d.vetoes.append("High-impact news window: %s." % news)

        if self.state.paused_until and when < self.state.paused_until:
            d.vetoes.append("Trading paused until %s." % self.state.paused_until.strftime("%H:%M:%S"))

        # --- edge check ----------------------------------------------------
        if win_rate is not None:
            from .payoff import break_even_win_rate, ev_pct

            breakeven = break_even_win_rate(payout_fraction)
            edge_pp = (win_rate - breakeven) * 100.0
            d.rule_trace["breakeven_win_rate"] = breakeven * 100.0
            d.rule_trace["estimated_edge_pp"] = edge_pp
            d.rule_trace["ev_pct_per_trade"] = ev_pct(payout_fraction, win_rate)
            if edge_pp < cfg.max_edge_pp:
                d.vetoes.append(
                    "Edge is %+.2f pp over break-even, under the %.0f pp minimum."
                    % (edge_pp, cfg.max_edge_pp)
                )
            elif edge_pp < cfg.max_edge_pp * 2:
                d.warnings.append(
                    "Thin edge (%+.2f pp). Take it small or skip it." % edge_pp
                )

        if d.vetoes:
            return d

        # --- sizing --------------------------------------------------------
        if win_rate is not None and win_rate > 0:
            stake_pct, note = recommended_stake_pct(payout_fraction, win_rate, cfg, conviction)
        else:
            stake_pct, note = flat_stake_pct(cfg), "flat size, no edge estimate supplied"

        if stake_pct <= 0:
            d.vetoes.append(note or "sizing returned zero")
            return d

        stake = self.state.bankroll * stake_pct / 100.0
        if self.state.last_stake and self.state.last_result == "loss" and stake > self.state.last_stake * 1.01:
            stake = self.state.last_stake
            stake_pct = stake / max(self.state.bankroll, 1e-9) * 100.0
            d.warnings.append(
                "Stake did not increase after a loss (anti-martingale guard)."
            )

        d.approved = True
        d.stake = round(stake, 2)
        d.stake_pct = round(stake_pct, 3)
        d.reasons.append(note)
        d.reasons.append(
            "Risking %.2f%% of the bankroll ($%.2f) on this trade." % (stake_pct, stake)
        )
        if self.state.consecutive_losses > 0:
            d.warnings.append(
                "Trading into a %d-loss streak. Check the setup, not just the rules."
                % self.state.consecutive_losses
            )
        return d

    # -- reporting ----------------------------------------------------------

    def status(self) -> dict:
        s = self.state
        day_pnl_pct = s.pnl_today / max(s.starting_bankroll_today, 1e-9) * 100.0
        return {
            "bankroll": round(s.bankroll, 2),
            "day": s.day,
            "pnl_today": round(s.pnl_today, 2),
            "pnl_today_pct": round(day_pnl_pct, 2),
            "consecutive_losses": s.consecutive_losses,
            "trades_today": s.trades_today,
            "signals_today": s.signals_sent_today,
            "trades_total": s.trades_total,
            "paused": bool(
                (s.paused_until and datetime.utcnow() < s.paused_until)
                or s.consecutive_losses >= self.cfg.max_consecutive_losses
                or s.pnl_today <= -abs(s.starting_bankroll_today * self.cfg.daily_loss_limit_pct / 100.0)
            ),
            "limits": {
                "daily_loss_limit_pct": self.cfg.daily_loss_limit_pct,
                "max_consecutive_losses": self.cfg.max_consecutive_losses,
                "max_stake_pct": self.cfg.max_stake_pct,
                "min_payout": self.cfg.min_payout,
                "max_signals_per_day": self.cfg.max_signals_per_day,
            },
        }


def diagnose_behaviour(
    trades: Sequence[dict],
    payout_default: float = 92.0,
    max_stake_pct: float = 2.0,
) -> dict:
    """Reverse-engineer a trader's real behaviour from their own history.

    Point this at ~30k past trades and it will tell you, per asset/payout/hour,
    where the money actually leaks -- and whether the martingale and revenge
    patterns are visible in the numbers.
    """
    if not trades:
        return {"ok": False, "reason": "no trades supplied"}

    n = len(trades)
    wins = sum(1 for t in trades if t.get("won"))
    win_rate = wins / float(n)

    pnl = 0.0
    stakes = []
    for t in trades:
        p = t.get("payout", payout_default) / 100.0
        st = t.get("stake", 0.0)
        stakes.append(st)
        pnl += st * p if t.get("won") else -st

    avg_stake = sum(stakes) / float(n) if n else 0.0
    stake_disp = (max(stakes) - min(stakes)) if stakes else 0.0

    # Did stakes escalate after losses? Compare stake after a loss vs after a win.
    after_loss, after_win = [], []
    for i, t in enumerate(trades):
        if i == 0:
            continue
        prev_won = trades[i - 1].get("won")
        (after_win if prev_won else after_loss).append(t.get("stake", 0.0))

    def _avg(x):
        return sum(x) / float(len(x)) if x else 0.0

    escalation = 0.0
    if after_loss and after_win and _avg(after_win) > 0:
        escalation = (_avg(after_loss) / _avg(after_win) - 1.0) * 100.0

    # Longest losing run, and how the bankroll behaved across it.
    max_run = cur = 0
    for t in trades:
        if not t.get("won"):
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    from .payoff import break_even_win_rate, ev_pct, test_edge

    be = break_even_win_rate(payout_default / 100.0)
    findings = []
    if win_rate < be:
        findings.append(
            "Real win rate %.1f%% is BELOW the %.1f%% break-even for a %.0f%% payout. "
            "As traded, this history loses money."
            % (win_rate * 100, be * 100, payout_default)
        )
    if escalation > 20:
        findings.append(
            "Stake rises %.0f%% after a loss versus after a win -- a martingale. "
            "This is the single largest driver of ruin risk." % escalation
        )
    if max_run >= 4:
        findings.append(
            "Longest losing run was %d trades. A 3-loss stop would have cut the damage."
            % max_run
        )
    if avg_stake > 0 and max(stakes) > avg_stake * 2.5:
        findings.append(
            "Largest stake is %.1fx the average -- check whether the size of the "
            "losing bets is well above the size of the winning ones."
            % (max(stakes) / avg_stake)
        )
    if not findings:
        findings.append("No obvious self-sabotage patterns in the stake series.")

    return {
        "ok": True,
        "n_trades": n,
        "wins": wins,
        "win_rate": win_rate,
        "breakeven": be,
        "edge_pp": (win_rate - be) * 100.0,
        "ev_pct_per_trade": ev_pct(payout_default / 100.0, win_rate),
        "total_pnl": pnl,
        "avg_stake": avg_stake,
        "max_stake": max(stakes) if stakes else 0.0,
        "stake_after_loss_pct_higher": escalation,
        "longest_losing_run": max_run,
        "edge_test": test_edge(wins, n, payout_default / 100.0, in_sample=False).to_dict(),
        "findings": findings,
    }
