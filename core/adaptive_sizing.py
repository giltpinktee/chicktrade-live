"""Bayesian + Fractional Kelly Adaptive Volume Engine for CHICKTRADE.

Continuously blends 1-year historical backtest priors with live trade outcomes
to dynamically scale position volume up to mathematical maximums for high-WR / low-SL setups,
and automatically throttle down or pause setups whose live edge decays.
"""
from dataclasses import dataclass
import math


@dataclass
class SetupPrior:
    coin: str
    tf: str
    setup_code: str
    prior_trades: int
    prior_wins: int
    avg_win_u: float
    avg_loss_u: float
    sl_full_pct: float
    prior_weight_cap: int = 35  # Cap prior pseudo-count so live trades meaningfully update estimates


@dataclass
class AdaptiveDecision:
    posterior_wr: float         # Updated Bayesian winrate (0..1)
    payoff_ratio: float         # AvgWin_U / AvgLoss_U
    kelly_full_pct: float       # Full Kelly optimal fraction (%)
    quarter_kelly_pct: float    # Safe Quarter-Kelly fraction (%)
    vol_mult: float             # Recommended Volume Multiplier (0.0 .. max_mult)
    tier_label: str             # S+ / A+ / A / B / C / PAUSE
    live_trades: int


class BayesianKellySizer:
    """Computes live-updated volume multipliers from 1Y prior + rolling live results."""

    def __init__(
        self,
        min_wr_active: float = 0.60,
        ref_kelly_for_1x: float = 0.25,  # Full Kelly >= 25% maps to 1.0x baseline
        max_vol_mult: float = 1.30,      # Hard mathematical ceiling (1.30x for S+ low-SL setups)
        min_vol_mult: float = 0.40,
        live_decay: float = 0.96,        # Slight recency weighting for newer live trades
    ):
        self.min_wr_active = min_wr_active
        self.ref_kelly_for_1x = ref_kelly_for_1x
        self.max_vol_mult = max_vol_mult
        self.min_vol_mult = min_vol_mult
        self.live_decay = live_decay

    def evaluate(
        self,
        prior: SetupPrior,
        live_outcomes_u: list[float] | None = None,
    ) -> AdaptiveDecision:
        """Compute updated posterior winrate, Kelly fraction, and optimal volume multiplier."""
        live_outcomes_u = live_outcomes_u or []

        # 1. Scale 1Y prior to prior_weight_cap so live performance adapts smoothly
        scale = min(1.0, prior.prior_weight_cap / max(prior.prior_trades, 1))
        alpha = prior.prior_wins * scale
        beta = (prior.prior_trades - prior.prior_wins) * scale

        # 2. Blend live trades with exponential recency weighting
        live_wins_u = []
        live_loss_u = []
        n_live = len(live_outcomes_u)
        for idx, net_u in enumerate(live_outcomes_u):
            # More recent trades have weight closer to 1.0
            age = (n_live - 1) - idx
            w = self.live_decay ** age
            if net_u > 0:
                alpha += w
                live_wins_u.append(net_u)
            else:
                beta += w
                live_loss_u.append(-net_u)

        post_wr = alpha / max(alpha + beta, 1e-9)

        # 3. Blend payoff ratio (AvgWin / AvgLoss)
        win_w = len(live_wins_u) / (len(live_wins_u) + 10.0)
        loss_w = len(live_loss_u) / (len(live_loss_u) + 10.0)
        avg_win = (1.0 - win_w) * prior.avg_win_u + win_w * (sum(live_wins_u) / len(live_wins_u) if live_wins_u else prior.avg_win_u)
        avg_loss = (1.0 - loss_w) * prior.avg_loss_u + loss_w * (sum(live_loss_u) / len(live_loss_u) if live_loss_u else prior.avg_loss_u)
        payoff_b = avg_win / max(avg_loss, 1e-6)

        # 4. Kelly Criterion: f* = (p*b - q) / b = p - q/b
        q = 1.0 - post_wr
        kelly_full = max(0.0, post_wr - (q / max(payoff_b, 1e-6)))
        quarter_kelly = kelly_full * 0.25

        # 5. Map to safe bounded Volume Multiplier
        if post_wr < self.min_wr_active or kelly_full <= 0.02:
            return AdaptiveDecision(
                posterior_wr=post_wr,
                payoff_ratio=payoff_b,
                kelly_full_pct=kelly_full * 100.0,
                quarter_kelly_pct=quarter_kelly * 100.0,
                vol_mult=0.0,
                tier_label="PAUSE",
                live_trades=n_live,
            )

        raw_mult = kelly_full / self.ref_kelly_for_1x
        # Bonus cap up to max_vol_mult (1.30x) ONLY if WR >= 76% and low SL_Full (<22%)
        cap = self.max_vol_mult if (post_wr >= 0.76 and prior.sl_full_pct <= 22.0) else (1.15 if post_wr >= 0.73 else 1.00)
        vol_mult = round(min(max(raw_mult, self.min_vol_mult), cap), 2)

        if vol_mult >= 1.15:
            tier = "S+ (Diamond)"
        elif vol_mult >= 0.95:
            tier = "A+ (Core)"
        elif vol_mult >= 0.75:
            tier = "A (Standard)"
        elif vol_mult >= 0.60:
            tier = "B (Support)"
        else:
            tier = "C (Satellite)"

        return AdaptiveDecision(
            posterior_wr=post_wr,
            payoff_ratio=payoff_b,
            kelly_full_pct=kelly_full * 100.0,
            quarter_kelly_pct=quarter_kelly * 100.0,
            vol_mult=vol_mult,
            tier_label=tier,
            live_trades=n_live,
        )
