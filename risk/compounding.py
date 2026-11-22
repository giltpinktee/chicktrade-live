"""20-step (+8%/step) compounding ladder & ATR-normalized 200-unit position sizing."""
from dataclasses import dataclass
from typing import List

from config import RiskConfig


@dataclass
class SizePlan:
    tier_label: str         # "FULL", "HALF", or "SKIP"
    size_mult: float        # 1.0, 0.5, or 0.0
    unit_size: float        # USD per 1 Unit (= atr_to_unit_mult * ATR)
    qty: float              # Coin quantity to open
    notional: float         # qty * entry_price
    effective_leverage: float
    sl_distance: float      # sl_units * unit_size
    tp_distance: float      # tp_units * unit_size
    be_distance: float      # be_trigger_units * unit_size
    sl_risk_usd: float      # Expected loss if SL hit (before fees)
    drop_to_wipe_pct: float # % adverse move for 1 position to wipe current tier capital


class CompoundingManager:
    """Tracks 20 compounding steps (+8%/step) and computes 200-unit buffer sizing."""

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg
        self.step: int = 1
        self.tier_capital: float = cfg.initial_capital
        self.target_capital: float = self._step_target(1)
        self.max_step_reached: int = 1
        self.step_history: List[dict] = [
            {"step": 1, "tier_capital": self.tier_capital, "target_capital": self.target_capital}
        ]

    def _step_target(self, step: int) -> float:
        return self.cfg.initial_capital * ((1.0 + self.cfg.step_gain_pct) ** step)

    def _step_base(self, step: int) -> float:
        return self.cfg.initial_capital * ((1.0 + self.cfg.step_gain_pct) ** (step - 1))

    def update_equity(self, equity: float, timestamp=None) -> bool:
        """Advance step(s) whenever equity reaches current step's +8% target."""
        stepped_up = False
        while self.step < self.cfg.compounding_steps and equity >= self.target_capital:
            self.step += 1
            self.tier_capital = self._step_base(self.step)
            self.target_capital = self._step_target(self.step)
            self.max_step_reached = max(self.max_step_reached, self.step)
            self.step_history.append(
                {
                    "step": self.step,
                    "tier_capital": self.tier_capital,
                    "target_capital": self.target_capital,
                    "equity": equity,
                    "timestamp": str(timestamp) if timestamp is not None else "",
                }
            )
            stepped_up = True
        return stepped_up

    def tier_for_winrate(self, winrate: float) -> tuple[str, float]:
        if winrate < self.cfg.min_winrate_to_trade:
            return "SKIP", 0.0
        if winrate > self.cfg.full_size_winrate:
            return "FULL", 1.0
        return "HALF", self.cfg.half_size_mult

    def compute_size(
        self,
        entry_price: float,
        atr: float,
        setup_winrate: float,
        current_equity: float,
    ) -> SizePlan:
        """Compute coin quantity using 200-unit safety buffer scaled by confidence tier."""
        tier_label, size_mult = self.tier_for_winrate(setup_winrate)
        unit_size = max(atr * self.cfg.atr_to_unit_mult, entry_price * 1e-5)

        if size_mult <= 0.0 or current_equity <= 0.0:
            return SizePlan(
                tier_label="SKIP",
                size_mult=0.0,
                unit_size=unit_size,
                qty=0.0,
                notional=0.0,
                effective_leverage=0.0,
                sl_distance=self.cfg.sl_units * unit_size,
                tp_distance=self.cfg.tp_units * unit_size,
                be_distance=self.cfg.be_trigger_units * unit_size,
                sl_risk_usd=0.0,
                drop_to_wipe_pct=0.0,
            )

        # Use min(tier_capital, current_equity) so drawdown never over-leverages
        sizing_base = min(self.tier_capital, current_equity)
        # Full size: Qty = sizing_base / (200 * unit_size)
        # Half size (51-70%): Qty = 0.5 * Full size => 2 concurrent trades share 200-unit buffer
        qty = (sizing_base * size_mult) / (self.cfg.buffer_units * unit_size)
        notional = qty * entry_price
        eff_lev = notional / current_equity if current_equity > 0 else 0.0
        sl_dist = self.cfg.sl_units * unit_size
        tp_dist = self.cfg.tp_units * unit_size
        be_dist = self.cfg.be_trigger_units * unit_size
        sl_risk_usd = qty * sl_dist
        drop_to_wipe_pct = (current_equity / notional * 100.0) if notional > 0 else 0.0

        return SizePlan(
            tier_label=tier_label,
            size_mult=size_mult,
            unit_size=unit_size,
            qty=qty,
            notional=notional,
            effective_leverage=eff_lev,
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            be_distance=be_dist,
            sl_risk_usd=sl_risk_usd,
            drop_to_wipe_pct=drop_to_wipe_pct,
        )
