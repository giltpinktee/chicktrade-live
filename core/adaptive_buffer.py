"""Dynamic Drawdown-Targeted Adaptive Buffer Controller for CHICKTRADE.

Automatically adjusts the sizing buffer `G` (between `min_buffer=230G` and `max_buffer=500G`)
to target Max Drawdown <= 12% (hard ceiling 15%, severe halt 18%), while allowing Auto-Turbo
step-down on new equity highs subject to the hard 1.10x ($11,000 on $10,000 capital) worst-case
simultaneous SL sweep floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


def max_concurrent_for_buffer(effective_buffer: float) -> int:
    """Map effective buffer units `G` cleanly to maximum concurrent open positions."""
    b = float(effective_buffer)
    if b <= 250.0:
        return 2
    if b <= 300.0:
        return 3
    if b <= 400.0:
        return 5
    return 7


@dataclass
class AdaptiveBufferController:
    """Drawdown-targeted dynamic buffer controller with hard worst-case SL floor guard."""

    base_buffer: float = 400.0
    min_buffer: float = 230.0
    max_buffer: float = 500.0
    target_dd_pct: float = 12.0
    hard_dd_ceiling_pct: float = 15.0
    severe_dd_pct: float = 18.0
    turbo_buffer: float = 230.0
    turbo_trigger_ratio: float = 1.25
    turbo_disengage_ratio: float = 1.15
    floor_ratio: float = 1.10
    step_units: float = 10.0

    def snap_buffer(self, value: float) -> float:
        """Snap buffer to nearest `step_units` (10G) and clamp to `[min_buffer, max_buffer]`."""
        step = max(1.0, float(self.step_units))
        snapped = round(float(value) / step) * step
        return float(max(self.min_buffer, min(self.max_buffer, snapped)))

    def scale_buffer_for_dd(self, current_buffer: float, current_dd_pct: float) -> float:
        """Proportionally scale buffer up when |DD| > target_dd_pct, snapped to 10G and capped at max_buffer."""
        dd_mag = abs(float(current_dd_pct))
        if dd_mag <= self.target_dd_pct or self.target_dd_pct <= 0.0:
            return self.snap_buffer(current_buffer)
        scaled = float(current_buffer) * (dd_mag / self.target_dd_pct)
        return self.snap_buffer(scaled)

    def max_concurrent_for_buffer(self, effective_buffer: float) -> int:
        """Map effective buffer units to maximum concurrent open positions."""
        return max_concurrent_for_buffer(effective_buffer)

    def check_worst_case_floor(
        self,
        equity: float,
        initial_capital: float,
        existing_sl_risk_usd: float = 0.0,
        candidate_sl_risk_usd: float = 0.0,
    ) -> Tuple[bool, float, float]:
        """Verify that simultaneous SL sweep of all open + candidate trades stays >= 1.10x initial capital."""
        protected_floor = float(initial_capital) * self.floor_ratio
        worst_case_equity = float(equity) - (float(existing_sl_risk_usd) + float(candidate_sl_risk_usd))
        return (worst_case_equity >= protected_floor), worst_case_equity, protected_floor

    def compute_effective_buffer(
        self,
        current_buffer: float,
        current_dd_pct: float,
        equity: float,
        peak_equity: float,
        initial_capital: float,
        turbo_enabled: bool,
        *,
        existing_sl_risk_usd: float = 0.0,
        candidate_sl_risk_usd: float = 0.0,
        turbo_target_buffer: Optional[float] = None,
        step_down_units: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Compute `(effective_buffer, reason)` based on live drawdown, equity high, and hard SL floor.

        Rules:
        1. `|DD| >= severe_dd_pct` (18%): jump immediately to `max_buffer` (500G).
        2. `|DD| >= hard_dd_ceiling_pct` (15%): jump immediately to at least `max(base_buffer, 400G)`
           (or proportionally scaled higher, capped at 500G).
        3. `|DD| > target_dd_pct` (12%): scale buffer up via `current_buffer * (|DD| / target_dd_pct)`,
           snapped to nearest 10G step and capped at `max_buffer`.
        4. `equity >= peak_equity` and `turbo_enabled` and `equity >= initial_capital * 1.25`:
           step down smoothly toward `turbo_buffer` (250G or 230G), ALWAYS subject to the
           `1.10x` ($11,000 on $10,000) worst-case SL sweep floor check.
        """
        dd_mag = abs(float(current_dd_pct))
        cur_buf = float(max(self.min_buffer, min(self.max_buffer, float(current_buffer))))
        eq = float(equity)
        peak = float(peak_equity)
        init_cap = float(initial_capital)
        target_turbo = float(
            max(
                self.min_buffer,
                min(self.base_buffer, turbo_target_buffer if turbo_target_buffer is not None else self.turbo_buffer),
            )
        )

        # 1. Severe drawdown ceiling (>= 18%) -> max defense 500G
        if dd_mag >= self.severe_dd_pct:
            scaled = self.scale_buffer_for_dd(cur_buf, dd_mag)
            eff = self.snap_buffer(max(scaled, self.max_buffer, 500.0))
            return eff, f"DD_SEVERE_CEILING: DD=-{dd_mag:.2f}% >= {self.severe_dd_pct:.1f}% -> {eff:.0f}G"

        # 2. Hard drawdown ceiling (>= 15%) -> immediately at least max(base_buffer, 400G)
        if dd_mag >= self.hard_dd_ceiling_pct:
            scaled = self.scale_buffer_for_dd(cur_buf, dd_mag)
            ceiling_floor = max(self.base_buffer, 400.0)
            eff = self.snap_buffer(max(scaled, ceiling_floor))
            return eff, f"DD_HARD_CEILING: DD=-{dd_mag:.2f}% >= {self.hard_dd_ceiling_pct:.1f}% -> {eff:.0f}G"

        # 3. Above target drawdown (> 12%) -> proportional adaptive scale-up
        if dd_mag > self.target_dd_pct:
            scaled = self.scale_buffer_for_dd(cur_buf, dd_mag)
            eff = max(cur_buf, scaled)
            if (not turbo_enabled) or (eq < init_cap * self.turbo_disengage_ratio):
                eff = max(eff, self.base_buffer)
            eff = self.snap_buffer(eff)
            return eff, f"DD_ADAPTIVE_SCALE: DD=-{dd_mag:.2f}% > {self.target_dd_pct:.1f}% -> {eff:.0f}G"

        # 4. New equity high + Turbo eligible (>= 1.25x initial capital)
        if eq >= peak and turbo_enabled and eq >= init_cap * self.turbo_trigger_ratio:
            floor_ok, worst_case, protected_floor = self.check_worst_case_floor(
                equity=eq,
                initial_capital=init_cap,
                existing_sl_risk_usd=existing_sl_risk_usd,
                candidate_sl_risk_usd=candidate_sl_risk_usd,
            )
            if not floor_ok:
                fallback_buf = self.snap_buffer(max(self.base_buffer, 400.0))
                return (
                    fallback_buf,
                    f"FLOOR_GUARD: worst_case=${worst_case:.2f} < floor=${protected_floor:.2f} -> fallback {fallback_buf:.0f}G",
                )
            if step_down_units is not None and step_down_units > 0.0 and cur_buf > target_turbo:
                eff = self.snap_buffer(max(target_turbo, cur_buf - float(step_down_units)))
            else:
                eff = self.snap_buffer(target_turbo)
            return eff, f"TURBO_ENGAGED: equity=${eq:.2f} >= {self.turbo_trigger_ratio:.2f}x -> {eff:.0f}G"

        # 5. Turbo hysteresis hold (within [1.15x, peak) while DD <= target_dd_pct)
        if turbo_enabled and eq >= init_cap * self.turbo_disengage_ratio and cur_buf < self.base_buffer:
            floor_ok, worst_case, protected_floor = self.check_worst_case_floor(
                equity=eq,
                initial_capital=init_cap,
                existing_sl_risk_usd=existing_sl_risk_usd,
                candidate_sl_risk_usd=candidate_sl_risk_usd,
            )
            if not floor_ok:
                fallback_buf = self.snap_buffer(max(self.base_buffer, 400.0))
                return (
                    fallback_buf,
                    f"FLOOR_GUARD: worst_case=${worst_case:.2f} < floor=${protected_floor:.2f} -> fallback {fallback_buf:.0f}G",
                )
            return cur_buf, f"TURBO_HOLD: equity=${eq:.2f} >= {self.turbo_disengage_ratio:.2f}x -> {cur_buf:.0f}G"

        # 6. Recovery back to new equity high from elevated defensive buffer (> base_buffer)
        if eq >= peak and cur_buf > self.base_buffer:
            if step_down_units is not None and step_down_units > 0.0:
                eff = self.snap_buffer(max(self.base_buffer, cur_buf - float(step_down_units)))
            else:
                eff = self.snap_buffer(self.base_buffer)
            return eff, f"RECOVERY_STEP_DOWN: new_high=${eq:.2f} -> {eff:.0f}G"

        # 7. Non-turbo or below disengage threshold -> at least base_buffer
        if (not turbo_enabled) or (eq < init_cap * self.turbo_disengage_ratio):
            eff = self.snap_buffer(max(cur_buf, self.base_buffer))
            return eff, f"BASE_BUFFER: DD=-{dd_mag:.2f}% <= {self.target_dd_pct:.1f}% -> {eff:.0f}G"

        return cur_buf, f"HOLD_BUFFER: DD=-{dd_mag:.2f}% -> {cur_buf:.0f}G"
