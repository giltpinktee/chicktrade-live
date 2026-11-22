"""CHICKTRADE Signal Engine: 3-EMA Wave Detection, HTF Trend, ADX, 3-Layer Storm Guard,
Bayesian-Kelly Adaptive Volume, and Auto-Turbo with Hard 1.10x Worst-Case SL Floor Guard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import StrategyConfig
from core.adaptive_buffer import AdaptiveBufferController
from core.adaptive_sizing import AdaptiveDecision, BayesianKellySizer, SetupPrior
from strategy.indicators import add_indicators

from bot.bot_config import BotConfig
from bot.execution.base import OrderPlan
from bot.state_store import StateStore

HTF_MAP: Dict[str, str] = {
    "5m": "1h",
    "15m": "1h",
    "30m": "4h",
    "1h": "4h",
}

VALID_EXIT_ARCHS = ("ONLY_TP1", "ONLY_1R", "PARTIAL_NO_MOVE", "PARTIAL_MOVE_SL")


@dataclass
class StrategyRule:
    coin: str
    tf: str
    ema: str
    setup_code: str
    sl_u: float
    tp1_u: float
    tp2_u: float
    leg2_sl_u: float
    exit_arch: str = "PARTIAL_MOVE_SL"
    tp1_frac: float = 0.5
    prior_trades: int = 20
    prior_wins: int = 14
    avg_win_u: float = 3.5
    avg_loss_u: float = 6.25
    sl_full_pct: float = 25.0
    is_scout: bool = False

    def to_setup_prior(self) -> SetupPrior:
        return SetupPrior(
            coin=self.coin,
            tf=self.tf,
            setup_code=self.setup_code,
            prior_trades=self.prior_trades,
            prior_wins=self.prior_wins,
            avg_win_u=self.avg_win_u,
            avg_loss_u=self.avg_loss_u,
            sl_full_pct=self.sl_full_pct,
        )


@dataclass
class StormStatus:
    in_storm: bool
    elevated_vol: bool
    atr_ratio: float
    kaufman_er: float
    sweep_candle: bool
    reason: str = "OK"


@dataclass
class CoinScanSnapshot:
    symbol: str
    last_price: float
    tf_metrics: Dict[str, Dict[str, object]] = field(default_factory=dict)
    storm_active: bool = False
    storm_reason: str = "OK"


@dataclass
class SizingDecision:
    allowed: bool
    reason: str
    order_plan: Optional[OrderPlan] = None
    adaptive: Optional[AdaptiveDecision] = None
    active_buffer: float = 400.0
    turbo_used: bool = False
    turbo_fallback_reason: str = ""
    buffer_reason: str = ""


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX(14) matching CHICKTRADE research modules."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = high.diff().clip(lower=0.0)
    minus_dm = (-low.diff()).clip(lower=0.0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_storm_metrics(
    df: pd.DataFrame,
    atr_spike_mult: float = 1.8,
    er_min: float = 0.15,
    er_atr_mult: float = 1.20,
    elevated_atr_mult: float = 1.35,
) -> pd.DataFrame:
    """Compute zero-lookahead volatility spike, sweep candle, and Kaufman ER(10) flags."""
    out = pd.DataFrame(index=df.index)
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    atr_ma50 = atr.rolling(50, min_periods=10).mean()
    atr_ratio = (atr / atr_ma50.replace(0, np.nan)).fillna(1.0)

    crange = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    wick_frac = (1.0 - body / crange).fillna(0.0)
    sweep_candle = (crange >= 2.8 * atr) | ((crange >= 2.0 * atr) & (wick_frac >= 0.70))

    net_move = (close - close.shift(10)).abs()
    path_len = close.diff().abs().rolling(10, min_periods=10).sum().replace(0, np.nan)
    er = (net_move / path_len).fillna(0.5)

    er_whipsaw = (er < er_min) & (atr_ratio >= er_atr_mult)
    raw_storm = (atr_ratio >= atr_spike_mult) | sweep_candle | er_whipsaw

    out["atr"] = atr
    out["atr_ratio"] = atr_ratio
    out["er"] = er
    out["sweep_candle"] = sweep_candle
    out["er_whipsaw"] = er_whipsaw
    out["elevated_vol"] = (atr_ratio >= elevated_atr_mult) & (~raw_storm)
    out["raw_storm"] = raw_storm
    return out


def compute_htf_trend_series(df_htf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (timestamps, trend_values) on closed HTF candles (+1 bull, -1 bear, 0 neutral)."""
    htf = df_htf.copy()
    htf["htf_ema34"] = htf["close"].ewm(span=34, adjust=False).mean()
    htf["htf_ema200"] = htf["close"].ewm(span=200, adjust=False).mean()
    htf["htf_trend"] = np.where(
        (htf["close"] > htf["htf_ema200"]) & (htf["htf_ema34"] > htf["htf_ema200"]),
        1,
        np.where((htf["close"] < htf["htf_ema200"]) & (htf["htf_ema34"] < htf["htf_ema200"]), -1, 0),
    )
    # Note: df_htf already contains strictly CLOSED candles from MarketFeed.
    return htf["timestamp"].values, htf["htf_trend"].astype(int).values


class SignalEngine:
    """Loads CHICKTRADE strategy rules, scans closed klines, applies Storm Guard & Adaptive Turbo Sizing."""

    def __init__(self, cfg: BotConfig, store: StateStore, rules_dict: Optional[dict] = None):
        self.cfg = cfg
        self.store = store
        self.strategy_cfg = StrategyConfig()
        self.kelly_sizer = BayesianKellySizer()
        self.buffer_controller = AdaptiveBufferController(
            base_buffer=cfg.buffer_units,
            min_buffer=min(cfg.auto_turbo_buffer_units, 230.0),
            max_buffer=500.0,
            target_dd_pct=12.0,
            hard_dd_ceiling_pct=15.0,
            turbo_buffer=cfg.auto_turbo_buffer_units,
            turbo_trigger_ratio=cfg.auto_turbo_trigger_ratio,
            turbo_disengage_ratio=cfg.auto_turbo_disengage_ratio,
            floor_ratio=cfg.auto_turbo_floor_ratio,
        )

        if rules_dict is None:
            with open(cfg.rules_path, encoding="utf-8") as f:
                rules_dict = json.load(f)
        self.raw_rules = rules_dict
        self.version = str(rules_dict.get("version", "1.5"))

        self.ema_risk_multipliers: Dict[str, float] = {
            k: float(v)
            for k, v in rules_dict.get(
                "ema_risk_multipliers", {"EMA34": 1.0, "EMA89": 0.95, "EMA200": 0.90}
            ).items()
        }
        self.max_open_mult_cap = float(rules_dict.get("max_open_mult_cap", cfg.max_open_mult_cap))

        sg = rules_dict.get("storm_guard", {})
        self.cooldown_hours: float = float(sg.get("cooldown_hours", 4))
        self.atr_spike_mult: float = float(sg.get("atr_spike_mult", 1.8))
        self.btc_er_min: float = float(sg.get("btc_er_min", 0.15))
        self.btc_er_atr_mult: float = float(sg.get("btc_er_atr_mult", 1.2))
        self.elevated_atr_mult: float = float(sg.get("elevated_atr_mult", 1.35))
        self.dd_brake_pct: float = float(sg.get("dd_brake_pct", -6.0))
        self.consec_sl_brake: int = int(sg.get("consec_sl_brake", 2))
        self.daily_sl_stop: int = int(sg.get("daily_sl_stop", 3))
        self.daily_loss_stop_pct: float = float(sg.get("daily_loss_stop_pct", -5.0))
        self.min_adx: float = 20.0

        self.rules: Dict[Tuple[str, str, str], StrategyRule] = {}
        self.coins: List[str] = []
        self.tfs: List[str] = []
        self._load_rules(rules_dict.get("rules", []))

    def _load_rules(self, rules_list: List[dict]) -> None:
        coins_set = set()
        tfs_set = set()
        for r in rules_list:
            exit_arch = str(r.get("exit_arch", "PARTIAL_MOVE_SL")).upper().strip()
            if exit_arch not in VALID_EXIT_ARCHS:
                exit_arch = "PARTIAL_MOVE_SL"
            default_frac = 1.0 if exit_arch in ("ONLY_TP1", "ONLY_1R") else 0.5
            tp1_frac = float(r.get("tp1_frac", default_frac))

            prior = r.get("prior_1y", {})
            rule = StrategyRule(
                coin=str(r["coin"]).upper(),
                tf=str(r["tf"]),
                ema=str(r["ema"]).upper(),
                setup_code=str(r["setup_code"]).upper(),
                sl_u=float(r["sl_u"]),
                tp1_u=float(r["tp1_u"]),
                tp2_u=float(r["tp2_u"]),
                leg2_sl_u=float(r.get("leg2_sl_u", 0.0)),
                exit_arch=exit_arch,
                tp1_frac=tp1_frac,
                prior_trades=int(prior.get("trades", 20)),
                prior_wins=int(prior.get("wins", 14)),
                avg_win_u=float(prior.get("avg_win_u", 3.5)),
                avg_loss_u=float(prior.get("avg_loss_u", 6.25)),
                sl_full_pct=float(prior.get("sl_full_pct", 25.0)),
                is_scout=bool(r.get("is_scout", False)),
            )
            self.rules[(rule.coin, rule.tf, rule.setup_code)] = rule
            coins_set.add(rule.coin)
            tfs_set.add(rule.tf)

        self.coins = sorted(coins_set)
        tf_order = {"5m": 0, "15m": 1, "30m": 2, "1h": 3, "4h": 4}
        self.tfs = sorted(tfs_set, key=lambda x: tf_order.get(x, 99))

    def required_htfs(self) -> List[str]:
        return sorted({HTF_MAP[tf] for tf in self.tfs if tf in HTF_MAP})

    # ---- Storm Guard (3-Layer: Local + BTC Macro + 4h Cooldown) ----

    def evaluate_storm(
        self,
        df: pd.DataFrame,
        scope: str,
        now: Optional[datetime] = None,
    ) -> StormStatus:
        """Evaluate latest closed candle storm metrics and persist 4h cooldown if triggered."""
        now_dt = now if now is not None else datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        metrics = compute_storm_metrics(
            df,
            atr_spike_mult=self.atr_spike_mult,
            er_min=self.btc_er_min,
            er_atr_mult=self.btc_er_atr_mult,
            elevated_atr_mult=self.elevated_atr_mult,
        )
        last = metrics.iloc[-1]
        atr_ratio = float(last["atr_ratio"])
        er = float(last["er"])
        sweep = bool(last["sweep_candle"])
        raw_storm = bool(last["raw_storm"])
        elevated = bool(last["elevated_vol"])

        reason = "OK"
        if raw_storm:
            parts = []
            if atr_ratio >= self.atr_spike_mult:
                parts.append(f"ATR_SPIKE({atr_ratio:.2f}x)")
            if sweep:
                parts.append("SWEEP_CANDLE")
            if bool(last["er_whipsaw"]):
                parts.append(f"WHIPSAW(ER={er:.2f},ATR={atr_ratio:.2f}x)")
            reason = "+".join(parts) or "STORM"
            bar_ts = pd.Timestamp(df["timestamp"].iloc[-1]).to_pydatetime()
            if bar_ts.tzinfo is None:
                bar_ts = bar_ts.replace(tzinfo=timezone.utc)
            until_dt = max(now_dt, bar_ts) + timedelta(hours=self.cooldown_hours)
            self.store.set_storm_cooldown(scope, until_dt, reason)
            return StormStatus(
                in_storm=True,
                elevated_vol=False,
                atr_ratio=atr_ratio,
                kaufman_er=er,
                sweep_candle=sweep,
                reason=reason,
            )

        in_cd, cd_reason = self.store.is_in_storm_cooldown(scope, now=now_dt)
        if in_cd:
            return StormStatus(
                in_storm=True,
                elevated_vol=elevated,
                atr_ratio=atr_ratio,
                kaufman_er=er,
                sweep_candle=sweep,
                reason=f"COOLDOWN({cd_reason})",
            )

        return StormStatus(
            in_storm=False,
            elevated_vol=elevated,
            atr_ratio=atr_ratio,
            kaufman_er=er,
            sweep_candle=sweep,
            reason="ELEVATED_VOL" if elevated else "OK",
        )

    # ---- 3-EMA Wave Detection on latest closed bar ----

    def detect_latest_signals(
        self,
        symbol: str,
        tf: str,
        df: pd.DataFrame,
        df_htf: pd.DataFrame,
    ) -> Tuple[List[dict], Dict[str, object]]:
        """Run 3-EMA wave state machine across `df` and return any signals firing on the latest closed bar."""
        data = add_indicators(df, self.strategy_cfg)
        adx_series = compute_adx(df, 14)
        adx = adx_series.values
        htf_ts, htf_tr = compute_htf_trend_series(df_htf)

        n = len(data)
        last_adx = float(adx[-1]) if n > 0 and not np.isnan(adx[-1]) else 0.0
        last_htf_trend = int(htf_tr[-1]) if len(htf_tr) > 0 else 0
        metrics_summary: Dict[str, object] = {
            "adx": round(last_adx, 1),
            "htf_trend": last_htf_trend,
            "bars": n,
        }
        if n < 206:
            return [], metrics_summary

        closes = data["close"].values
        opens = data["open"].values
        highs = data["high"].values
        lows = data["low"].values
        ema34 = data["ema34"].values
        ema89 = data["ema89"].values
        ema200 = data["ema200"].values
        atr = data["atr"].values
        ts_arr = data["timestamp"].values

        ema_specs = [
            (34, ema34, 0.8, 3),
            (89, ema89, 1.0, 5),
            (200, ema200, 1.2, 8),
        ]

        latest_idx = n - 1
        found: List[dict] = []

        for ema_p, et_arr, imp_mult, min_gap in ema_specs:
            trend = 0
            wave_count = 0
            armed = False
            last_touch = -999

            for i in range(205, n):
                c, o, h, l, a = closes[i], opens[i], highs[i], lows[i], atr[i]
                e34, e89, e200, et = ema34[i], ema89[i], ema200[i], et_arr[i]
                if np.isnan(a) or a <= 0 or np.isnan(e200):
                    continue

                if ema_p == 34:
                    new_trend = 1 if (c > e200 and e34 > e89 > e200) else (-1 if (c < e200 and e34 < e89 < e200) else 0)
                elif ema_p == 89:
                    new_trend = 1 if (c > e200 and e89 > e200) else (-1 if (c < e200 and e89 < e200) else 0)
                else:
                    new_trend = (
                        1
                        if (e89 > e200 and c >= e200 - 0.8 * a)
                        else (-1 if (e89 < e200 and c <= e200 + 0.8 * a) else 0)
                    )

                if new_trend != trend:
                    trend = new_trend
                    wave_count = 0
                    armed = False
                    last_touch = -999
                    continue

                if trend == 0:
                    continue

                if trend == 1:
                    if (h - et) >= imp_mult * a and l > et:
                        armed = True
                else:
                    if (et - l) >= imp_mult * a and h < et:
                        armed = True

                if not armed or (i - last_touch) < min_gap:
                    continue

                touch_buf = 0.20 * a
                breach_buf = 0.65 * a
                if trend == 1:
                    touched = (l <= et + touch_buf) and (c >= et - breach_buf)
                else:
                    touched = (h >= et - touch_buf) and (c <= et + breach_buf)

                if not touched:
                    continue

                wave_count += 1
                armed = False
                last_touch = i

                if i != latest_idx or wave_count > 4:
                    continue

                if np.isnan(adx[i]) or adx[i] < self.min_adx:
                    continue

                h_idx = int(np.searchsorted(htf_ts, ts_arr[i], side="right")) - 1
                if h_idx < 0:
                    continue
                ht = int(htf_tr[h_idx])

                if wave_count <= 3:
                    if ht != trend:
                        continue
                    direction = trend
                    setup_code = f"EMA{ema_p}_W{wave_count}"
                else:
                    direction = -trend
                    setup_code = f"EMA{ema_p}_W4_REV"

                if (symbol.upper(), tf, setup_code) not in self.rules:
                    continue

                ts = pd.Timestamp(ts_arr[i])
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")

                found.append(
                    {
                        "symbol": symbol.upper(),
                        "tf": tf,
                        "ema": f"EMA{ema_p}",
                        "setup_code": setup_code,
                        "direction": int(direction),
                        "wave": int(wave_count),
                        "entry_price": float(c),
                        "atr": float(a),
                        "adx": float(adx[i]),
                        "bar_ts": ts.isoformat(),
                    }
                )

        return found, metrics_summary

    # ---- Adaptive Volume + Auto-Turbo with Hard 1.10x SL Floor Guard ----

    def evaluate_sizing_and_build_plan(
        self,
        rule: StrategyRule,
        direction: int,
        entry_price: float,
        atr: float,
        signal_bar_ts: str,
        elevated_vol: bool = False,
        now: Optional[datetime] = None,
    ) -> SizingDecision:
        """Compute Bayesian-Kelly adaptive volume, Auto-Turbo buffer, and hard 1.10x worst-case SL floor."""
        # 1. Daily stop & trade count checks
        dc = self.store.get_daily_counter()
        if dc.trades_count >= self.cfg.max_trades_per_day:
            return SizingDecision(False, f"MAX_TRADES_PER_DAY ({dc.trades_count}/{self.cfg.max_trades_per_day})")
        if dc.sl_count >= self.daily_sl_stop:
            return SizingDecision(False, f"DAILY_SL_STOP ({dc.sl_count}/{self.daily_sl_stop})")
        if dc.start_equity > 0:
            daily_loss_pct = (dc.realized_pnl_usd / dc.start_equity) * 100.0
            if daily_loss_pct <= self.daily_loss_stop_pct:
                return SizingDecision(False, f"DAILY_LOSS_STOP ({daily_loss_pct:.2f}%)")

        # 2. Bayesian-Kelly Adaptive Volume
        live_outcomes = self.store.get_setup_outcomes(rule.coin, rule.tf, rule.setup_code)
        adaptive = self.kelly_sizer.evaluate(rule.to_setup_prior(), live_outcomes)
        if adaptive.vol_mult <= 0.0:
            return SizingDecision(
                False,
                f"BAYESIAN_KELLY_PAUSE (post_wr={adaptive.posterior_wr*100:.1f}%, kelly={adaptive.kelly_full_pct:.1f}%)",
                adaptive=adaptive,
            )

        # 3. EMA Risk Multiplier + Account Risk Brakes
        ema_mult = self.ema_risk_multipliers.get(rule.ema, 1.0)
        comp = self.store.get_compounding_info()
        equity = comp["equity"]
        peak_equity = comp["peak_equity"]
        tier_capital = comp["tier_capital"]
        consec_sl = comp["consec_sl"]

        brake_mult = 1.0
        dd_pct = ((equity - peak_equity) / peak_equity * 100.0) if peak_equity > 0 else 0.0
        if dd_pct <= self.dd_brake_pct or consec_sl >= self.consec_sl_brake or elevated_vol:
            brake_mult = 0.5

        eff_mult = round(adaptive.vol_mult * ema_mult * brake_mult, 4)
        open_mult_sum = self.store.get_open_mult_sum()
        remaining_mult_budget = self.max_open_mult_cap - open_mult_sum
        if remaining_mult_budget <= 0.05:
            return SizingDecision(
                False,
                f"MAX_OPEN_MULT_CAP ({open_mult_sum:.2f}/{self.max_open_mult_cap:.2f})",
                adaptive=adaptive,
            )
        eff_mult = min(eff_mult, remaining_mult_budget)

        # 4. Unit size, Auto-Turbo hysteresis & Dynamic Drawdown-Targeted Adaptive Buffer
        unit_size = max(atr * self.cfg.atr_to_unit_mult, entry_price * 1e-5)
        sizing_base = min(tier_capital, equity)
        if sizing_base <= 0.0:
            return SizingDecision(False, "ZERO_EQUITY", adaptive=adaptive)

        turbo_active = bool(comp["turbo_active"])
        if self.cfg.auto_turbo_enabled:
            if equity >= self.cfg.initial_capital * self.cfg.auto_turbo_trigger_ratio:
                turbo_active = True
            elif equity < self.cfg.initial_capital * self.cfg.auto_turbo_disengage_ratio:
                turbo_active = False
        else:
            turbo_active = False
        self.store.set_state("turbo_active", turbo_active)

        base_buffer = self.cfg.buffer_units
        turbo_buffer = self.cfg.auto_turbo_buffer_units
        stored_buffer = float(self.store.get_state("current_buffer", base_buffer))
        existing_sl_risk_usd = self.store.get_open_sl_risk_usd()

        cand_qty_turbo = (sizing_base * eff_mult) / (turbo_buffer * unit_size)
        cand_sl_risk_turbo = cand_qty_turbo * (rule.sl_u * unit_size)

        active_buffer, buffer_reason = self.buffer_controller.compute_effective_buffer(
            current_buffer=stored_buffer,
            current_dd_pct=dd_pct,
            equity=equity,
            peak_equity=peak_equity,
            initial_capital=self.cfg.initial_capital,
            turbo_enabled=turbo_active,
            existing_sl_risk_usd=existing_sl_risk_usd,
            candidate_sl_risk_usd=cand_sl_risk_turbo,
            turbo_target_buffer=turbo_buffer,
        )

        is_floor_fallback = "FLOOR_GUARD" in buffer_reason
        turbo_used = bool(turbo_active and (active_buffer < base_buffer))
        turbo_fallback_reason = buffer_reason if is_floor_fallback else ""
        if not is_floor_fallback:
            self.store.set_state("current_buffer", active_buffer)

        # 5. Concurrent positions check for active buffer & 1-direction-per-coin guard
        open_positions = self.store.get_open_positions()
        if self.cfg.max_concurrent_override is not None and self.cfg.max_concurrent_override > 0:
            max_conc = int(self.cfg.max_concurrent_override)
        else:
            max_conc = self.buffer_controller.max_concurrent_for_buffer(active_buffer)
        if len(open_positions) >= max_conc:
            return SizingDecision(
                False,
                f"MAX_CONCURRENT ({len(open_positions)}/{max_conc} @ {active_buffer:.0f}G)",
                adaptive=adaptive,
                active_buffer=active_buffer,
                turbo_used=turbo_used,
                turbo_fallback_reason=turbo_fallback_reason,
                buffer_reason=buffer_reason,
            )

        # Strict 1-direction-per-coin: forbid any duplicate same-symbol open trade
        for p in open_positions:
            if p.symbol == rule.coin:
                return SizingDecision(
                    False,
                    f"DUPLICATE_SYMBOL_OPEN ({rule.coin} already open: {p.tf} {p.setup_code})",
                    adaptive=adaptive,
                    active_buffer=active_buffer,
                    turbo_used=turbo_used,
                    turbo_fallback_reason=turbo_fallback_reason,
                    buffer_reason=buffer_reason,
                )

        # 6. Final quantity & prices across all 4 Exit Architectures
        qty = (sizing_base * eff_mult) / (active_buffer * unit_size)
        sl_dist = rule.sl_u * unit_size
        sl_risk_usd = qty * sl_dist

        d = int(direction)
        sl_price = entry_price - d * sl_dist

        if rule.exit_arch == "ONLY_1R":
            tp1_u_eff = rule.sl_u
            tp1_frac = 1.0
        elif rule.exit_arch == "ONLY_TP1":
            tp1_u_eff = rule.tp1_u
            tp1_frac = 1.0
        else:
            tp1_u_eff = rule.tp1_u
            tp1_frac = rule.tp1_frac

        tp1_price = entry_price + d * tp1_u_eff * unit_size
        tp2_price = entry_price + d * rule.tp2_u * unit_size

        if rule.exit_arch == "PARTIAL_NO_MOVE":
            leg2_sl_price = sl_price
        elif rule.leg2_sl_u > 0:
            slip = self.cfg.slippage_bps / 10_000.0
            fee_pad = self.cfg.maker_fee_rate + self.cfg.taker_fee_rate + slip
            leg2_sl_price = entry_price + d * entry_price * fee_pad
        else:
            leg2_sl_price = entry_price + d * rule.leg2_sl_u * unit_size

        plan = OrderPlan(
            symbol=rule.coin,
            tf=rule.tf,
            setup_code=rule.setup_code,
            ema=rule.ema,
            direction=d,
            entry_price=entry_price,
            qty=qty,
            unit_size=unit_size,
            sl_u=rule.sl_u,
            tp1_u=tp1_u_eff,
            tp2_u=rule.tp2_u,
            leg2_sl_u=rule.leg2_sl_u,
            exit_arch=rule.exit_arch,
            tp1_frac=tp1_frac,
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            leg2_sl_price=leg2_sl_price,
            sl_risk_usd=sl_risk_usd,
            vol_mult=eff_mult,
            buffer_used=active_buffer,
            signal_bar_ts=signal_bar_ts,
            turbo_used=turbo_used,
            turbo_fallback_reason=turbo_fallback_reason,
            tier_label=adaptive.tier_label,
            posterior_wr=adaptive.posterior_wr,
        )
        return SizingDecision(
            allowed=True,
            reason="OK",
            order_plan=plan,
            adaptive=adaptive,
            active_buffer=active_buffer,
            turbo_used=turbo_used,
            turbo_fallback_reason=turbo_fallback_reason,
            buffer_reason=buffer_reason,
        )
