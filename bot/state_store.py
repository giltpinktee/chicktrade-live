"""SQLite persistence layer for CHICKTRADE Live & Paper Trading Bot.

Persists open positions, closed trade history, per-setup live `net_u` outcomes
(for `BayesianKellySizer`), compounding step/equity, daily risk counters,
Storm Guard cooldown timestamps, and processed signal keys across restarts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_today_str(now: Optional[datetime] = None) -> str:
    dt = now if now is not None else datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class OpenPosition:
    position_id: str
    symbol: str
    tf: str
    setup_code: str
    ema: str
    direction: int                  # +1 = LONG, -1 = SHORT
    entry_price: float
    qty: float
    remaining_qty: float
    unit_size: float
    sl_u: float
    tp1_u: float
    tp2_u: float
    leg2_sl_u: float
    exit_arch: str                  # ONLY_TP1 | ONLY_1R | PARTIAL_NO_MOVE | PARTIAL_MOVE_SL
    tp1_frac: float                 # 0.5 for partial, 1.0 for single-leg
    tp1_filled: bool
    leg2_sl_moved: bool
    sl_price: float
    tp1_price: float
    tp2_price: float
    leg2_sl_price: float
    sl_risk_usd: float
    vol_mult: float
    buffer_used: float
    leg1_realized_pnl_usd: float = 0.0
    leg1_realized_net_u: float = 0.0
    entry_fee_usd: float = 0.0
    opened_at: str = ""
    signal_bar_ts: str = ""
    exchange_order_id: str = ""
    sl_order_id: str = ""
    tp1_order_id: str = ""
    tp2_order_id: str = ""


@dataclass
class ClosedTrade:
    position_id: str
    symbol: str
    tf: str
    setup_code: str
    ema: str
    direction: int
    exit_arch: str
    entry_price: float
    exit_price: float
    qty: float
    unit_size: float
    outcome: str                    # SL | TP1 | TP2 | BE | LEG2_SL | MANUAL
    tp1_hit: bool
    leg2_sl_hit: bool
    tp2_hit: bool
    pnl_usd: float
    fees_usd: float
    net_u: float
    opened_at: str
    closed_at: str


@dataclass
class DailyCounter:
    utc_date: str
    trades_count: int
    sl_count: int
    realized_pnl_usd: float
    start_equity: float


class StateStore:
    """Crash-resilient SQLite state store for CHICKTRADE bot."""

    def __init__(self, db_path: Path | str, initial_capital: float = 10_000.0, step_gain_pct: float = 0.08):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.initial_capital = float(initial_capital)
        self.step_gain_pct = float(step_gain_pct)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()
        self._init_defaults()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS open_positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    tf TEXT NOT NULL,
                    setup_code TEXT NOT NULL,
                    ema TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    qty REAL NOT NULL,
                    remaining_qty REAL NOT NULL,
                    unit_size REAL NOT NULL,
                    sl_u REAL NOT NULL,
                    tp1_u REAL NOT NULL,
                    tp2_u REAL NOT NULL,
                    leg2_sl_u REAL NOT NULL,
                    exit_arch TEXT NOT NULL,
                    tp1_frac REAL NOT NULL,
                    tp1_filled INTEGER NOT NULL DEFAULT 0,
                    leg2_sl_moved INTEGER NOT NULL DEFAULT 0,
                    sl_price REAL NOT NULL,
                    tp1_price REAL NOT NULL,
                    tp2_price REAL NOT NULL,
                    leg2_sl_price REAL NOT NULL,
                    sl_risk_usd REAL NOT NULL,
                    vol_mult REAL NOT NULL,
                    buffer_used REAL NOT NULL,
                    leg1_realized_pnl_usd REAL NOT NULL DEFAULT 0.0,
                    leg1_realized_net_u REAL NOT NULL DEFAULT 0.0,
                    entry_fee_usd REAL NOT NULL DEFAULT 0.0,
                    opened_at TEXT NOT NULL,
                    signal_bar_ts TEXT NOT NULL,
                    exchange_order_id TEXT NOT NULL DEFAULT '',
                    sl_order_id TEXT NOT NULL DEFAULT '',
                    tp1_order_id TEXT NOT NULL DEFAULT '',
                    tp2_order_id TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS closed_trades (
                    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    tf TEXT NOT NULL,
                    setup_code TEXT NOT NULL,
                    ema TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    exit_arch TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    qty REAL NOT NULL,
                    unit_size REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    tp1_hit INTEGER NOT NULL,
                    leg2_sl_hit INTEGER NOT NULL,
                    tp2_hit INTEGER NOT NULL,
                    pnl_usd REAL NOT NULL,
                    fees_usd REAL NOT NULL,
                    net_u REAL NOT NULL,
                    opened_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS setup_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    tf TEXT NOT NULL,
                    setup_code TEXT NOT NULL,
                    net_u REAL NOT NULL,
                    recorded_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_setup_outcomes_key
                    ON setup_outcomes(symbol, tf, setup_code, id);

                CREATE TABLE IF NOT EXISTS daily_counters (
                    utc_date TEXT PRIMARY KEY,
                    trades_count INTEGER NOT NULL DEFAULT 0,
                    sl_count INTEGER NOT NULL DEFAULT 0,
                    realized_pnl_usd REAL NOT NULL DEFAULT 0.0,
                    start_equity REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS storm_cooldowns (
                    scope TEXT PRIMARY KEY,
                    until_ts TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS processed_signals (
                    signal_key TEXT PRIMARY KEY,
                    processed_at TEXT NOT NULL
                );
                """
            )

    def _init_defaults(self) -> None:
        defaults = {
            "initial_capital": self.initial_capital,
            "equity": self.initial_capital,
            "peak_equity": self.initial_capital,
            "compounding_step": 1,
            "tier_capital": self.initial_capital,
            "target_capital": self.initial_capital * (1.0 + self.step_gain_pct),
            "turbo_active": False,
            "consec_sl": 0,
        }
        now = utc_now_iso()
        with self._conn:
            for k, v in defaults.items():
                self._conn.execute(
                    "INSERT OR IGNORE INTO bot_state(key, value, updated_at) VALUES (?, ?, ?)",
                    (k, json.dumps(v), now),
                )

    # ---- Key-value bot state ----

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM bot_state WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def set_state(self, key: str, value: Any) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO bot_state(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, json.dumps(value), utc_now_iso()),
            )

    def get_equity(self) -> float:
        return float(self.get_state("equity", self.initial_capital))

    def get_peak_equity(self) -> float:
        return float(self.get_state("peak_equity", self.initial_capital))

    def get_compounding_info(self) -> Dict[str, Any]:
        return {
            "step": int(self.get_state("compounding_step", 1)),
            "tier_capital": float(self.get_state("tier_capital", self.initial_capital)),
            "target_capital": float(self.get_state("target_capital", self.initial_capital * (1.0 + self.step_gain_pct))),
            "equity": self.get_equity(),
            "peak_equity": self.get_peak_equity(),
            "turbo_active": bool(self.get_state("turbo_active", False)),
            "consec_sl": int(self.get_state("consec_sl", 0)),
        }

    def update_equity(
        self,
        new_equity: float,
        compounding_steps: int = 20,
        step_gain_pct: Optional[float] = None,
    ) -> bool:
        """Persist updated equity, peak equity, and advance compounding step if target reached."""
        gain = self.step_gain_pct if step_gain_pct is None else step_gain_pct
        peak = max(self.get_peak_equity(), new_equity)
        step = int(self.get_state("compounding_step", 1))
        tier_cap = float(self.get_state("tier_capital", self.initial_capital))
        target_cap = float(self.get_state("target_capital", self.initial_capital * (1.0 + gain)))

        stepped = False
        while step < compounding_steps and new_equity >= target_cap:
            step += 1
            tier_cap = self.initial_capital * ((1.0 + gain) ** (step - 1))
            target_cap = self.initial_capital * ((1.0 + gain) ** step)
            stepped = True

        now = utc_now_iso()
        with self._conn:
            for k, v in (
                ("equity", float(new_equity)),
                ("peak_equity", float(peak)),
                ("compounding_step", int(step)),
                ("tier_capital", float(tier_cap)),
                ("target_capital", float(target_cap)),
            ):
                self._conn.execute(
                    """
                    INSERT INTO bot_state(key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                    """,
                    (k, json.dumps(v), now),
                )
        return stepped

    # ---- Open positions ----

    def save_open_position(self, pos: OpenPosition) -> None:
        d = asdict(pos)
        d["tp1_filled"] = int(bool(pos.tp1_filled))
        d["leg2_sl_moved"] = int(bool(pos.leg2_sl_moved))
        if not d["opened_at"]:
            d["opened_at"] = utc_now_iso()
        cols = list(d.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "position_id")
        sql = (
            f"INSERT INTO open_positions ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(position_id) DO UPDATE SET {updates}"
        )
        with self._conn:
            self._conn.execute(sql, [d[c] for c in cols])

    def delete_open_position(self, position_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM open_positions WHERE position_id = ?", (position_id,))

    def get_open_positions(self, symbol: Optional[str] = None) -> List[OpenPosition]:
        if symbol:
            rows = self._conn.execute(
                "SELECT * FROM open_positions WHERE symbol = ? ORDER BY opened_at ASC", (symbol,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM open_positions ORDER BY opened_at ASC").fetchall()
        out: List[OpenPosition] = []
        for r in rows:
            data = dict(r)
            data["tp1_filled"] = bool(data["tp1_filled"])
            data["leg2_sl_moved"] = bool(data["leg2_sl_moved"])
            out.append(OpenPosition(**data))
        return out

    def get_open_sl_risk_usd(self) -> float:
        """Total worst-case SL risk (USD) across all currently open positions."""
        row = self._conn.execute("SELECT COALESCE(SUM(sl_risk_usd), 0.0) AS total FROM open_positions").fetchone()
        return float(row["total"]) if row else 0.0

    def get_open_mult_sum(self) -> float:
        row = self._conn.execute("SELECT COALESCE(SUM(vol_mult), 0.0) AS total FROM open_positions").fetchone()
        return float(row["total"]) if row else 0.0

    # ---- Closed trades & setup outcomes ----

    def record_closed_trade(self, trade: ClosedTrade, record_bayesian: bool = True) -> None:
        d = asdict(trade)
        d["tp1_hit"] = int(bool(trade.tp1_hit))
        d["leg2_sl_hit"] = int(bool(trade.leg2_sl_hit))
        d["tp2_hit"] = int(bool(trade.tp2_hit))
        cols = list(d.keys())
        placeholders = ", ".join(["?"] * len(cols))
        with self._conn:
            self._conn.execute(
                f"INSERT INTO closed_trades ({', '.join(cols)}) VALUES ({placeholders})",
                [d[c] for c in cols],
            )
            if record_bayesian:
                self._conn.execute(
                    """
                    INSERT INTO setup_outcomes(symbol, tf, setup_code, net_u, recorded_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (trade.symbol, trade.tf, trade.setup_code, float(trade.net_u), trade.closed_at or utc_now_iso()),
                )

    def get_closed_trades(self, limit: int = 100) -> List[ClosedTrade]:
        rows = self._conn.execute(
            "SELECT * FROM closed_trades ORDER BY trade_id DESC LIMIT ?", (int(limit),)
        ).fetchall()
        out: List[ClosedTrade] = []
        for r in rows:
            data = dict(r)
            data.pop("trade_id", None)
            data["tp1_hit"] = bool(data["tp1_hit"])
            data["leg2_sl_hit"] = bool(data["leg2_sl_hit"])
            data["tp2_hit"] = bool(data["tp2_hit"])
            out.append(ClosedTrade(**data))
        return out

    def get_setup_outcomes(self, symbol: str, tf: str, setup_code: str) -> List[float]:
        rows = self._conn.execute(
            """
            SELECT net_u FROM setup_outcomes
            WHERE symbol = ? AND tf = ? AND setup_code = ?
            ORDER BY id ASC
            """,
            (symbol, tf, setup_code),
        ).fetchall()
        return [float(r["net_u"]) for r in rows]

    def record_setup_outcome(self, symbol: str, tf: str, setup_code: str, net_u: float) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO setup_outcomes(symbol, tf, setup_code, net_u, recorded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, tf, setup_code, float(net_u), utc_now_iso()),
            )

    # ---- Daily counters ----

    def get_daily_counter(self, utc_date: Optional[str] = None) -> DailyCounter:
        day = utc_date or utc_today_str()
        row = self._conn.execute("SELECT * FROM daily_counters WHERE utc_date = ?", (day,)).fetchone()
        if row is None:
            eq = self.get_equity()
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO daily_counters(utc_date, trades_count, sl_count, realized_pnl_usd, start_equity)
                    VALUES (?, 0, 0, 0.0, ?)
                    """,
                    (day, eq),
                )
            return DailyCounter(utc_date=day, trades_count=0, sl_count=0, realized_pnl_usd=0.0, start_equity=eq)
        return DailyCounter(
            utc_date=row["utc_date"],
            trades_count=int(row["trades_count"]),
            sl_count=int(row["sl_count"]),
            realized_pnl_usd=float(row["realized_pnl_usd"]),
            start_equity=float(row["start_equity"]),
        )

    def increment_daily_trades(self, utc_date: Optional[str] = None) -> DailyCounter:
        dc = self.get_daily_counter(utc_date)
        with self._conn:
            self._conn.execute(
                "UPDATE daily_counters SET trades_count = trades_count + 1 WHERE utc_date = ?",
                (dc.utc_date,),
            )
        return self.get_daily_counter(dc.utc_date)

    def record_daily_pnl(self, pnl_usd: float, is_full_sl: bool = False, utc_date: Optional[str] = None) -> DailyCounter:
        dc = self.get_daily_counter(utc_date)
        sl_inc = 1 if is_full_sl else 0
        with self._conn:
            self._conn.execute(
                """
                UPDATE daily_counters
                SET realized_pnl_usd = realized_pnl_usd + ?,
                    sl_count = sl_count + ?
                WHERE utc_date = ?
                """,
                (float(pnl_usd), sl_inc, dc.utc_date),
            )
        return self.get_daily_counter(dc.utc_date)

    # ---- Storm Guard cooldowns ----

    def set_storm_cooldown(self, scope: str, until_dt: datetime, reason: str) -> None:
        if until_dt.tzinfo is None:
            until_dt = until_dt.replace(tzinfo=timezone.utc)
        until_iso = until_dt.astimezone(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO storm_cooldowns(scope, until_ts, reason, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(scope) DO UPDATE SET
                    until_ts = CASE WHEN excluded.until_ts > storm_cooldowns.until_ts THEN excluded.until_ts ELSE storm_cooldowns.until_ts END,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at
                """,
                (scope, until_iso, reason, utc_now_iso()),
            )

    def get_storm_cooldown(self, scope: str) -> Optional[Dict[str, str]]:
        row = self._conn.execute(
            "SELECT scope, until_ts, reason, updated_at FROM storm_cooldowns WHERE scope = ?",
            (scope,),
        ).fetchone()
        return dict(row) if row else None

    def is_in_storm_cooldown(self, scope: str, now: Optional[datetime] = None) -> tuple[bool, Optional[str]]:
        row = self.get_storm_cooldown(scope)
        if not row:
            return False, None
        now_dt = now if now is not None else datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        until_dt = datetime.fromisoformat(row["until_ts"])
        if until_dt.tzinfo is None:
            until_dt = until_dt.replace(tzinfo=timezone.utc)
        if now_dt < until_dt:
            return True, row["reason"]
        return False, None

    def get_active_cooldowns(self, now: Optional[datetime] = None) -> Dict[str, Dict[str, str]]:
        now_dt = now if now is not None else datetime.now(timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        rows = self._conn.execute("SELECT scope, until_ts, reason, updated_at FROM storm_cooldowns").fetchall()
        active: Dict[str, Dict[str, str]] = {}
        for r in rows:
            until_dt = datetime.fromisoformat(r["until_ts"])
            if until_dt.tzinfo is None:
                until_dt = until_dt.replace(tzinfo=timezone.utc)
            if now_dt < until_dt:
                active[r["scope"]] = dict(r)
        return active

    # ---- Signal idempotency ----

    def is_signal_processed(self, signal_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM processed_signals WHERE signal_key = ?", (signal_key,)
        ).fetchone()
        return row is not None

    def mark_signal_processed(self, signal_key: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO processed_signals(signal_key, processed_at) VALUES (?, ?)",
                (signal_key, utc_now_iso()),
            )
