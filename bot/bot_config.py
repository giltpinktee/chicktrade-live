"""Configuration loader for CHICKTRADE Live & Paper Trading Bot.

Reads a `.env` file using a plain Python parser (no external `python-dotenv` dependency)
and merges with process environment variables and safe defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_RULES_PATH = PROJECT_ROOT / "core" / "core_strategy_rules.json"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chicktrade_state.db"

BUFFER_CONCURRENT_MAP: Dict[int, int] = {
    250: 2,
    300: 3,
    400: 5,
    500: 7,
}


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a standard KEY=VALUE .env file without external dependencies."""
    values: Dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        # Strip matching surrounding quotes
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        else:
            # Strip inline comment when unquoted
            if " #" in val:
                val = val.split(" #", 1)[0].strip()
        values[key] = val
    return values


def _parse_bool(raw: Optional[str], default: bool = False) -> bool:
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_int(raw: Optional[str], default: int) -> int:
    if raw is None or raw.strip() == "":
        return default
    return int(float(raw.strip()))


def _parse_float(raw: Optional[str], default: float) -> float:
    if raw is None or raw.strip() == "":
        return default
    return float(raw.strip())


def resolve_max_concurrent(buffer_units: float, override: Optional[int] = None) -> int:
    """Auto-map buffer units to max concurrent trades unless explicitly overridden."""
    if override is not None and override > 0:
        return int(override)
    b_int = int(round(buffer_units))
    if b_int in BUFFER_CONCURRENT_MAP:
        return BUFFER_CONCURRENT_MAP[b_int]
    # Nearest safe lower mapping
    if b_int <= 250:
        return 2
    if b_int <= 300:
        return 3
    if b_int <= 400:
        return 5
    return 7


@dataclass
class BotConfig:
    execution_mode: str = "PAPER"  # PAPER | TESTNET | LIVE
    buffer_units: float = 400.0
    max_concurrent_override: Optional[int] = None

    # Auto-Turbo settings (400G -> 230G with hard 1.10x worst-case SL floor)
    auto_turbo_enabled: bool = False
    auto_turbo_buffer_units: float = 230.0
    auto_turbo_trigger_ratio: float = 1.25
    auto_turbo_disengage_ratio: float = 1.15
    auto_turbo_floor_ratio: float = 1.10

    # Capital & risk limits
    initial_capital: float = 10_000.0
    max_trades_per_day: int = 10
    compounding_steps: int = 20
    step_gain_pct: float = 0.08
    atr_to_unit_mult: float = 0.5
    max_open_mult_cap: float = 2.4

    # Execution fees & slippage (Binance USD-M Futures)
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0005
    slippage_bps: float = 1.0

    # Runtime & notifications
    poll_interval_seconds: int = 30
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Exchange credentials & live safety switch
    binance_api_key: str = ""
    binance_api_secret: str = ""
    live_confirmed: bool = False

    # Paths
    rules_path: Path = field(default_factory=lambda: DEFAULT_RULES_PATH)
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)

    @property
    def max_concurrent_trades(self) -> int:
        return resolve_max_concurrent(self.buffer_units, self.max_concurrent_override)

    @property
    def protected_floor_usd(self) -> float:
        return self.initial_capital * self.auto_turbo_floor_ratio

    def max_concurrent_for_buffer(self, active_buffer: float) -> int:
        return resolve_max_concurrent(active_buffer, self.max_concurrent_override)

    def validate(self) -> None:
        mode = self.execution_mode.upper().strip()
        if mode not in ("PAPER", "TESTNET", "LIVE"):
            raise ValueError(f"Unsupported EXECUTION_MODE={self.execution_mode!r}. Use PAPER, TESTNET, or LIVE.")
        self.execution_mode = mode
        if self.initial_capital <= 0:
            raise ValueError("INITIAL_CAPITAL must be > 0")
        if self.buffer_units <= 0:
            raise ValueError("BUFFER_UNITS must be > 0")
        if self.auto_turbo_buffer_units <= 0:
            raise ValueError("AUTO_TURBO_BUFFER_UNITS must be > 0")
        if self.execution_mode == "LIVE" and not self.live_confirmed:
            raise PermissionError(
                "EXECUTION_MODE=LIVE requires LIVE_CONFIRMED=true in .env or environment."
            )

    @classmethod
    def load(
        cls,
        env_path: Optional[Path | str] = None,
        db_path: Optional[Path | str] = None,
        rules_path: Optional[Path | str] = None,
        environ: Optional[Dict[str, str]] = None,
    ) -> "BotConfig":
        """Load configuration from `.env` file with OS environment overrides."""
        resolved_env = Path(env_path) if env_path else DEFAULT_ENV_PATH
        file_vals = parse_env_file(resolved_env)
        env_src = dict(os.environ) if environ is None else dict(environ)

        def get(key: str, default: Optional[str] = None) -> Optional[str]:
            if key in env_src and env_src[key] != "":
                return env_src[key]
            return file_vals.get(key, default)

        raw_max_conc = get("MAX_CONCURRENT_TRADES", "")
        max_conc_override = int(raw_max_conc) if (raw_max_conc and raw_max_conc.strip().isdigit()) else None

        cfg = cls(
            execution_mode=(get("EXECUTION_MODE", "PAPER") or "PAPER").upper().strip(),
            buffer_units=_parse_float(get("BUFFER_UNITS", "400"), 400.0),
            max_concurrent_override=max_conc_override,
            auto_turbo_enabled=_parse_bool(get("AUTO_TURBO_ENABLED", "false"), False),
            auto_turbo_buffer_units=_parse_float(get("AUTO_TURBO_BUFFER_UNITS", "230"), 230.0),
            auto_turbo_trigger_ratio=_parse_float(get("AUTO_TURBO_TRIGGER_RATIO", "1.25"), 1.25),
            auto_turbo_disengage_ratio=_parse_float(get("AUTO_TURBO_DISENGAGE_RATIO", "1.15"), 1.15),
            auto_turbo_floor_ratio=_parse_float(get("AUTO_TURBO_FLOOR_RATIO", "1.10"), 1.10),
            initial_capital=_parse_float(get("INITIAL_CAPITAL", "10000"), 10_000.0),
            max_trades_per_day=_parse_int(get("MAX_TRADES_PER_DAY", "10"), 10),
            poll_interval_seconds=_parse_int(get("POLL_INTERVAL_SECONDS", "30"), 30),
            telegram_bot_token=(get("TELEGRAM_BOT_TOKEN", "") or "").strip(),
            telegram_chat_id=(get("TELEGRAM_CHAT_ID", "") or "").strip(),
            binance_api_key=(get("BINANCE_API_KEY", "") or "").strip(),
            binance_api_secret=(get("BINANCE_API_SECRET", "") or "").strip(),
            live_confirmed=_parse_bool(get("LIVE_CONFIRMED", "false"), False),
            rules_path=Path(rules_path) if rules_path else DEFAULT_RULES_PATH,
            db_path=Path(db_path) if db_path else DEFAULT_DB_PATH,
        )
        cfg.validate()
        return cfg
