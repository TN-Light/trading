# ============================================================================
# PROMETHEUS — Production Preflight Checks
# ============================================================================
"""Runtime preflight checks for production readiness."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import importlib.util
import yaml

from prometheus.config import load_config, SETTINGS_FILE, CREDENTIALS_FILE, CREDENTIAL_ENV_MAP

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_FALLBACK_MAP = {
    "broker.api_key": "broker.api_key",
    "broker.api_secret": "broker.api_secret",
    "broker.access_token": "broker.access_token",
    "zerodha.api_key": "broker.api_key",
    "zerodha.api_secret": "broker.api_secret",
    "zerodha.access_token": "broker.access_token",
    "telegram.bot_token": "interface.telegram.bot_token",
    "telegram.chat_id": "interface.telegram.chat_id",
    "groq.api_key": "ai.groq.api_key",
    "gemini.api_key": "ai.gemini.api_key",
}

PLACEHOLDER_TOKENS = (
    "your_",
    "replace",
    "changeme",
    "placeholder",
    "example",
    "dummy",
)


@dataclass
class PreflightReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        lines: List[str] = []
        if self.errors:
            lines.append("Errors:")
            for msg in self.errors:
                lines.append(f"  - {msg}")
        if self.warnings:
            lines.append("Warnings:")
            for msg in self.warnings:
                lines.append(f"  - {msg}")
        if self.notes:
            lines.append("Notes:")
            for msg in self.notes:
                lines.append(f"  - {msg}")
        if not lines:
            lines.append("Preflight OK")
        return "\n".join(lines)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _ensure_dir(path: Path, report: PreflightReport, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        report.errors.append(f"{label} directory not writable: {path} ({exc})")
        return
    if not os.access(path, os.W_OK):
        report.errors.append(f"{label} directory not writable: {path}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _get_nested(data: Dict[str, Any], key_path: str) -> Optional[Any]:
    value: Any = data
    for key in key_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
        if value is None:
            return None
    return value


def _get_env_override(key_path: str) -> Optional[str]:
    env_key = CREDENTIAL_ENV_MAP.get(key_path)
    if not env_key:
        return None
    value = os.getenv(env_key, "").strip()
    return value or None


def _get_config_override(config: Dict[str, Any], key_path: str) -> Optional[Any]:
    config_key = CONFIG_FALLBACK_MAP.get(key_path)
    if not config_key:
        return None
    return _get_nested(config, config_key)


def _is_placeholder(value: Optional[Any]) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    lowered = text.lower()
    return any(token in lowered for token in PLACEHOLDER_TOKENS)


def _collect_credential_value(
    key_path: str,
    creds: Dict[str, Any],
    config: Dict[str, Any],
) -> Optional[Any]:
    env_value = _get_env_override(key_path)
    if env_value:
        return env_value
    value = _get_nested(creds, key_path)
    if value not in (None, ""):
        return value
    return _get_config_override(config, key_path)


def run_preflight(
    config_path: Optional[str],
    mode: str,
    allow_paper_fallback: bool,
) -> PreflightReport:
    report = PreflightReport()
    cfg_path = Path(config_path) if config_path else SETTINGS_FILE

    try:
        config = load_config(str(cfg_path))
    except FileNotFoundError as exc:
        report.errors.append(str(exc))
        return report

    report.notes.append(f"Config: {cfg_path}")
    report.notes.append(f"Mode: {mode}")
    report.notes.append(f"Allow paper fallback: {allow_paper_fallback}")

    logging_cfg = config.get("logging", {})
    log_file = logging_cfg.get("file", "logs/prometheus.log")
    _ensure_dir(_resolve_path(log_file).parent, report, "Log")

    data_cfg = config.get("data", {})
    cache_dir = data_cfg.get("cache_dir", "data/cache")
    db_path = data_cfg.get("db_path", "data/prometheus.db")
    _ensure_dir(_resolve_path(cache_dir), report, "Cache")
    _ensure_dir(_resolve_path(db_path).parent, report, "Data")

    capital_cfg = config.get("capital", {})
    initial_capital = capital_cfg.get("initial", 0)
    if initial_capital <= 0:
        report.errors.append("capital.initial must be > 0")

    risk_cfg = config.get("risk", {})
    if risk_cfg.get("max_daily_loss", 0) <= 0:
        report.errors.append("risk.max_daily_loss must be > 0")
    if risk_cfg.get("max_weekly_loss", 0) <= 0:
        report.errors.append("risk.max_weekly_loss must be > 0")

    market_cfg = config.get("market", {})
    if not market_cfg.get("indices"):
        report.errors.append("market.indices is empty")

    intraday_v2_cfg = config.get("intraday", {}).get("v2", {})
    event_gate = intraday_v2_cfg.get("event_risk_gate", {})
    if event_gate.get("enabled"):
        event_calendar = event_gate.get("event_calendar_file", "")
        if event_calendar:
            calendar_path = _resolve_path(event_calendar)
            if not calendar_path.exists():
                report.warnings.append(
                    f"event calendar file missing: {calendar_path}"
                )

    live_mode = mode in ("semi_auto", "full_auto")
    creds_path = (Path(config_path).parent / "credentials.yaml") if config_path else CREDENTIALS_FILE
    creds = _load_yaml(creds_path)

    if live_mode:
        if importlib.util.find_spec("kiteconnect") is None:
            report.errors.append("kiteconnect not installed (required for live mode)")

        for key_path in ("broker.api_key", "broker.api_secret", "broker.access_token"):
            value = _collect_credential_value(key_path, creds, config)
            if _is_placeholder(value):
                report.errors.append(f"Missing credential: {key_path}")

    telegram_cfg = config.get("interface", {}).get("telegram", {})
    if telegram_cfg.get("enabled", False):
        for key_path in ("telegram.bot_token", "telegram.chat_id"):
            value = _collect_credential_value(key_path, creds, config)
            if _is_placeholder(value):
                report.warnings.append(f"Telegram enabled but {key_path} is missing")

    ai_cfg = config.get("ai", {})
    if ai_cfg.get("groq", {}).get("enabled", False):
        value = _collect_credential_value("groq.api_key", creds, config)
        if _is_placeholder(value):
            report.warnings.append("Groq enabled but groq.api_key is missing")

    if ai_cfg.get("gemini", {}).get("enabled", False):
        value = _collect_credential_value("gemini.api_key", creds, config)
        if _is_placeholder(value):
            report.warnings.append("Gemini enabled but gemini.api_key is missing")

    return report
