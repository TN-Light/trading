# ============================================================================
# PROMETHEUS — Configuration Loader
# ============================================================================
"""
Loads and validates system configuration from YAML files.
Provides typed access to all settings throughout the system.
"""

import os
import yaml
from typing import Any, Optional
from pathlib import Path


_config: dict = {}
_credentials: dict = {}

CONFIG_DIR = Path(__file__).parent
SETTINGS_FILE = CONFIG_DIR / "settings.yaml"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.yaml"

CREDENTIAL_ENV_MAP = {
    "broker.api_key": "PROM_BROKER_API_KEY",
    "broker.api_secret": "PROM_BROKER_API_SECRET",
    "broker.access_token": "PROM_BROKER_ACCESS_TOKEN",
    "zerodha.api_key": "PROM_BROKER_API_KEY",
    "zerodha.api_secret": "PROM_BROKER_API_SECRET",
    "zerodha.access_token": "PROM_BROKER_ACCESS_TOKEN",
    "telegram.bot_token": "PROM_TELEGRAM_BOT_TOKEN",
    "telegram.chat_id": "PROM_TELEGRAM_CHAT_ID",
    "groq.api_key": "PROM_GROQ_API_KEY",
    "gemini.api_key": "PROM_GEMINI_API_KEY",
    "angelone.api_key": "PROM_ANGELONE_API_KEY",
    "angelone.client_code": "PROM_ANGELONE_CLIENT_CODE",
    "angelone.password": "PROM_ANGELONE_PASSWORD",
    "angelone.totp_secret": "PROM_ANGELONE_TOTP_SECRET",
}


def _get_env_credential(key_path: str) -> Optional[str]:
    env_key = CREDENTIAL_ENV_MAP.get(key_path)
    if not env_key:
        return None
    value = os.getenv(env_key, "").strip()
    return value or None


def load_config(settings_path: Optional[str] = None) -> dict:
    """Load main settings configuration."""
    global _config
    path = Path(settings_path) if settings_path else SETTINGS_FILE

    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    with open(path, "r") as f:
        _config = yaml.safe_load(f)

    return _config


def load_credentials(creds_path: Optional[str] = None) -> dict:
    """Load credentials (API keys, tokens)."""
    global _credentials
    path = Path(creds_path) if creds_path else CREDENTIALS_FILE

    if not path.exists():
        raise FileNotFoundError(f"Credentials file not found: {path}")

    with open(path, "r") as f:
        _credentials = yaml.safe_load(f)

    return _credentials


def get(key_path: str, default: Any = None) -> Any:
    """
    Get a config value by dot-separated path.
    Example: get("risk.max_daily_loss") -> 5000
    """
    if not _config:
        load_config()

    keys = key_path.split(".")
    value = _config
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default
    return value


def get_credential(key_path: str) -> Optional[str]:
    """
    Get a credential value by dot-separated path.
    Example: get_credential("zerodha.api_key")
    Falls back: broker.* -> zerodha.* for backwards compatibility.
    """
    env_override = _get_env_credential(key_path)
    if env_override:
        return env_override

    if not _credentials:
        try:
            load_credentials()
        except FileNotFoundError:
            return None

    keys = key_path.split(".")
    value = _credentials
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
        if value is None:
            # Fallback: "broker.X" -> "zerodha.X"
            if keys[0] == "broker":
                return get_credential("zerodha." + ".".join(keys[1:]))
            return None
    return value


def get_risk_limits() -> dict:
    """Get all risk management limits as a dict."""
    return get("risk", {})


def get_capital_config() -> dict:
    """Get capital configuration."""
    return get("capital", {})


def get_mode() -> str:
    """Get current operating mode."""
    return get("system.mode", "paper")


def is_paper_mode() -> bool:
    """Check if running in paper trading mode."""
    return get_mode() == "paper"
