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
    """
    if not _credentials:
        load_credentials()

    keys = key_path.split(".")
    value = _credentials
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
        if value is None:
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
