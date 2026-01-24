"""
config.py - load Alpaca API credentials

Reads from ~/.alpaca-keys:
  ALPACA_API_KEY=your_key
  ALPACA_SECRET_KEY=your_secret
"""

import os
from pathlib import Path

KEYS_FILE = Path.home() / ".alpaca-keys"


def load_keys() -> tuple[str, str]:
    """Load API keys from ~/.alpaca-keys"""
    if not KEYS_FILE.exists():
        raise FileNotFoundError(
            f"API keys not found. Create {KEYS_FILE} with:\n"
            "ALPACA_API_KEY=your_key\n"
            "ALPACA_SECRET_KEY=your_secret"
        )

    keys = {}
    for line in KEYS_FILE.read_text().strip().split('\n'):
        line = line.strip()
        if line and '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            keys[key.strip()] = value.strip()

    api_key = keys.get('ALPACA_API_KEY')
    secret_key = keys.get('ALPACA_SECRET_KEY')

    if not api_key or not secret_key:
        raise ValueError(f"Missing keys in {KEYS_FILE}")

    return api_key, secret_key


def keys_exist() -> bool:
    """Check if keys file exists"""
    return KEYS_FILE.exists()
