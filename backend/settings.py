"""App settings: provider/model in a JSON file, API keys in the OS credential
store (Windows Credential Manager / macOS Keychain via `keyring`). Keys are
write-only through the API — reads only ever reveal whether a key exists.
Headless systems without a keyring backend fall back to an owner-only file.
"""
import json
import logging
import os
import stat
from pathlib import Path

import platformdirs
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # keys + optional defaults

log = logging.getLogger(__name__)

APP_NAME = "production-rag"
CONFIG_DIR = Path(os.environ.get("RAG_APP_DIR") or platformdirs.user_config_dir(APP_NAME))
SETTINGS_FILE = CONFIG_DIR / "settings.json"
KEY_FALLBACK_FILE = CONFIG_DIR / "keys.json"

# provider name (init_chat_model) -> env var LangChain reads the key from
PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}
DEFAULTS = {
    "provider": os.environ.get("RAG_PROVIDER", "anthropic"),
    "model": os.environ.get("RAG_MODEL", "claude-sonnet-5"),
}


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        return DEFAULTS | json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    return dict(DEFAULTS)


def save_settings(settings: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def _fallback_keys() -> dict:
    if KEY_FALLBACK_FILE.exists():
        return json.loads(KEY_FALLBACK_FILE.read_text(encoding="utf-8"))
    return {}


def _save_fallback_keys(keys: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    KEY_FALLBACK_FILE.write_text(json.dumps(keys), encoding="utf-8")
    KEY_FALLBACK_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)


def set_api_key(provider: str, api_key: str):
    if provider not in PROVIDER_ENV:
        raise ValueError(f"Unknown provider: {provider}")
    try:
        import keyring

        keyring.set_password(APP_NAME, provider, api_key)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # keyring backends can raise native panics (BaseException)
        log.warning("No usable keyring backend; storing key in an owner-only file")
        keys = _fallback_keys()
        keys[provider] = api_key
        _save_fallback_keys(keys)


def get_api_key(provider: str) -> str | None:
    try:
        import keyring

        key = keyring.get_password(APP_NAME, provider)
        if key:
            return key
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # see set_api_key
        pass
    return _fallback_keys().get(provider) or os.environ.get(PROVIDER_ENV.get(provider, ""))


def export_key_to_env(provider: str) -> bool:
    """Put the stored key where LangChain's provider client will find it."""
    key = get_api_key(provider)
    if key:
        os.environ[PROVIDER_ENV[provider]] = key
        return True
    return False
