"""Configuration management for smart-run.

Resolution order (later wins):
    1. built-in defaults
    2. config file  (./.smart-run.json, then ~/.smart-run/config.json)
    3. environment variables (SMART_RUN_*)
    4. command-line flags

All HTTP work uses the standard library only, so there are no third-party
dependencies to install -- the tool works as soon as Python is present.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

ENV_PREFIX = "SMART_RUN_"

CONFIG_FILE_CANDIDATES = (
    Path(".smart-run.json"),
    Path.home() / ".smart-run" / "config.json",
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass
class LLMConfig:
    """Optional OpenAI-compatible chat completion endpoint.

    If ``base_url`` and ``api_key`` are set, the analyzer will additionally ask
    the model for a plain-language explanation of the crash. Falls back to the
    built-in regex library otherwise (or when the request fails).
    """

    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout: float = 30.0


@dataclass
class Config:
    # --- capture / analysis -------------------------------------------------
    tail_lines: int = 60
    analyze_on_success: bool = False
    use_llm: bool = False
    llm: LLMConfig = field(default_factory=LLMConfig)

    # --- notification --------------------------------------------------------
    feishu_webhook: str = ""
    wecom_webhook: str = ""
    notify_on_success: bool = False
    mention: str = ""

    # --- runtime -------------------------------------------------------------
    log_file: str = ""
    passthrough: bool = True
    shell: bool = False
    cwd: str = ""
    timeout: Optional[float] = None

    # ------------------------------------------------------------------ io ---
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ---------------------------------------------------------------- helpers
    @property
    def has_notifier(self) -> bool:
        return bool(self.feishu_webhook or self.wecom_webhook)


def _load_config_file() -> dict[str, Any]:
    for candidate in CONFIG_FILE_CANDIDATES:
        try:
            path = candidate.resolve()
        except OSError:
            continue
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError):
                continue
    return {}


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: Optional[float]) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def from_env() -> dict[str, Any]:
    """Read overrides from SMART_RUN_* environment variables."""
    g = os.environ.get
    llm = {
        "enabled": _env_bool(f"{ENV_PREFIX}LLM_ENABLED"),
        "base_url": g(f"{ENV_PREFIX}LLM_BASE_URL", ""),
        "api_key": g(f"{ENV_PREFIX}LLM_API_KEY", ""),
        "model": g(f"{ENV_PREFIX}LLM_MODEL", "gpt-4o-mini"),
        "timeout": _coerce_float(g(f"{ENV_PREFIX}LLM_TIMEOUT"), 30.0),
    }
    return {
        "tail_lines": _coerce_int(g(f"{ENV_PREFIX}TAIL_LINES"), 60),
        "analyze_on_success": _env_bool(f"{ENV_PREFIX}ANALYZE_ON_SUCCESS"),
        "use_llm": _env_bool(f"{ENV_PREFIX}USE_LLM"),
        "llm": llm,
        "feishu_webhook": g(f"{ENV_PREFIX}FEISHU_WEBHOOK", ""),
        "wecom_webhook": g(f"{ENV_PREFIX}WECOM_WEBHOOK", ""),
        "notify_on_success": _env_bool(f"{ENV_PREFIX}NOTIFY_ON_SUCCESS"),
        "mention": g(f"{ENV_PREFIX}MENTION", ""),
        "log_file": g(f"{ENV_PREFIX}LOG_FILE", ""),
        "passthrough": _env_bool(f"{ENV_PREFIX}PASSTHROUGH", True),
        "shell": _env_bool(f"{ENV_PREFIX}SHELL"),
        "cwd": g(f"{ENV_PREFIX}CWD", ""),
        "timeout": _coerce_float(g(f"{ENV_PREFIX}TIMEOUT"), None),
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_config(cli_overrides: Optional[dict[str, Any]] = None) -> Config:
    """Assemble a :class:`Config` from defaults, file, env, and CLI flags."""
    merged: dict[str, Any] = {}
    merged = _deep_merge(merged, _load_config_file())
    merged = _deep_merge(merged, from_env())
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    llm_raw = merged.pop("llm", None) or {}
    if not isinstance(llm_raw, dict):
        llm_raw = {}

    enabled = llm_raw.get("enabled", False)
    if not isinstance(enabled, bool):
        enabled = str(enabled).strip().lower() in {"1", "true", "yes", "on", "y"}

    llm_cfg = LLMConfig(
        enabled=enabled,
        base_url=str(llm_raw.get("base_url", "")),
        api_key=str(llm_raw.get("api_key", "")),
        model=str(llm_raw.get("model", "gpt-4o-mini")),
        timeout=_coerce_float(llm_raw.get("timeout"), 30.0),
    )

    filtered = {k: v for k, v in merged.items() if k in Config.__dataclass_fields__}
    return Config(llm=llm_cfg, **filtered)
