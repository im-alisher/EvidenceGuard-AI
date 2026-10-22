"""Typed, strict loading of YAML experiment configurations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from evidenceguard.errors import ConfigError

CONFIG_VERSION: Final = 1
_NAME_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_LOG_LEVELS: Final = frozenset({"CRITICAL", "DEBUG", "ERROR", "INFO", "WARNING"})

_TOP_LEVEL_KEYS: Final = frozenset({"logging", "name", "paths", "seed", "version"})
_PATH_KEYS: Final = frozenset({"datasets", "outputs"})
_LOGGING_KEYS: Final = frozenset({"level"})


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Resolved filesystem paths used by an experiment."""

    datasets: Path
    outputs: Path


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Logging options for an experiment."""

    level: str = "INFO"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Validated configuration for one reproducible experiment."""

    version: int
    name: str
    seed: int
    paths: PathsConfig
    logging: LoggingConfig


def load_experiment_config(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> ExperimentConfig:
    """Load and validate an experiment configuration.

    Relative paths in the configuration are resolved against ``base_dir``,
    which defaults to the current working directory. Run commands from the
    repository root unless an explicit base directory is supplied.
    """
    config_path = Path(path)
    resolved_base_dir = Path.cwd() if base_dir is None else Path(base_dir)
    resolved_base_dir = resolved_base_dir.resolve()

    raw = _load_yaml_mapping(config_path)
    _check_keys(raw, _TOP_LEVEL_KEYS, "configuration")

    version = _expect_int(raw["version"], "version")
    if version != CONFIG_VERSION:
        raise ConfigError(
            f"Unsupported configuration version {version}; expected {CONFIG_VERSION}."
        )

    name = _expect_string(raw["name"], "name")
    if not _NAME_PATTERN.fullmatch(name):
        raise ConfigError(
            "Configuration name must start with a letter or digit and contain only "
            "letters, digits, dots, underscores, or hyphens."
        )

    seed = _expect_int(raw["seed"], "seed")
    if not 0 <= seed <= 2**32 - 1:
        raise ConfigError("Configuration seed must be between 0 and 4294967295.")

    paths = _parse_paths(raw["paths"], resolved_base_dir)
    logging_config = _parse_logging(raw["logging"])

    return ExperimentConfig(
        version=version,
        name=name,
        seed=seed,
        paths=paths,
        logging=logging_config,
    )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read configuration file {path}: {exc}") from exc

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    return _expect_mapping(loaded, "configuration root")


def _parse_paths(value: object, base_dir: Path) -> PathsConfig:
    mapping = _expect_mapping(value, "paths")
    _check_keys(mapping, _PATH_KEYS, "paths")

    datasets = _resolve_path(_expect_string(mapping["datasets"], "paths.datasets"), base_dir)
    outputs = _resolve_path(_expect_string(mapping["outputs"], "paths.outputs"), base_dir)
    return PathsConfig(datasets=datasets, outputs=outputs)


def _parse_logging(value: object) -> LoggingConfig:
    mapping = _expect_mapping(value, "logging")
    _check_keys(mapping, _LOGGING_KEYS, "logging")

    level = _expect_string(mapping["level"], "logging.level").upper()
    if level not in _LOG_LEVELS:
        allowed = ", ".join(sorted(_LOG_LEVELS))
        raise ConfigError(f"logging.level must be one of: {allowed}.")
    return LoggingConfig(level=level)


def _expect_mapping(value: object, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a YAML mapping.")

    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{where} contains a non-string key: {key!r}.")
        result[key] = item
    return result


def _expect_string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} must be a non-empty string.")
    return value


def _expect_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where} must be an integer.")
    return value


def _check_keys(mapping: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected.difference(mapping))
    if missing:
        raise ConfigError(f"Missing required key(s) in {where}: {', '.join(missing)}.")

    unknown = sorted(set(mapping).difference(expected))
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where}: {', '.join(unknown)}.")


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
