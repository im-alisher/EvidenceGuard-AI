from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from evidenceguard.config import (
    CONFIG_VERSION,
    ExperimentConfig,
    load_experiment_config,
)
from evidenceguard.errors import ConfigError


def _valid_config() -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "name": "offline-smoke",
        "seed": 13,
        "paths": {
            "datasets": "datasets",
            "outputs": "reports/runs",
        },
        "logging": {"level": "INFO"},
    }


def _write_config(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_loads_valid_config_and_resolves_relative_paths(tmp_path: Path) -> None:
    path = _write_config(tmp_path, _valid_config())

    config = load_experiment_config(path, base_dir=tmp_path)

    assert isinstance(config, ExperimentConfig)
    assert config.version == CONFIG_VERSION
    assert config.name == "offline-smoke"
    assert config.seed == 13
    assert config.logging.level == "INFO"
    assert config.paths.datasets == (tmp_path / "datasets").resolve()
    assert config.paths.outputs == (tmp_path / "reports" / "runs").resolve()


def test_loads_smoke_config_from_repository_root() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    config = load_experiment_config(
        repository_root / "configs" / "smoke.yaml",
        base_dir=repository_root,
    )

    assert config.name == "smoke"
    assert config.paths.datasets == (repository_root / "datasets").resolve()
    assert config.paths.outputs == (repository_root / "reports" / "runs").resolve()


def test_missing_config_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_experiment_config(tmp_path / "missing.yaml", base_dir=tmp_path)


def test_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("version: [unterminated", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_experiment_config(path, base_dir=tmp_path)


def test_root_must_be_mapping(tmp_path: Path) -> None:
    path = _write_config(tmp_path, ["not", "a", "mapping"])

    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_experiment_config(path, base_dir=tmp_path)


@pytest.mark.parametrize("missing_key", ["version", "name", "seed", "paths", "logging"])
def test_required_top_level_keys_are_enforced(tmp_path: Path, missing_key: str) -> None:
    data = _valid_config()
    del data[missing_key]
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="Missing required key"):
        load_experiment_config(path, base_dir=tmp_path)


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    data = _valid_config()
    data["unexpected"] = True
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="Unknown key"):
        load_experiment_config(path, base_dir=tmp_path)


def test_unknown_nested_key_is_rejected(tmp_path: Path) -> None:
    data = _valid_config()
    data["logging"]["unexpected"] = True
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="Unknown key"):
        load_experiment_config(path, base_dir=tmp_path)


def test_unsupported_version_is_rejected(tmp_path: Path) -> None:
    data = _valid_config()
    data["version"] = CONFIG_VERSION + 1
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="Unsupported configuration version"):
        load_experiment_config(path, base_dir=tmp_path)


@pytest.mark.parametrize("name", ["", "has spaces", "../escape"])
def test_invalid_name_is_rejected(tmp_path: Path, name: str) -> None:
    data = _valid_config()
    data["name"] = name
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="name"):
        load_experiment_config(path, base_dir=tmp_path)


@pytest.mark.parametrize("seed", [-1, 2**32, True, "13"])
def test_invalid_seed_is_rejected(tmp_path: Path, seed: object) -> None:
    data = _valid_config()
    data["seed"] = seed
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="seed"):
        load_experiment_config(path, base_dir=tmp_path)


@pytest.mark.parametrize("field", ["datasets", "outputs"])
def test_empty_path_is_rejected(tmp_path: Path, field: str) -> None:
    data = _valid_config()
    data["paths"][field] = " "
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match=field):
        load_experiment_config(path, base_dir=tmp_path)


def test_invalid_log_level_is_rejected(tmp_path: Path) -> None:
    data = _valid_config()
    data["logging"]["level"] = "VERBOSE"
    path = _write_config(tmp_path, data)

    with pytest.raises(ConfigError, match="logging.level"):
        load_experiment_config(path, base_dir=tmp_path)
