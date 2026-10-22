from __future__ import annotations

from io import StringIO

import pytest

from evidenceguard.logging_utils import configure_logging, get_logger


def test_configure_logging_writes_one_formatted_record() -> None:
    stream = StringIO()
    configure_logging("DEBUG", stream=stream, force=True)

    get_logger("config").debug("loaded configuration")

    output = stream.getvalue()
    assert output.count("loaded configuration") == 1
    assert "DEBUG" in output
    assert "evidenceguard.config" in output


def test_repeated_configuration_does_not_duplicate_handlers() -> None:
    stream = StringIO()
    configure_logging("INFO", stream=stream, force=True)
    configure_logging("INFO")

    get_logger().info("single record")

    assert stream.getvalue().count("single record") == 1


def test_level_filtering_is_applied() -> None:
    stream = StringIO()
    configure_logging("WARNING", stream=stream, force=True)

    logger = get_logger("filtering")
    logger.info("hidden")
    logger.warning("visible")

    output = stream.getvalue()
    assert "hidden" not in output
    assert "visible" in output


@pytest.mark.parametrize("level", ["VERBOSE", 15, True])
def test_invalid_log_level_is_rejected(level: object) -> None:
    with pytest.raises(ValueError, match="log level|Log level"):
        configure_logging(level, stream=StringIO(), force=True)  # type: ignore[arg-type]
