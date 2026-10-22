"""Command-line entry point for configured experiment runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from evidenceguard.config import load_experiment_config
from evidenceguard.errors import EvidenceGuardError
from evidenceguard.experiments.runner import run_configured_experiment
from evidenceguard.logging_utils import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    """Return the experiment command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m evidenceguard.experiments",
        description="Run a configured local EvidenceGuard experiment.",
    )
    parser.add_argument("config", type=Path, help="Path to a YAML experiment config.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for resolving relative config paths.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output artifact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        config = load_experiment_config(args.config, base_dir=args.base_dir)
        configure_logging(config.logging.level)
        logger = get_logger("experiments")
        output_path = run_configured_experiment(
            config,
            overwrite=args.overwrite,
        )
    except (EvidenceGuardError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    logger.info("Wrote experiment artifact: %s", output_path)
    print(output_path)
    return 0
