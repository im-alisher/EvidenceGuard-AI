"""Project-specific exception types."""

from __future__ import annotations


class EvidenceGuardError(Exception):
    """Base class for expected EvidenceGuard errors."""


class ConfigError(EvidenceGuardError):
    """Raised when a configuration file is missing, malformed, or invalid."""


class DataValidationError(EvidenceGuardError):
    """Raised when a dataset or benchmark transformation is invalid."""
