"""Helpers for importing optional third-party dependencies."""

from __future__ import annotations

import importlib
from types import ModuleType

from evidenceguard.errors import OptionalDependencyError


def import_optional_dependency(
    module_name: str,
    *,
    package_name: str,
    install_hint: str,
) -> ModuleType:
    """Import an optional module or raise a clear installation error."""

    if not isinstance(module_name, str) or not module_name.strip():
        raise ValueError("module_name must be a non-empty string.")
    if not isinstance(package_name, str) or not package_name.strip():
        raise ValueError("package_name must be a non-empty string.")
    if not isinstance(install_hint, str) or not install_hint.strip():
        raise ValueError("install_hint must be a non-empty string.")

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        message = (
            f"Optional dependency '{package_name}' is required for this feature. "
            f"Install it manually with: {install_hint}"
        )
        raise OptionalDependencyError(message) from exc
