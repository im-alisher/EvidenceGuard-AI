"""Deterministic seed and identifier helpers for reproducible experiments."""

from __future__ import annotations

import hashlib
import random
from typing import Final

import numpy as np

MAX_SEED: Final = 2**32 - 1
_SEED_FIELD_SEPARATOR: Final = "\x1f"


def validate_seed(seed: int) -> int:
    """Validate and return a seed accepted by Python and NumPy."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Seed must be an integer, not a boolean or other type.")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"Seed must be between 0 and {MAX_SEED}.")
    return seed


def seed_everything(seed: int) -> int:
    """Seed Python and NumPy global random generators, returning the seed."""
    valid_seed = validate_seed(seed)
    random.seed(valid_seed)
    np.random.seed(valid_seed)
    return valid_seed


def derive_seed(*parts: object) -> int:
    """Derive a stable 32-bit seed from ordered, human-readable components.

    The derivation is independent of Python's hash randomization so that seeds
    remain stable across processes and machines.
    """
    if not parts:
        raise ValueError("At least one component is required to derive a seed.")
    payload = _SEED_FIELD_SEPARATOR.join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big")
