from __future__ import annotations

import random

import numpy as np
import pytest

from evidenceguard.reproducibility import MAX_SEED, derive_seed, seed_everything


def test_seed_everything_repeats_python_and_numpy_sequences() -> None:
    seed_everything(7)
    first_python = random.random()
    first_numpy = np.random.random(3).tolist()

    seed_everything(7)
    assert random.random() == first_python
    assert np.random.random(3).tolist() == first_numpy


def test_derive_seed_is_stable_and_component_order_matters() -> None:
    first = derive_seed("scifact", "test", 17)
    second = derive_seed("scifact", "test", 17)
    reordered = derive_seed(17, "test", "scifact")

    assert first == second
    assert first != reordered
    assert 0 <= first <= MAX_SEED


def test_derive_seed_requires_components() -> None:
    with pytest.raises(ValueError, match="At least one component"):
        derive_seed()


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1, True, 1.5, "13"])
def test_invalid_seeds_are_rejected(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        seed_everything(seed)  # type: ignore[arg-type]
