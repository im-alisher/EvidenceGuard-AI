from __future__ import annotations

import pytest

from evidenceguard.data.splits import QuerySplit, assign_query_splits
from evidenceguard.errors import DataValidationError


def _all_ids(split: QuerySplit) -> set[str]:
    return {*split.train, *split.validation, *split.test}


def test_query_split_is_deterministic_and_disjoint() -> None:
    query_ids = [f"q-{index:02d}" for index in range(20)]

    first = assign_query_splits(query_ids, seed=13)
    second = assign_query_splits(reversed(query_ids), seed=13)

    assert first == second
    assert _all_ids(first) == set(query_ids)
    assert len(first.train) == 12
    assert len(first.validation) == 4
    assert len(first.test) == 4
    assert not set(first.train).intersection(first.validation)
    assert not set(first.train).intersection(first.test)
    assert not set(first.validation).intersection(first.test)


def test_query_split_prevents_duplicate_queries() -> None:
    with pytest.raises(DataValidationError, match="unique"):
        assign_query_splits(["q-1", "q-1"], seed=13)


def test_query_split_requires_queries() -> None:
    with pytest.raises(DataValidationError, match="At least one query"):
        assign_query_splits([], seed=13)


def test_query_split_validates_seed() -> None:
    with pytest.raises(DataValidationError, match="Seed"):
        assign_query_splits(["q-1"], seed=-1)


@pytest.mark.parametrize(
    "fractions",
    [
        (0.5, 0.5, 0.5),
        (-0.1, 0.6, 0.5),
        (0.6, 0.2),
    ],
)
def test_query_split_validates_fractions(
    fractions: tuple[float, ...],
) -> None:
    with pytest.raises(DataValidationError, match="fractions"):
        assign_query_splits(
            ["q-1", "q-2", "q-3"],
            seed=13,
            fractions=fractions,  # type: ignore[arg-type]
        )


def test_query_split_exposes_membership() -> None:
    split = assign_query_splits(["q-1", "q-2", "q-3", "q-4", "q-5"], seed=13)

    for query_id in _all_ids(split):
        assert split.split_for(query_id) in {"train", "validation", "test"}

    with pytest.raises(DataValidationError, match="not present"):
        split.split_for("q-unknown")
