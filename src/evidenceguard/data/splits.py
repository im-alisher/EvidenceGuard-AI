"""Deterministic query-level splits that prevent query leakage."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from evidenceguard.errors import DataValidationError
from evidenceguard.reproducibility import derive_seed, validate_seed


@dataclass(frozen=True, slots=True)
class QuerySplit:
    """Disjoint query ID assignments for train, validation, and test."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = {
            "train": tuple(self.train),
            "validation": tuple(self.validation),
            "test": tuple(self.test),
        }
        seen: set[str] = set()
        for name, query_ids in groups.items():
            for query_id in query_ids:
                if not isinstance(query_id, str) or not query_id.strip():
                    raise DataValidationError(f"{name} contains an invalid query ID.")
                if query_id in seen:
                    raise DataValidationError(
                        f"Query {query_id!r} appears in more than one split."
                    )
                seen.add(query_id)
        object.__setattr__(self, "train", groups["train"])
        object.__setattr__(self, "validation", groups["validation"])
        object.__setattr__(self, "test", groups["test"])

    def split_for(self, query_id: str) -> str:
        """Return the split name containing a query, or raise an error."""
        if query_id in self.train:
            return "train"
        if query_id in self.validation:
            return "validation"
        if query_id in self.test:
            return "test"
        raise DataValidationError(f"Query {query_id!r} is not present in the split.")


def assign_query_splits(
    query_ids: Iterable[str],
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> QuerySplit:
    """Assign queries to disjoint splits deterministically.

    Splitting is at the query level. Shared documents remain in the retrieval
    corpus for all splits, which is standard for retrieval evaluation and avoids
    treating corpus documents as query-level leakage.
    """
    try:
        valid_seed = validate_seed(seed)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(str(exc)) from exc

    _validate_fractions(fractions)
    normalized_ids = tuple(query_ids)
    if not normalized_ids:
        raise DataValidationError("At least one query ID is required for splitting.")
    if any(not isinstance(query_id, str) or not query_id.strip() for query_id in normalized_ids):
        raise DataValidationError("All query IDs must be non-empty strings.")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise DataValidationError("Query IDs must be unique before splitting.")

    ordered_ids = sorted(normalized_ids)
    rng = random.Random(derive_seed("query-split", valid_seed))
    rng.shuffle(ordered_ids)

    count = len(ordered_ids)
    train_count = int(count * fractions[0])
    validation_count = int(count * fractions[1])
    if count >= 3 and train_count == 0:
        train_count = 1
    if count >= 3 and validation_count == 0:
        validation_count = 1

    train_end = train_count
    validation_end = min(count, train_end + validation_count)
    return QuerySplit(
        train=tuple(ordered_ids[:train_end]),
        validation=tuple(ordered_ids[train_end:validation_end]),
        test=tuple(ordered_ids[validation_end:]),
    )


def _validate_fractions(fractions: tuple[float, float, float]) -> None:
    if len(fractions) != 3:
        raise DataValidationError("fractions must contain train, validation, and test values.")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in fractions
    ):
        raise DataValidationError("Split fractions must be numeric.")
    if any(value < 0 for value in fractions):
        raise DataValidationError("Split fractions must be non-negative.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise DataValidationError("Split fractions must sum to 1.0.")


__all__ = ["QuerySplit", "assign_query_splits"]
