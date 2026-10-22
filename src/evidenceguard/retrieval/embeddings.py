"""Embedding interfaces and a lazily imported dense encoder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from evidenceguard.optional import import_optional_dependency

FloatArray: TypeAlias = NDArray[np.float64]


class EmbeddingEncoder(Protocol):
    """Convert a batch of texts into a two-dimensional embedding matrix."""

    def encode(self, texts: Sequence[str]) -> FloatArray:
        """Return one row per input text."""

        ...


class SentenceTransformerEncoder:
    """Optional sentence-transformers encoder loaded only when instantiated."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("device must be a non-empty string or None.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        module = import_optional_dependency(
            "sentence_transformers",
            package_name="sentence-transformers",
            install_hint='python -m pip install -e ".[dense]"',
        )
        model_class = getattr(module, "SentenceTransformer")
        self._model = model_class(model_name, device=device)
        self._batch_size = batch_size
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""

        return self._model_name

    def encode(self, texts: Sequence[str]) -> FloatArray:
        """Encode texts with the already-loaded sentence-transformers model."""

        batch = tuple(texts)
        if not batch:
            raise ValueError("At least one text is required for encoding.")
        if any(not isinstance(text, str) for text in batch):
            raise ValueError("All texts to encode must be strings.")
        encoded = self._model.encode(
            list(batch),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(encoded, dtype=np.float64)
