"""Deterministic text preparation for lexical and dense retrieval."""

from __future__ import annotations

import re

from evidenceguard.data.models import Document
from evidenceguard.errors import DataValidationError

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase text and split it into ASCII alphanumeric tokens."""

    if not isinstance(text, str):
        raise DataValidationError("Text to tokenize must be a string.")
    return _TOKEN_PATTERN.findall(text.casefold())


def document_text(document: Document) -> str:
    """Combine title and body text for retrieval indexing."""

    if not isinstance(document, Document):
        raise DataValidationError("document_text requires a Document object.")
    if not document.title:
        return document.text
    return f"{document.title}\n{document.text}"
