from __future__ import annotations

import pytest

import evidenceguard.retrieval.embeddings as embeddings_module
from evidenceguard.errors import OptionalDependencyError
from evidenceguard.optional import import_optional_dependency
from evidenceguard.retrieval.embeddings import SentenceTransformerEncoder


def test_optional_dependency_imports_available_module() -> None:
    module = import_optional_dependency(
        "json",
        package_name="unused",
        install_hint="not-needed",
    )

    assert module.__name__ == "json"


def test_optional_dependency_reports_clear_installation_error() -> None:
    with pytest.raises(OptionalDependencyError, match="fake-package") as error:
        import_optional_dependency(
            "evidenceguard_missing_optional_dependency_12345",
            package_name="fake-package",
            install_hint='python -m pip install "fake-package"',
        )

    assert "fake-package" in str(error.value)
    assert "pip install" in str(error.value)


def test_optional_dependency_rejects_empty_arguments() -> None:
    with pytest.raises(ValueError, match="module_name"):
        import_optional_dependency("", package_name="pkg", install_hint="hint")
    with pytest.raises(ValueError, match="package_name"):
        import_optional_dependency("json", package_name="", install_hint="hint")
    with pytest.raises(ValueError, match="install_hint"):
        import_optional_dependency("json", package_name="pkg", install_hint="")


def test_sentence_transformer_encoder_validates_before_loading() -> None:
    with pytest.raises(ValueError, match="model_name"):
        SentenceTransformerEncoder("")


def test_sentence_transformer_encoder_propagates_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_import(
        module_name: str,
        *,
        package_name: str,
        install_hint: str,
    ) -> object:
        raise OptionalDependencyError(f"Optional dependency '{package_name}' is missing.")

    monkeypatch.setattr(embeddings_module, "import_optional_dependency", fail_import)

    with pytest.raises(OptionalDependencyError, match="sentence-transformers"):
        SentenceTransformerEncoder("test-model")
