from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from evidenceguard.data import Document, Query, RetrievalDataset, load_beir_dataset
from evidenceguard.errors import EvaluationError
from evidenceguard.evaluation import evaluate_retriever
from evidenceguard.retrieval.base import RetrievalResult
from evidenceguard.retrieval.bm25 import BM25Retriever


class RankingRetriever:
    def __init__(
        self,
        dataset: RetrievalDataset,
        rankings: dict[str, list[str]],
        *,
        name: str = "ranking",
    ) -> None:
        self._documents = {document.doc_id: document for document in dataset.documents}
        self._rankings = rankings
        self._name = name
        self.last_top_k: int | None = None

    @property
    def name(self) -> str:
        return self._name

    def retrieve(
        self,
        query: str | Query,
        *,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        query_id = query.query_id if isinstance(query, Query) else query
        self.last_top_k = top_k
        return [
            RetrievalResult(document=self._documents[doc_id], score=1.0, rank=rank)
            for rank, doc_id in enumerate(
                self._rankings.get(query_id, [])[:top_k],
                start=1,
            )
        ]


class FixedResultsRetriever:
    def __init__(self, results: Sequence[object], *, name: str = "fixed") -> None:
        self._results = tuple(results)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def retrieve(
        self,
        query: str | Query,
        *,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        return list(self._results[:top_k])  # type: ignore[return-value]


def test_evaluate_bm25_on_synthetic_fixture(
    mini_beir_dir: Path,
) -> None:
    dataset = load_beir_dataset(mini_beir_dir)
    evaluation = evaluate_retriever(
        BM25Retriever(dataset.documents),
        dataset,
        cutoffs=(1, 3),
    )

    assert evaluation.dataset_name == "mini_beir"
    assert evaluation.split == "test"
    assert evaluation.retriever_name == "bm25"
    assert evaluation.top_k == 3
    assert evaluation.cutoffs == (1, 3)
    assert evaluation.skipped_query_ids == ("q-unanswerable",)
    assert len(evaluation.per_query) == 4
    assert [metrics.query_count for metrics in evaluation.aggregates] == [2, 2]
    assert [metrics.mean_precision for metrics in evaluation.aggregates] == [
        pytest.approx(1.0),
        pytest.approx(1.0 / 3.0),
    ]
    for aggregate in evaluation.aggregates:
        assert aggregate.mean_recall == pytest.approx(1.0)
        assert aggregate.mean_reciprocal_rank == pytest.approx(1.0)
        assert aggregate.mean_average_precision == pytest.approx(1.0)
        assert aggregate.mean_ndcg == pytest.approx(1.0)
        assert aggregate.hit_rate == pytest.approx(1.0)


def test_evaluate_retriever_macro_averages_and_skips(
    mini_beir_dir: Path,
) -> None:
    dataset = load_beir_dataset(mini_beir_dir)
    retriever = RankingRetriever(
        dataset,
        {
            "q-answerable": ["doc-b", "doc-a"],
            "q-properties": ["doc-e"],
            "q-unanswerable": ["doc-a"],
        },
    )

    evaluation = evaluate_retriever(retriever, dataset, cutoffs=(1, 2))

    assert retriever.last_top_k == 2
    first, second = evaluation.aggregates
    assert first.cutoff == 1
    assert first.mean_recall == pytest.approx(0.5)
    assert first.mean_precision == pytest.approx(0.5)
    assert first.mean_reciprocal_rank == pytest.approx(0.5)
    assert first.mean_average_precision == pytest.approx(0.5)
    assert first.mean_ndcg == pytest.approx(0.5)
    assert first.hit_rate == pytest.approx(0.5)
    assert second.cutoff == 2
    assert second.mean_recall == pytest.approx(1.0)
    assert second.mean_precision == pytest.approx(0.5)
    assert second.mean_reciprocal_rank == pytest.approx(0.75)
    assert second.mean_average_precision == pytest.approx(0.75)
    assert second.mean_ndcg == pytest.approx(
        (1.0 / math.log2(3.0) + 1.0) / 2.0
    )
    assert second.hit_rate == pytest.approx(1.0)


def test_evaluate_retriever_uses_explicit_larger_depth(
    mini_beir_dir: Path,
) -> None:
    dataset = load_beir_dataset(mini_beir_dir)
    retriever = RankingRetriever(dataset, {})

    evaluation = evaluate_retriever(
        retriever,
        dataset,
        cutoffs=(1, 2),
        top_k=5,
    )

    assert evaluation.top_k == 5
    assert retriever.last_top_k == 5


def test_evaluate_retriever_serializes_results(mini_beir_dir: Path) -> None:
    dataset = load_beir_dataset(mini_beir_dir)
    evaluation = evaluate_retriever(
        BM25Retriever(dataset.documents),
        dataset,
        cutoffs=(1,),
    )

    payload = evaluation.to_dict()

    assert payload["dataset_name"] == "mini_beir"
    assert payload["cutoffs"] == [1]
    assert len(payload["aggregates"]) == 1
    assert len(payload["per_query"]) == 2
    json.dumps(payload)


def test_evaluate_retriever_rejects_invalid_cutoffs_and_depth(
    mini_beir_dir: Path,
) -> None:
    dataset = load_beir_dataset(mini_beir_dir)
    retriever = BM25Retriever(dataset.documents)

    for cutoffs in ([], [0], [1, 1], [True]):
        with pytest.raises(ValueError, match="cutoffs"):
            evaluate_retriever(retriever, dataset, cutoffs=cutoffs)  # type: ignore[arg-type]
    for top_k in (0, 4, True):
        with pytest.raises(ValueError, match="top_k"):
            evaluate_retriever(
                retriever,
                dataset,
                cutoffs=(1, 5),
                top_k=top_k,  # type: ignore[arg-type]
            )


def test_evaluate_retriever_requires_an_evaluable_query(
    mini_beir_dir: Path,
) -> None:
    dataset = load_beir_dataset(mini_beir_dir)
    no_qrels = RetrievalDataset(
        name="no-qrels",
        split="test",
        documents=dataset.documents,
        queries=dataset.queries,
        qrels=(),
    )

    with pytest.raises(EvaluationError, match="No queries"):
        evaluate_retriever(BM25Retriever(dataset.documents), no_qrels)


def test_evaluate_retriever_rejects_unknown_or_duplicate_documents(
    mini_beir_dir: Path,
) -> None:
    dataset = load_beir_dataset(mini_beir_dir)
    document = dataset.get_document("doc-a")
    unknown = Document(doc_id="unknown", title="", text="Unknown.")

    unknown_retriever = FixedResultsRetriever(
        [RetrievalResult(document=unknown, score=1.0, rank=1)]
    )
    with pytest.raises(EvaluationError, match="outside the dataset"):
        evaluate_retriever(unknown_retriever, dataset, cutoffs=(1,))

    duplicate_retriever = FixedResultsRetriever(
        [
            RetrievalResult(document=document, score=1.0, rank=1),
            RetrievalResult(document=document, score=0.5, rank=2),
        ]
    )
    with pytest.raises(EvaluationError, match="duplicate"):
        evaluate_retriever(duplicate_retriever, dataset, cutoffs=(2,))


def test_evaluate_retriever_rejects_invalid_result_contracts(
    mini_beir_dir: Path,
) -> None:
    dataset = load_beir_dataset(mini_beir_dir)
    document = dataset.get_document("doc-a")
    non_result = FixedResultsRetriever(["doc-a"])
    with pytest.raises(EvaluationError, match="RetrievalResult"):
        evaluate_retriever(non_result, dataset, cutoffs=(1,))

    wrong_rank = FixedResultsRetriever(
        [RetrievalResult(document=document, score=1.0, rank=2)]
    )
    with pytest.raises(EvaluationError, match="ranks"):
        evaluate_retriever(wrong_rank, dataset, cutoffs=(1,))


def test_evaluate_retriever_validates_inputs(mini_beir_dir: Path) -> None:
    dataset = load_beir_dataset(mini_beir_dir)

    with pytest.raises(EvaluationError, match="RetrievalDataset"):
        evaluate_retriever(BM25Retriever(dataset.documents), object())  # type: ignore[arg-type]
    with pytest.raises(EvaluationError, match="name"):
        evaluate_retriever(object(), dataset)  # type: ignore[arg-type]
