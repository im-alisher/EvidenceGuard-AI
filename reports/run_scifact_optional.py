"""Run optional SciFact baselines through the existing evaluation/reporting APIs."""
from pathlib import Path

from evidenceguard.data import load_scifact
from evidenceguard.evaluation.retrieval import evaluate_retriever
from evidenceguard.experiments.result import ExperimentResult, write_experiment_result
from evidenceguard.reporting import write_markdown_report, write_retrieval_csv
from evidenceguard.retrieval.bm25 import BM25Retriever
from evidenceguard.retrieval.dense import DenseRetriever
from evidenceguard.retrieval.embeddings import SentenceTransformerEncoder
from evidenceguard.retrieval.hybrid import ReciprocalRankFusionRetriever
from evidenceguard.reproducibility import seed_everything

seed_everything(13)
dataset = load_scifact('datasets/raw/scifact')
encoder = SentenceTransformerEncoder('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
dense = DenseRetriever(dataset.documents, encoder)
bm25 = BM25Retriever(dataset.documents)
hybrid = ReciprocalRankFusionRetriever([bm25, dense], rrf_k=60, candidate_k=10)
for retriever in (dense, hybrid):
    name = 'scifact-' + retriever.name.replace('_', '-')
    evaluation = evaluate_retriever(retriever, dataset, cutoffs=(10,), top_k=10)
    result = ExperimentResult(name=name, seed=13, dataset_name=dataset.name,
                              split=dataset.split, retriever_name=retriever.name,
                              retrieval=evaluation)
    path = write_experiment_result(result, Path('reports/runs') / (name + '.json'))
    write_markdown_report(path, Path('reports') / (name + '.md'))
    write_retrieval_csv(path, Path('reports') / (name + '.csv'))
    print(evaluation.to_dict()['aggregates'], flush=True)
