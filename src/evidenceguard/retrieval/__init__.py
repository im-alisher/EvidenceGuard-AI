"""Independent lexical, dense, and hybrid retrieval components."""

from evidenceguard.retrieval.base import RetrievalResult, Retriever
from evidenceguard.retrieval.bm25 import BM25Retriever
from evidenceguard.retrieval.dense import DenseRetriever
from evidenceguard.retrieval.embeddings import EmbeddingEncoder, SentenceTransformerEncoder
from evidenceguard.retrieval.hybrid import ReciprocalRankFusionRetriever
from evidenceguard.retrieval.text import document_text, tokenize

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "EmbeddingEncoder",
    "ReciprocalRankFusionRetriever",
    "RetrievalResult",
    "Retriever",
    "SentenceTransformerEncoder",
    "document_text",
    "tokenize",
]
