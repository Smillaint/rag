# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import jieba
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_RETRIEVE_TOP_K = 5
DEFAULT_RERANK_TOP_K = 3
DEFAULT_VECTORSTORE_BATCH_SIZE = 128

RRF_K = 60
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 1.0

FUSION_CANDIDATE_MULTIPLIER = 2
CONTENT_FINGERPRINT_LENGTH = 80


class BM25Index:
    """Lazy BM25 index for keyword retrieval.

    The full keyword index can be large for book-sized corpora, so indexes are
    built on first use and cached per collection.
    """

    def __init__(self, chunks: list[Document]) -> None:
        self.chunks = chunks
        self._indexes: dict[str, tuple[list[Document], BM25Okapi | None]] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return list(jieba.cut(text))

    @staticmethod
    def _matches_collection(doc: Document, collection: str | None = None) -> bool:
        return collection is None or doc.metadata.get("collection", "default") == collection

    @staticmethod
    def _collection_key(collection: str | None) -> str:
        return collection if collection is not None else "__all__"

    def _get_index(self, collection: str | None = None) -> tuple[list[Document], BM25Okapi | None]:
        key = self._collection_key(collection)
        if key in self._indexes:
            return self._indexes[key]

        docs = [
            doc for doc in self.chunks
            if self._matches_collection(doc, collection)
        ]
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in docs]
        index = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        self._indexes[key] = (docs, index)
        scope = f"collection={collection}" if collection else "all collections"
        logger.info("Built BM25 index for %d chunks (%s)", len(docs), scope)
        return docs, index

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_RETRIEVE_TOP_K,
        collection: str | None = None,
    ) -> list[Document]:
        """One-shot BM25 retrieval, scores discarded."""
        return [doc for doc, _ in self.search_with_scores(query, top_k=top_k, collection=collection)]

    def search_with_scores(
        self,
        query: str,
        top_k: int = DEFAULT_RETRIEVE_TOP_K,
        collection: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Return (Document, bm25_score) pairs ranked by relevance."""
        docs, index = self._get_index(collection)
        if not index:
            return []

        tokenized_query = self._tokenize(query)
        scores = index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(docs[i], float(scores[i])) for i in top_indices]


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the multilingual embedding model (bge-m3, 1024-dim, normalized)."""
    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        logger.warning("Embedding model remote check failed, retrying local cache: %s", exc)
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )


def validate_batch_size(batch_size: int) -> int:
    """Validate that *batch_size* is a positive integer.

    Raises ValueError with a clear message otherwise. Returns the validated
    value so callers can chain ``batch_size = validate_batch_size(batch_size)``.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be greater than 0, got {batch_size}.")
    return batch_size


def _chunk_ids(chunks: list[Document]) -> list[str]:
    return [str(chunk.metadata.get("chunk_id")) for chunk in chunks]


def _batched(chunks: list[Document], batch_size: int) -> Iterator[list[Document]]:
    """Lazily yield successive *batch_size* slices from *chunks*.

    Unlike a list materialisation this generator only ever holds one batch in
    memory at a time, which matters for book-sized corpora. Callers must
    consume the iterator; passing it to ``len()`` will raise TypeError.
    """
    validate_batch_size(batch_size)
    for index in range(0, len(chunks), batch_size):
        yield chunks[index:index + batch_size]


def _add_documents_in_batches(vectorstore: Chroma, chunks: list[Document], batch_size: int) -> None:
    validate_batch_size(batch_size)
    total = len(chunks)
    total_batches = (total + batch_size - 1) // batch_size
    for batch_number, batch in enumerate(_batched(chunks, batch_size), start=1):
        vectorstore.add_documents(batch, ids=_chunk_ids(batch))
        logger.info("Added vector batch %d/%d (%d chunks)", batch_number, total_batches, len(batch))


def build_vectorstore(
    chunks: list[Document],
    persist_dir: str = "./vectorstore",
    batch_size: int = DEFAULT_VECTORSTORE_BATCH_SIZE,
) -> Chroma:
    """Build and persist a Chroma vector store."""
    if not chunks:
        raise ValueError("No document chunks were provided. Put PDFs in ./data first.")

    embedding = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
    )
    _add_documents_in_batches(vectorstore, chunks, batch_size=batch_size)
    logger.info("Built vector store with %d chunks at %s", len(chunks), persist_dir)
    return vectorstore


def load_vectorstore(persist_dir: str = "./vectorstore") -> Chroma:
    """Load an existing Chroma vector store, failing clearly when missing."""
    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}. Run with --rebuild first."
        )

    embedding = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
    )
    logger.info("Loaded vector store from %s", persist_dir)
    return vectorstore


def _embedding_meta_path(persist_dir: str) -> Path:
    return Path(persist_dir) / ".embedding_meta.json"


def _embedding_model_changed(meta_path: Path) -> bool:
    """True when the persisted vector store was built with a different embedding model."""
    if not meta_path.exists():
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read embedding meta %s: %s", meta_path, exc)
        return True
    return (
        meta.get("model") != EMBEDDING_MODEL_NAME
        or meta.get("dimension") != EMBEDDING_DIMENSION
    )


def _write_embedding_meta(meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {"model": EMBEDDING_MODEL_NAME, "dimension": EMBEDDING_DIMENSION},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def sync_vectorstore(
    chunks: list[Document],
    persist_dir: str,
    changed_sources: list[str],
    deleted_chunk_ids: list[str],
    full_rebuild: bool = False,
    batch_size: int = DEFAULT_VECTORSTORE_BATCH_SIZE,
) -> Chroma:
    """Load or update Chroma using stable chunk IDs.

    Auto-detects embedding-model changes via a sidecar metadata file and forces
    a full rebuild to avoid dimension-mismatch corruption.
    """
    vectorstore_path = Path(persist_dir)
    meta_path = _embedding_meta_path(persist_dir)

    if vectorstore_path.exists() and _embedding_model_changed(meta_path):
        logger.warning(
            "Embedding model changed since last build (now %s/%dd); "
            "forcing full vector store rebuild.",
            EMBEDDING_MODEL_NAME,
            EMBEDDING_DIMENSION,
        )
        full_rebuild = True

    if full_rebuild:
        if vectorstore_path.exists():
            shutil.rmtree(vectorstore_path)
        vectorstore = build_vectorstore(chunks, persist_dir=persist_dir, batch_size=batch_size)
        _write_embedding_meta(meta_path)
        return vectorstore
    if not vectorstore_path.exists():
        vectorstore = build_vectorstore(chunks, persist_dir=persist_dir, batch_size=batch_size)
        _write_embedding_meta(meta_path)
        return vectorstore

    vectorstore = load_vectorstore(persist_dir)
    if deleted_chunk_ids:
        vectorstore.delete(ids=deleted_chunk_ids)
        logger.info("Deleted %d stale chunks from vector store", len(deleted_chunk_ids))

    changed = set(changed_sources)
    new_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("source_path", chunk.metadata.get("source")) in changed
    ]
    if new_chunks:
        _add_documents_in_batches(vectorstore, new_chunks, batch_size=batch_size)
        logger.info("Added %d changed chunks to vector store", len(new_chunks))

    _write_embedding_meta(meta_path)
    return vectorstore


def bm25_search(
    chunks: list[Document],
    query: str,
    top_k: int = DEFAULT_RETRIEVE_TOP_K,
    collection: str | None = None,
) -> list[Document]:
    """One-off BM25 keyword retrieval. Prefer BM25Index for repeated queries."""
    return BM25Index(chunks).search(query, top_k=top_k, collection=collection)


def _doc_key(doc: Document) -> tuple:
    """Stable dedup key for fusion: prefer chunk_id, fall back to content fingerprint."""
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id:
        return ("chunk_id", str(chunk_id))
    return (
        str(doc.metadata.get("source", "")),
        doc.metadata.get("page", ""),
        doc.page_content[:CONTENT_FINGERPRINT_LENGTH],
    )


def _rrf_score(rank: int, weight: float, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion contribution: weight / (k + rank)."""
    return weight / (k + rank)


def hybrid_search_with_scores(
    vectorstore: Chroma,
    chunks: list[Document],
    query: str,
    top_k: int = DEFAULT_RETRIEVE_TOP_K,
    bm25_index: BM25Index | None = None,
    collection: str | None = None,
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    rrf_k: int = RRF_K,
    vector_query: str | None = None,
) -> list[tuple[Document, float]]:
    """Weighted Reciprocal Rank Fusion of vector and BM25 retrieval.

    RRF is rank-based, so it does not require score normalization between the
    two retrievers (vector distance vs. BM25 score). Each retriever contributes
    weight / (k + rank) for every hit; overlapping documents accumulate score,
    which naturally boosts items both retrievers agree on.

    When ``vector_query`` is provided (e.g. a HyDE hypothetical document) the
    vector side searches with it while BM25 keeps the raw user query, since
    BM25 relies on exact term overlap which a generated passage would dilute.
    """
    actual_vector_query = vector_query or query
    search_kwargs: dict[str, object] = {"k": top_k}
    if collection:
        search_kwargs["filter"] = {"collection": collection}

    try:
        raw_vector_hits = vectorstore.similarity_search_with_score(
            actual_vector_query, **search_kwargs
        )
    except Exception as exc:
        logger.warning("Vector search failed, falling back to BM25 only: %s", exc)
        raw_vector_hits = []
    vector_ranked = [doc for doc, _ in sorted(raw_vector_hits, key=lambda item: item[1])]

    bm25_ranked = (bm25_index or BM25Index(chunks)).search_with_scores(
        query, top_k=top_k, collection=collection
    )
    bm25_ranked_docs = [doc for doc, _ in bm25_ranked]

    fusion: dict[tuple, float] = {}
    doc_by_key: dict[tuple, Document] = {}

    for rank, doc in enumerate(vector_ranked, start=1):
        key = _doc_key(doc)
        doc_by_key.setdefault(key, doc)
        fusion[key] = fusion.get(key, 0.0) + _rrf_score(rank, vector_weight, rrf_k)

    for rank, doc in enumerate(bm25_ranked_docs, start=1):
        key = _doc_key(doc)
        doc_by_key.setdefault(key, doc)
        fusion[key] = fusion.get(key, 0.0) + _rrf_score(rank, bm25_weight, rrf_k)

    ordered = sorted(fusion.items(), key=lambda item: item[1], reverse=True)
    scope = f"collection={collection}" if collection else "all collections"
    logger.info(
        "Hybrid RRF search fused %d vector + %d BM25 hits into %d candidates (%s)",
        len(vector_ranked),
        len(bm25_ranked_docs),
        len(ordered),
        scope,
    )
    return [(doc_by_key[key], score) for key, score in ordered[: top_k * FUSION_CANDIDATE_MULTIPLIER]]


def hybrid_search(
    vectorstore: Chroma,
    chunks: list[Document],
    query: str,
    top_k: int = DEFAULT_RETRIEVE_TOP_K,
    bm25_index: BM25Index | None = None,
    collection: str | None = None,
) -> list[Document]:
    """Hybrid retrieval via weighted RRF fusion (vector + BM25), scores discarded."""
    return [
        doc
        for doc, _ in hybrid_search_with_scores(
            vectorstore,
            chunks,
            query,
            top_k=top_k,
            bm25_index=bm25_index,
            collection=collection,
        )
    ]
