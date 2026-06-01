# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path

import jieba
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class BM25Index:
    """Lazy BM25 index for keyword retrieval.

    The full keyword index can be large for book-sized corpora, so indexes are
    built on first use and cached per collection.
    """

    def __init__(self, chunks: list[Document]):
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
        print(f"Built BM25 index for {len(docs)} chunks ({scope})")
        return docs, index

    def search(
        self,
        query: str,
        top_k: int = 5,
        collection: str | None = None,
    ) -> list[Document]:
        docs, index = self._get_index(collection)
        if not index:
            return []

        tokenized_query = self._tokenize(query)
        scores = index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [docs[i] for i in top_indices]


def get_embedding_model():
    """Load the multilingual embedding model."""
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
        )
    except Exception as exc:
        print(f"Embedding model remote check failed, retrying local cache: {exc}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu", "local_files_only": True},
        )


def _chunk_ids(chunks: list[Document]) -> list[str]:
    return [str(chunk.metadata.get("chunk_id")) for chunk in chunks]


def _batched(chunks: list[Document], batch_size: int) -> list[list[Document]]:
    return [chunks[index:index + batch_size] for index in range(0, len(chunks), batch_size)]


def _add_documents_in_batches(vectorstore, chunks: list[Document], batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    total = len(chunks)
    for batch_number, batch in enumerate(_batched(chunks, batch_size), start=1):
        vectorstore.add_documents(batch, ids=_chunk_ids(batch))
        print(
            "Added vector batch "
            f"{batch_number}/{(total + batch_size - 1) // batch_size} "
            f"({len(batch)} chunks)"
        )


def build_vectorstore(
    chunks: list[Document],
    persist_dir: str = "./vectorstore",
    batch_size: int = 256,
):
    """Build and persist a Chroma vector store."""
    if not chunks:
        raise ValueError("No document chunks were provided. Put PDFs in ./data first.")

    embedding = get_embedding_model()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
    )
    _add_documents_in_batches(vectorstore, chunks, batch_size=batch_size)
    print(f"Built vector store with {len(chunks)} chunks at {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir: str = "./vectorstore"):
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
    print(f"Loaded vector store from {persist_dir}")
    return vectorstore


def sync_vectorstore(
    chunks: list[Document],
    persist_dir: str,
    changed_sources: list[str],
    deleted_chunk_ids: list[str],
    full_rebuild: bool = False,
    batch_size: int = 256,
):
    """Load or update Chroma using stable chunk IDs."""
    vectorstore_path = Path(persist_dir)
    if full_rebuild:
        if vectorstore_path.exists():
            shutil.rmtree(vectorstore_path)
        return build_vectorstore(chunks, persist_dir=persist_dir, batch_size=batch_size)
    if not vectorstore_path.exists():
        return build_vectorstore(chunks, persist_dir=persist_dir, batch_size=batch_size)

    vectorstore = load_vectorstore(persist_dir)
    if deleted_chunk_ids:
        vectorstore.delete(ids=deleted_chunk_ids)
        print(f"Deleted {len(deleted_chunk_ids)} stale chunks from vector store")

    changed = set(changed_sources)
    new_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata.get("source_path", chunk.metadata.get("source")) in changed
    ]
    if new_chunks:
        _add_documents_in_batches(vectorstore, new_chunks, batch_size=batch_size)
        print(f"Added {len(new_chunks)} changed chunks to vector store")

    return vectorstore


def bm25_search(
    chunks: list[Document],
    query: str,
    top_k: int = 5,
    collection: str | None = None,
) -> list[Document]:
    """One-off BM25 keyword retrieval. Prefer BM25Index for repeated queries."""
    return BM25Index(chunks).search(query, top_k=top_k, collection=collection)


def hybrid_search(
    vectorstore,
    chunks: list[Document],
    query: str,
    top_k: int = 5,
    bm25_index: BM25Index | None = None,
    collection: str | None = None,
) -> list[Document]:
    """Hybrid retrieval: vector search + BM25, merged and de-duplicated."""
    search_kwargs = {"k": top_k}
    if collection:
        search_kwargs["filter"] = {"collection": collection}
    vector_results = vectorstore.similarity_search(query, **search_kwargs)
    bm25_results = (bm25_index or BM25Index(chunks)).search(
        query,
        top_k=top_k,
        collection=collection,
    )

    seen: set[tuple[str, int | str, int | str]] = set()
    combined: list[Document] = []
    for doc in vector_results + bm25_results:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", ""),
            doc.metadata.get("chunk_id", doc.page_content[:80]),
        )
        if key in seen:
            continue
        seen.add(key)
        combined.append(doc)

    scope = f"collection={collection}" if collection else "all collections"
    print(f"Hybrid search returned {len(combined)} candidate chunks ({scope})")
    return combined[: top_k * 2]
