# -*- coding: utf-8 -*-
import time
from dataclasses import dataclass

from langchain_core.documents import Document

from src.generator import generate_answer_result, load_generator
from src.index_cache import load_or_update_chunks
from src.query_log import write_query_log
from src.reranker import load_reranker, rerank_with_scores
from src.retriever import BM25Index, hybrid_search, sync_vectorstore


CODE_QUERY_KEYWORDS = (
    "代码",
    "程序",
    "函数",
    "main.c",
    "demo",
    "编程",
    "解释",
    "code",
    "function",
    "program",
)


@dataclass
class RAGPipeline:
    chunks: list[Document]
    vectorstore: object
    bm25_index: BM25Index
    reranker: object
    client: object
    model: str
    log_dir: str

    @classmethod
    def from_paths(
        cls,
        api_key: str,
        data_dir: str = "./data",
        vectorstore_dir: str = "./vectorstore",
        rebuild: bool = False,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        cache_dir: str = "./.rag_cache",
        log_dir: str = "./logs",
        vectorstore_batch_size: int = 256,
    ) -> "RAGPipeline":
        index_result = load_or_update_chunks(
            data_dir=data_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cache_dir=cache_dir,
            force_rebuild=rebuild,
        )
        vectorstore = sync_vectorstore(
            chunks=index_result.chunks,
            persist_dir=vectorstore_dir,
            changed_sources=index_result.changed_sources,
            deleted_chunk_ids=index_result.deleted_chunk_ids,
            full_rebuild=rebuild or index_result.full_rebuild,
            batch_size=vectorstore_batch_size,
        )

        return cls(
            chunks=index_result.chunks,
            vectorstore=vectorstore,
            bm25_index=BM25Index(index_result.chunks),
            reranker=load_reranker(),
            client=load_generator(api_key=api_key, base_url=base_url, model=model)[0],
            model=model,
            log_dir=log_dir,
        )

    @staticmethod
    def is_code_query(query: str) -> bool:
        normalized = query.lower()
        return any(keyword.lower() in normalized for keyword in CODE_QUERY_KEYWORDS)

    @staticmethod
    def _chunk_key(doc: Document) -> tuple[str, int | str, int | str]:
        return (
            str(doc.metadata.get("source_path", doc.metadata.get("source", ""))),
            doc.metadata.get("page", ""),
            doc.metadata.get("chunk_in_page", doc.metadata.get("chunk_id", "")),
        )

    def expand_with_neighbor_chunks(
        self,
        docs: list[Document],
        before: int = 2,
        after: int = 3,
    ) -> list[Document]:
        """Add adjacent chunks so split code blocks stay complete across pages."""
        chunk_by_index = {
            chunk.metadata.get("chunk_index"): chunk
            for chunk in self.chunks
            if isinstance(chunk.metadata.get("chunk_index"), int)
        }
        expanded: list[Document] = []
        seen: set[tuple[str, int | str, int | str]] = set()

        for doc in docs:
            neighbor_docs: list[Document] = []
            chunk_index = doc.metadata.get("chunk_index")
            source_key = doc.metadata.get("source_path", doc.metadata.get("source"))
            collection = doc.metadata.get("collection", "default")
            if isinstance(chunk_index, int):
                for offset in range(-before, after + 1):
                    neighbor = chunk_by_index.get(chunk_index + offset)
                    if (
                        neighbor is not None
                        and neighbor.metadata.get("source_path", neighbor.metadata.get("source")) == source_key
                        and neighbor.metadata.get("collection", "default") == collection
                    ):
                        neighbor_docs.append(neighbor)
            else:
                neighbor_docs.append(doc)

            for neighbor in neighbor_docs:
                key = self._chunk_key(neighbor)
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(neighbor)

        print(f"Expanded code context to {len(expanded)} chunks")
        return expanded

    def retrieve(
        self,
        query: str,
        retrieve_top_k: int = 5,
        rerank_top_k: int = 3,
        collection: str | None = None,
    ) -> list[Document]:
        docs, _ = self.retrieve_with_trace(query, retrieve_top_k, rerank_top_k, collection)
        return docs

    def retrieve_with_trace(
        self,
        query: str,
        retrieve_top_k: int = 5,
        rerank_top_k: int = 3,
        collection: str | None = None,
    ) -> tuple[list[Document], dict]:
        started_at = time.perf_counter()
        code_query = self.is_code_query(query)
        requested_retrieve_top_k = retrieve_top_k
        requested_rerank_top_k = rerank_top_k
        if code_query:
            retrieve_top_k = max(retrieve_top_k, 12)
            rerank_top_k = max(rerank_top_k, 6)
            print(
                "Code query detected; "
                f"using retrieve_top_k={retrieve_top_k}, rerank_top_k={rerank_top_k}"
            )

        retrieval_started_at = time.perf_counter()
        candidates = hybrid_search(
            self.vectorstore,
            self.chunks,
            query,
            top_k=retrieve_top_k,
            bm25_index=self.bm25_index,
            collection=collection,
        )
        retrieval_ms = (time.perf_counter() - retrieval_started_at) * 1000

        rerank_started_at = time.perf_counter()
        ranked_docs = rerank_with_scores(self.reranker, query, candidates, top_k=rerank_top_k)
        rerank_ms = (time.perf_counter() - rerank_started_at) * 1000
        docs = [doc for doc, _ in ranked_docs]

        expansion_ms = 0.0
        expanded_docs = docs
        if code_query:
            expansion_started_at = time.perf_counter()
            expanded_docs = self.expand_with_neighbor_chunks(docs)
            expansion_ms = (time.perf_counter() - expansion_started_at) * 1000

        trace = {
            "collection": collection,
            "code_query": code_query,
            "requested_retrieve_top_k": requested_retrieve_top_k,
            "requested_rerank_top_k": requested_rerank_top_k,
            "effective_retrieve_top_k": retrieve_top_k,
            "effective_rerank_top_k": rerank_top_k,
            "candidate_count": len(candidates),
            "reranked_count": len(docs),
            "final_context_count": len(expanded_docs),
            "timing_ms": {
                "retrieval": round(retrieval_ms, 2),
                "rerank": round(rerank_ms, 2),
                "context_expansion": round(expansion_ms, 2),
                "retrieve_total": round((time.perf_counter() - started_at) * 1000, 2),
            },
            "rerank_scores": [
                {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "source": doc.metadata.get("source"),
                    "source_path": doc.metadata.get("source_path"),
                    "collection": doc.metadata.get("collection"),
                    "page": doc.metadata.get("page"),
                    "score": round(score, 4),
                }
                for doc, score in ranked_docs
            ],
        }
        return expanded_docs, trace

    def ask(
        self,
        query: str,
        retrieve_top_k: int = 5,
        rerank_top_k: int = 3,
        collection: str | None = None,
    ) -> str:
        return self.ask_with_sources(
            query,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
            collection=collection,
        )["answer"]

    def ask_with_sources(
        self,
        query: str,
        retrieve_top_k: int = 5,
        rerank_top_k: int = 3,
        collection: str | None = None,
    ) -> dict:
        total_started_at = time.perf_counter()
        docs, trace = self.retrieve_with_trace(
            query,
            retrieve_top_k=retrieve_top_k,
            rerank_top_k=rerank_top_k,
            collection=collection,
        )
        generation_started_at = time.perf_counter()
        generation = generate_answer_result(self.client, self.model, query, docs)
        generation_ms = (time.perf_counter() - generation_started_at) * 1000
        trace["timing_ms"]["generation"] = round(generation_ms, 2)
        trace["timing_ms"]["total"] = round((time.perf_counter() - total_started_at) * 1000, 2)
        trace["generation"] = {
            "mode": generation.mode,
            "error": generation.error,
        }
        sources = [
            {
                "source": doc.metadata.get("source"),
                "source_path": doc.metadata.get("source_path"),
                "collection": doc.metadata.get("collection", "default"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "chunk_in_page": doc.metadata.get("chunk_in_page"),
                "char_start": doc.metadata.get("char_start"),
                "char_end": doc.metadata.get("char_end"),
                "preview": doc.page_content[:160],
            }
            for doc in docs
        ]
        result = {
            "answer": generation.answer,
            "trace": trace,
            "citations": [
                {
                    "label": f"[Chunk {index}]",
                    "source": source.get("source"),
                    "source_path": source.get("source_path"),
                    "collection": source.get("collection"),
                    "page": source.get("page"),
                    "chunk_id": source.get("chunk_id"),
                }
                for index, source in enumerate(sources, start=1)
            ],
            "sources": sources,
        }
        if self.log_dir:
            log_path = write_query_log(
                self.log_dir,
                query=query,
                collection=collection,
                retrieve_top_k=retrieve_top_k,
                rerank_top_k=rerank_top_k,
                result=result,
            )
            result["trace"]["log_path"] = str(log_path)
        return result
