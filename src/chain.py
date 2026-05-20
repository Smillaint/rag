# -*- coding: utf-8 -*-
import shutil
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from src.generator import generate_answer, load_generator
from src.loader import load_pdfs, split_documents
from src.reranker import load_reranker, rerank
from src.retriever import BM25Index, build_vectorstore, hybrid_search, load_vectorstore


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
    ) -> "RAGPipeline":
        docs = load_pdfs(data_dir)
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        vectorstore_path = Path(vectorstore_dir)
        if rebuild and vectorstore_path.exists():
            shutil.rmtree(vectorstore_path)

        if rebuild or not vectorstore_path.exists():
            vectorstore = build_vectorstore(chunks, persist_dir=vectorstore_dir)
        else:
            vectorstore = load_vectorstore(vectorstore_dir)

        return cls(
            chunks=chunks,
            vectorstore=vectorstore,
            bm25_index=BM25Index(chunks),
            reranker=load_reranker(),
            client=load_generator(api_key=api_key, base_url=base_url, model=model)[0],
            model=model,
        )

    @staticmethod
    def is_code_query(query: str) -> bool:
        normalized = query.lower()
        return any(keyword.lower() in normalized for keyword in CODE_QUERY_KEYWORDS)

    @staticmethod
    def _chunk_key(doc: Document) -> tuple[str, int | str, int | str]:
        return (
            str(doc.metadata.get("source", "")),
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
            if isinstance(chunk_index, int):
                for offset in range(-before, after + 1):
                    neighbor = chunk_by_index.get(chunk_index + offset)
                    if neighbor is not None:
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

    def retrieve(self, query: str, retrieve_top_k: int = 5, rerank_top_k: int = 3) -> list[Document]:
        code_query = self.is_code_query(query)
        if code_query:
            retrieve_top_k = max(retrieve_top_k, 12)
            rerank_top_k = max(rerank_top_k, 6)
            print(
                "Code query detected; "
                f"using retrieve_top_k={retrieve_top_k}, rerank_top_k={rerank_top_k}"
            )

        candidates = hybrid_search(
            self.vectorstore,
            self.chunks,
            query,
            top_k=retrieve_top_k,
            bm25_index=self.bm25_index,
        )
        docs = rerank(self.reranker, query, candidates, top_k=rerank_top_k)
        if code_query:
            return self.expand_with_neighbor_chunks(docs)
        return docs

    def ask(self, query: str, retrieve_top_k: int = 5, rerank_top_k: int = 3) -> str:
        docs = self.retrieve(query, retrieve_top_k=retrieve_top_k, rerank_top_k=rerank_top_k)
        return generate_answer(self.client, self.model, query, docs)

    def ask_with_sources(
        self,
        query: str,
        retrieve_top_k: int = 5,
        rerank_top_k: int = 3,
    ) -> dict:
        docs = self.retrieve(query, retrieve_top_k=retrieve_top_k, rerank_top_k=rerank_top_k)
        answer = generate_answer(self.client, self.model, query, docs)
        return {
            "answer": answer,
            "sources": [
                {
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "chunk_in_page": doc.metadata.get("chunk_in_page"),
                    "char_start": doc.metadata.get("char_start"),
                    "char_end": doc.metadata.get("char_end"),
                    "preview": doc.page_content[:160],
                }
                for doc in docs
            ],
        }
