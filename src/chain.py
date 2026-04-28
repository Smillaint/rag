# -*- coding: utf-8 -*-
import shutil
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from src.generator import generate_answer, load_generator
from src.loader import load_pdfs, split_documents
from src.reranker import load_reranker, rerank
from src.retriever import BM25Index, build_vectorstore, hybrid_search, load_vectorstore


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

    def retrieve(self, query: str, retrieve_top_k: int = 5, rerank_top_k: int = 3) -> list[Document]:
        candidates = hybrid_search(
            self.vectorstore,
            self.chunks,
            query,
            top_k=retrieve_top_k,
            bm25_index=self.bm25_index,
        )
        return rerank(self.reranker, query, candidates, top_k=rerank_top_k)

    def ask(self, query: str, retrieve_top_k: int = 5, rerank_top_k: int = 3) -> str:
        docs = self.retrieve(query, retrieve_top_k=retrieve_top_k, rerank_top_k=rerank_top_k)
        return generate_answer(self.client, self.model, query, docs)
