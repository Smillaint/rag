# -*- coding: utf-8 -*-
"""MCP server exposing TraceRAG retrieval tools over StreamableHTTP.

This wraps the already-loaded ``RAGPipeline`` singleton into standard MCP
tools so that any MCP client (AgentFlow, Claude Desktop, etc.) can perform
vector + BM25 hybrid retrieval (RRF), HyDE query rewriting, and
CrossEncoder reranking against the local PDF knowledge base without
re-loading the heavy embedding and reranker models.

The server is built by :func:`build_mcp_server`, which takes two callables
that resolve the live pipeline and data directory. Tools run as synchronous
functions and are dispatched by the MCP runtime in a worker thread, so the
CPU-bound reranker never blocks the event loop.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer

from src.chain import RAGPipeline
from src.corpus_stats import build_corpus_stats
from src.loader import list_pdf_collections

logger = logging.getLogger(__name__)

PipelineGetter = Callable[[], "RAGPipeline | None"]
DataDirGetter = Callable[[], str]

SOURCE_PREVIEW_LENGTH = 160


def _source_from_doc(doc: Any, include_content: bool = True) -> dict[str, Any]:
    """Project a retrieved Document into a JSON-safe source payload."""
    metadata = doc.metadata or {}
    payload: dict[str, Any] = {
        "source": metadata.get("source"),
        "source_path": metadata.get("source_path"),
        "collection": metadata.get("collection", "default"),
        "page": metadata.get("page"),
        "chunk_id": metadata.get("chunk_id"),
        "chunk_index": metadata.get("chunk_index"),
        "chunk_in_page": metadata.get("chunk_in_page"),
        "char_start": metadata.get("char_start"),
        "char_end": metadata.get("char_end"),
        "preview": doc.page_content[:SOURCE_PREVIEW_LENGTH],
    }
    if include_content:
        payload["content"] = doc.page_content
    return payload


def build_mcp_server(
    get_pipeline: PipelineGetter,
    get_data_dir: DataDirGetter,
) -> MCPServer:
    """Build an MCP server backed by the shared RAGPipeline singleton.

    Args:
        get_pipeline: Returns the live pipeline, or None if not ready.
        get_data_dir: Returns the configured PDF data directory.
    """
    mcp = MCPServer(
        name="tracerag",
        title="TraceRAG",
        description=(
            "Local PDF RAG backend with bge-m3 vector retrieval + BM25 hybrid "
            "fusion (RRF), HyDE query rewriting, and bge-reranker CrossEncoder "
            "reranking. Supports collection isolation and code-query context "
            "expansion. Exposes retrieval, answering, stats, and collection "
            "listing as MCP tools."
        ),
        version="0.2.0",
    )

    @mcp.tool()
    def rag_search(
        query: str,
        retrieve_top_k: int = 5,
        rerank_top_k: int = 3,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant PDF chunks with hybrid vector + BM25 retrieval,
        HyDE query rewriting, and CrossEncoder reranking.

        Returns the ranked sources (with full chunk content for downstream
        generation) plus a detailed trace (HyDE status, RRF fusion scores,
        rerank scores, per-stage timing). Does NOT generate an answer —
        callers may synthesize one from the returned sources.

        Args:
            query: The user question.
            retrieve_top_k: Candidates to fetch before reranking (1-30).
            rerank_top_k: Final chunks to keep after reranking (1-20).
            collection: Optional collection name to scope retrieval.
        """
        pipeline = get_pipeline()
        if pipeline is None:
            return {"error": "RAG pipeline is not ready yet."}
        try:
            docs, trace = pipeline.retrieve_with_trace(
                query,
                retrieve_top_k=retrieve_top_k,
                rerank_top_k=rerank_top_k,
                collection=collection or None,
            )
            score_map = {
                item.get("chunk_id"): item.get("score")
                for item in trace.get("rerank_scores", [])
                if item.get("chunk_id")
            }
            sources = [
                {**_source_from_doc(doc, include_content=True), "score": score_map.get(doc.metadata.get("chunk_id"))}
                for doc in docs
            ]
            return {
                "query": query,
                "collection": collection,
                "sources": sources,
                "trace": trace,
                "source_count": len(sources),
            }
        except Exception as exc:  # noqa: BLE001 - surface as tool error, keep server alive.
            logger.exception("rag_search failed")
            return {"error": f"{type(exc).__name__}: {exc}"}

    @mcp.tool()
    def rag_ask(
        query: str,
        retrieve_top_k: int = 5,
        rerank_top_k: int = 3,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Answer a question against the PDF knowledge base end-to-end.

        Runs hybrid retrieval, reranking, context expansion (for code queries),
        and DeepSeek grounded generation with chunk citations. Returns the
        answer, citations, sources, and a full retrieval+generation trace.

        Args:
            query: The user question.
            retrieve_top_k: Candidates to fetch before reranking (1-30).
            rerank_top_k: Final chunks to keep after reranking (1-20).
            collection: Optional collection name to scope retrieval.
        """
        pipeline = get_pipeline()
        if pipeline is None:
            return {"error": "RAG pipeline is not ready yet."}
        try:
            result = pipeline.ask_with_sources(
                query,
                retrieve_top_k=retrieve_top_k,
                rerank_top_k=rerank_top_k,
                collection=collection or None,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("rag_ask failed")
            return {"error": f"{type(exc).__name__}: {exc}"}

    @mcp.tool()
    def rag_corpus_stats() -> dict[str, Any]:
        """Return corpus, chunk, and model statistics for the loaded knowledge base.

        Includes PDF/page counts, collection distribution, chunk length
        distribution, and the embedding/generation model names.
        """
        pipeline = get_pipeline()
        if pipeline is None:
            return {"error": "RAG pipeline is not ready yet."}
        try:
            data_dir = get_data_dir()
            stats = build_corpus_stats(data_dir, pipeline.chunks, pipeline.model)
            sources = sorted({str(c.metadata.get("source")) for c in pipeline.chunks})
            collections = sorted(
                {str(c.metadata.get("collection", "default")) for c in pipeline.chunks}
            )
            return {
                **stats,
                "source_count": len(sources),
                "collections": collections,
                "sources": sources,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("rag_corpus_stats failed")
            return {"error": f"{type(exc).__name__}: {exc}"}

    @mcp.tool()
    def rag_collections() -> dict[str, Any]:
        """List available PDF collections (data subdirectories) and their files.

        Each collection maps to a list of relative PDF source paths.
        """
        try:
            data_dir = get_data_dir()
            collections = list_pdf_collections(data_dir)
            return {
                "collections": collections,
                "collection_count": len(collections),
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("rag_collections failed")
            return {"error": f"{type(exc).__name__}: {exc}"}

    return mcp
