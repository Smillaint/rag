# -*- coding: utf-8 -*-
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)

MIN_PASSAGE_LENGTH = 12
DEFAULT_LEXICAL_SCORE = 0.2
GENERATION_TEMPERATURE = 0.3
GENERATION_MAX_TOKENS = 1024
MAX_EXTRACTIVE_PASSAGES = 5


@dataclass
class GenerationResult:
    """Result of answer generation with diagnostics."""

    answer: str
    mode: str
    error: str | None = None


def load_generator(
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
) -> tuple[OpenAI, str]:
    """Initialize an OpenAI-compatible chat client.

    Compatible providers include DeepSeek, Qwen (DashScope), and GLM (Zhipu).
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info("Generator initialized with model: %s", model)
    return client, model


def _format_source(metadata: dict) -> str:
    """Format chunk metadata into a human-readable source label."""
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    chunk_id = metadata.get("chunk_id")

    parts = [str(source)]
    if page is not None:
        parts.append(f"page {page}")
    if chunk_id is not None:
        parts.append(f"chunk {chunk_id}")
    return ", ".join(parts)


def _query_terms(query: str) -> set[str]:
    """Extract ASCII tokens and Chinese characters for lexical matching."""
    words = {word.lower() for word in re.findall(r"[A-Za-z0-9_./#-]{2,}", query)}
    chinese_chars = {char for char in query if "\u4e00" <= char <= "\u9fff"}
    return words | chinese_chars


def _split_passages(text: str) -> list[str]:
    """Split text into clean passages of at least MIN_PASSAGE_LENGTH characters."""
    passages = []
    for part in re.split(r"(?<=[。！？!?；;])|\n+", text):
        cleaned = " ".join(part.split())
        if len(cleaned) >= MIN_PASSAGE_LENGTH:
            passages.append(cleaned)
    return passages


def generate_extractive_answer(
    query: str,
    docs: list["Document"],
    max_passages: int = MAX_EXTRACTIVE_PASSAGES,
) -> str:
    """Build a deterministic local answer when the chat model is unavailable."""
    if not docs:
        return "未在本地文档中检索到相关内容。"

    terms = _query_terms(query)
    scored: list[tuple[float, int, str]] = []
    for chunk_number, doc in enumerate(docs, start=1):
        for passage in _split_passages(doc.page_content):
            normalized = passage.lower()
            lexical_score = sum(1 for term in terms if term and term.lower() in normalized)
            if lexical_score == 0:
                lexical_score = DEFAULT_LEXICAL_SCORE
            scored.append((lexical_score, chunk_number, passage))

    if not scored:
        return "检索到了相关片段，但片段内容过短，无法形成可靠回答。"

    selected = sorted(scored, key=lambda item: item[0], reverse=True)[:max_passages]
    lines = ["当前大模型生成不可用，以下是基于检索片段的本地抽取式回答："]
    for _, chunk_number, passage in selected:
        lines.append(f"- {passage} [Chunk {chunk_number}]")
    lines.append("以上内容仅来自检索片段，建议结合引用页码回看原文。")
    return "\n".join(lines)


def generate_answer_result(
    client: OpenAI,
    model: str,
    query: str,
    docs: list["Document"],
) -> GenerationResult:
    """Generate an answer and return generation diagnostics for API traces."""
    if not docs:
        return GenerationResult(answer="未在本地文档中检索到相关内容。", mode="empty_context")

    context_parts = []
    for index, doc in enumerate(docs, start=1):
        source = _format_source(doc.metadata)
        context_parts.append(f"[Chunk {index}] (source: {source})\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a document-grounded QA assistant. Answer only from the retrieved "
        "document chunks. Keep the answer concise and cite chunk numbers such as "
        "[Chunk 1]. If the chunks do not contain the answer, say that no relevant "
        "content was found in the documents. Do not invent facts. When the retrieved "
        "chunks contain source code, explain the code using the available code and "
        "nearby comments: describe the purpose, main variables, function calls, "
        "execution flow, and cleanup steps."
    )

    user_prompt = (
        "Use the following document chunks to answer the question.\n\n"
        f"Document chunks:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=GENERATION_TEMPERATURE,
            max_tokens=GENERATION_MAX_TOKENS,
        )
    except Exception as exc:
        logger.warning("LLM generation failed, using extractive fallback: %s", exc)
        return GenerationResult(
            answer=generate_extractive_answer(query, docs),
            mode="local_extractive_fallback",
            error=f"{type(exc).__name__}: {exc}",
        )

    content = response.choices[0].message.content or ""
    return GenerationResult(answer=content, mode="llm")


def generate_answer(
    client: OpenAI,
    model: str,
    query: str,
    docs: list["Document"],
) -> str:
    """Generate an answer from retrieved and reranked context documents."""
    return generate_answer_result(client, model, query, docs).answer
