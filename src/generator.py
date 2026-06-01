# -*- coding: utf-8 -*-
import re
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class GenerationResult:
    answer: str
    mode: str
    error: str | None = None


def load_generator(
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
):
    """
    Initialize an OpenAI-compatible chat client.

    Examples:
    - DeepSeek: base_url="https://api.deepseek.com", model="deepseek-chat"
    - Qwen: base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-plus"
    - GLM: base_url="https://open.bigmodel.cn/api/paas/v4", model="glm-4-flash"
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Generator initialized with model: {model}")
    return client, model


def _format_source(metadata: dict) -> str:
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
    words = {word.lower() for word in re.findall(r"[A-Za-z0-9_./#-]{2,}", query)}
    chinese_chars = {char for char in query if "\u4e00" <= char <= "\u9fff"}
    return words | chinese_chars


def _split_passages(text: str) -> list[str]:
    passages = []
    for part in re.split(r"(?<=[。！？!?；;])|\n+", text):
        cleaned = " ".join(part.split())
        if len(cleaned) >= 12:
            passages.append(cleaned)
    return passages


def generate_extractive_answer(query: str, docs: list, max_passages: int = 5) -> str:
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
                lexical_score = 0.2
            scored.append((lexical_score, chunk_number, passage))

    if not scored:
        return "检索到了相关片段，但片段内容过短，无法形成可靠回答。"

    selected = sorted(scored, key=lambda item: item[0], reverse=True)[:max_passages]
    lines = [
        "当前大模型生成不可用，以下是基于检索片段的本地抽取式回答："
    ]
    for _, chunk_number, passage in selected:
        lines.append(f"- {passage} [Chunk {chunk_number}]")
    lines.append("以上内容仅来自检索片段，建议结合引用页码回看原文。")
    return "\n".join(lines)


def generate_answer_result(client, model: str, query: str, docs: list) -> GenerationResult:
    """Generate an answer and return generation diagnostics for API traces."""
    if not docs:
        return GenerationResult(answer="未在本地文档中检索到相关内容。", mode="empty_context")

    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source = _format_source(doc.metadata)
        context_parts.append(f"[Chunk {i}] (source: {source})\n{doc.page_content}")
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
            temperature=0.3,
            max_tokens=1024,
        )
    except Exception as exc:
        return GenerationResult(
            answer=generate_extractive_answer(query, docs),
            mode="local_extractive_fallback",
            error=f"{type(exc).__name__}: {exc}",
        )

    content = response.choices[0].message.content or ""
    return GenerationResult(answer=content, mode="llm")


def generate_answer(client, model: str, query: str, docs: list) -> str:
    """Generate an answer from retrieved and reranked context documents."""
    return generate_answer_result(client, model, query, docs).answer
