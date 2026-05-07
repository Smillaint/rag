# -*- coding: utf-8 -*-
from openai import OpenAI


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


def generate_answer(client, model: str, query: str, docs: list) -> str:
    """Generate an answer from retrieved and reranked context documents."""
    if not docs:
        return "No relevant content was found in the documents."

    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source = _format_source(doc.metadata)
        context_parts.append(f"[Chunk {i}] (source: {source})\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a document-grounded QA assistant. Answer only from the retrieved "
        "document chunks. Keep the answer concise and cite chunk numbers such as "
        "[Chunk 1]. If the chunks do not contain the answer, say that no relevant "
        "content was found in the documents. Do not invent facts."
    )

    user_prompt = (
        "Use the following document chunks to answer the question.\n\n"
        f"Document chunks:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content
