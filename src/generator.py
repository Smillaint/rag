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
        return "文档中未找到相关内容"

    context_parts = []
    for i, doc in enumerate(docs, start=1):
        source = _format_source(doc.metadata)
        context_parts.append(f"[片段{i}]（来源：{source}）\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    system_prompt = (
        "你是一个专业的文档问答助手。"
        "请根据检索到的文档片段，准确、简洁地回答用户问题。"
        "如果文档中没有相关信息，请直接说'文档中未找到相关内容'，不要编造答案。"
    )

    user_prompt = (
        "请根据以下文档片段回答问题。\n\n"
        f"【文档片段】\n{context}\n\n"
        f"【用户问题】\n{query}\n\n"
        "【回答】"
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
