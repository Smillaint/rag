# -*- coding: utf-8 -*-
# src/generator.py

from openai import OpenAI


def load_generator(
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat"
):
    """
    初始化生成器
    - DeepSeek:  base_url="https://api.deepseek.com",       model="deepseek-chat"
    - Qwen:      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-plus"
    - GLM:       base_url="https://open.bigmodel.cn/api/paas/v4", model="glm-4-flash"
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"生成器初始化完成，使用模型：{model}")
    return client, model


def generate_answer(client, model: str, query: str, docs: list) -> str:
    """
    基于检索到的文档片段，调用大模型生成答案
    :param client: OpenAI client
    :param model:  模型名称
    :param query:  用户问题
    :param docs:   rerank 后的 Document 列表
    :return:       生成的答案字符串
    """
    # 拼接上下文
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "未知来源")
        context_parts.append(f"[片段{i+1}]（来源：{source}）\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # 构建 Prompt（注意：内层用单引号，避免与外层双引号冲突）
    system_prompt = (
        "你是一个专业的文档问答助手。"
        "请根据以下检索到的文档片段，准确、简洁地回答用户问题。"
        "如果文档中没有相关信息，请直接说'文档中未找到相关内容'，不要编造答案。"
    )

    user_prompt = (
        "请根据以下文档片段回答问题。\n\n"
        f"【文档片段】\n{context}\n\n"
        f"【用户问题】\n{query}\n\n"
        "【回答】"
    )

    # 调用模型
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )

    answer = response.choices[0].message.content
    return answer
