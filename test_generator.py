# -*- coding: utf-8 -*-
import os

from src.chain import RAGPipeline


API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set DEEPSEEK_API_KEY before running this test.")


pipeline = RAGPipeline.from_paths(api_key=API_KEY)

queries = [
    "联邦学习如何保护数据隐私？",
    "项目中使用了什么硬件加速方案？",
    "数据选择算法的核心指标是什么？",
]

for query in queries:
    print(f"\n{'=' * 50}")
    print(f"问题：{query}")
    print(f"回答：\n{pipeline.ask(query)}")
