# RAG 项目简历包装建议

## 推荐项目名称

本地知识库 PDF 智能问答系统（RAG）

## 一句话介绍

基于 LangChain、Chroma、BM25、CrossEncoder Reranker 和 OpenAI 兼容大模型接口，实现面向本地 PDF 文档的可追溯知识库问答系统，支持混合检索、精排、引用溯源和检索质量评测。

## 技术栈

Python, LangChain, ChromaDB, HuggingFace Embeddings, SentenceTransformers CrossEncoder, BM25, PyMuPDF, OpenAI SDK, DeepSeek API

## 简历 Bullet 示例

- 设计并实现本地 PDF RAG 问答链路，完成 PDF 解析、文本切分、向量索引构建、混合检索、精排和答案生成的端到端流程。
- 优化文档分片策略，基于段落、换行、中文标点和重叠窗口进行递归切分，并记录 `source/page/chunk_in_page/char_start/char_end` 元数据，支持检索结果定位和调参分析。
- 采用 Chroma 向量检索与 BM25 关键词检索融合，提高中文/中英混合文档场景下的召回稳定性，并通过去重逻辑保留 `source/page/chunk_id` 元数据用于溯源。
- 引入 CrossEncoder Reranker 对候选片段二次排序，将生成阶段上下文控制在 Top-K 高相关片段，降低无关上下文干扰。
- 针对代码解释类问题设计上下文扩展策略，自动识别代码查询并引入相邻 chunk，解决代码块被分片切断导致回答缺失的问题。
- 基于 OpenAI 兼容接口接入 DeepSeek 模型，设计只基于检索片段回答的提示词，并要求输出引用片段编号，减少幻觉风险。
- 补充检索评测脚本，支持以 JSONL 配置问题、期望来源和关键词，输出 source hit、keyword hit 与 pass rate，用于量化优化检索效果。

## 面试讲解顺序

1. 先讲业务目标：把本地 PDF 变成可问答的知识库，并且回答要能追溯到原文页码。
2. 再讲架构：PDF 解析 -> 带页码和字符偏移的 chunk -> 向量库/BM25 -> 混合召回 -> rerank -> LLM 生成。
3. 重点讲取舍：只用向量检索容易漏掉专有名词，只用 BM25 对语义问题不稳定，所以做混合检索。
4. 讲可靠性：保留元数据、提示词约束、引用 chunk、评测脚本。
5. 讲后续优化：可加入 query rewrite、多路召回权重、RAGAS/人工标注集、FastAPI 服务化和前端上传入口。

## 可以继续增强的方向

- 增加 FastAPI 接口和一个简单前端，让项目从 CLI demo 升级为可交互应用。
- 引入 query rewrite 或 multi-query retrieval，提高复杂问题召回。
- 增加检索指标 Recall@K、MRR、nDCG，并沉淀一组真实评测集。
- 支持 Markdown、Word、网页等多格式文档。
- 加入 Dockerfile，降低面试官或 reviewer 本地复现成本。
