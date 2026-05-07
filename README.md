# 本地 PDF RAG 智能问答系统

这是一个面向本地 PDF 文档的 RAG（Retrieval-Augmented Generation，检索增强生成）问答项目。项目支持 PDF 解析、可追溯 chunk 分片、向量检索、BM25 关键词检索、CrossEncoder 精排，以及通过 DeepSeek/OpenAI 兼容接口生成带引用的回答。

## 功能特性

- 按页读取本地 PDF，并保留 `source`、`page`、`chunk_id` 等元数据。
- 基于段落、换行、中文标点和重叠窗口进行递归分片。
- 为每个 chunk 记录 `chunk_index`、`chunk_in_page`、`char_start`、`char_end`、`chunk_length`，便于调试、溯源和后续评测。
- 使用 Chroma 构建可持久化的本地向量库。
- 融合向量检索和 BM25 关键词检索，提高中文/中英混合文档的召回稳定性。
- 使用 CrossEncoder Reranker 对候选片段进行精排，提高最终上下文相关性。
- 针对代码解释类问题自动扩大检索窗口，并引入相邻 chunk，避免代码块被分片切断。
- 通过提示词约束模型只基于检索片段回答，并输出 chunk 引用，降低幻觉风险。
- 提供轻量级检索评测脚本，可用 JSONL 配置问题、期望来源和关键词。

## 系统架构

```text
PDF 文件
  -> PyMuPDF 按页抽取文本
  -> 带页码/字符偏移元数据的递归 chunk 分片
  -> Chroma 向量索引 + BM25 关键词索引
  -> 混合召回候选片段
  -> CrossEncoder 精排
  -> 代码类问题的相邻 chunk 上下文扩展
  -> OpenAI 兼容大模型生成答案
```

## 目录结构

```text
src/
  loader.py       PDF 读取、chunk 分片、分片统计
  retriever.py    Chroma 向量检索、BM25、混合召回
  reranker.py     CrossEncoder 精排
  generator.py    OpenAI 兼容接口答案生成
  chain.py        端到端 RAGPipeline 编排
  evaluate.py     检索评测命令行工具
main.py           命令行入口
examples/         示例评测问题
docs/             架构说明和简历包装建议
```

## 环境安装

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

把需要问答的 PDF 放到本地 `data/` 目录：

```text
data/
  your-document.pdf
```

`data/` 和 `vectorstore/` 已加入 `.gitignore`，不会上传到 GitHub。

## API Key 配置

项目默认使用 DeepSeek 的 OpenAI 兼容接口：

```text
base_url = https://api.deepseek.com
model    = deepseek-chat
```

推荐在本地 `.env` 文件中配置密钥：

```text
DEEPSEEK_API_KEY="your_api_key_here"
```

`.env` 已被 `.gitignore` 忽略，不会提交到仓库。也可以在 PowerShell 中临时设置：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

## 本地运行

首次运行、PDF 变化、分片参数变化时，需要重建向量库：

```powershell
.\.venv\Scripts\python.exe main.py --rebuild "这份文档主要讲了什么？"
```

日常问答直接复用已有向量库：

```powershell
.\.venv\Scripts\python.exe main.py "密码实训平台如何开始实验？"
```

进入交互模式：

```powershell
.\.venv\Scripts\python.exe main.py
```

调整检索和分片参数：

```powershell
.\.venv\Scripts\python.exe main.py --retrieve-top-k 8 --rerank-top-k 5 --chunk-size 512 --chunk-overlap 64 "你的问题"
```

修改 chunk 参数后需要重建向量库：

```powershell
.\.venv\Scripts\python.exe main.py --rebuild --chunk-size 768 --chunk-overlap 96 "你的问题"
```

## 代码问答增强

当问题中包含 `代码`、`程序`、`函数`、`main.c`、`demo`、`编程`、`解释` 等关键词时，系统会自动进入代码问答模式：

- 自动提高召回数量和精排数量。
- 基于 `chunk_index` 引入命中片段前后的相邻 chunk。
- 支持代码跨页时的上下文拼接。
- 生成阶段会重点解释代码目的、主要变量、函数调用、执行流程和资源释放。

示例：

```powershell
.\.venv\Scripts\python.exe main.py "解释一下sm4实验给的代码"
```

## 检索评测

在 `examples/eval_questions.jsonl` 中配置评测问题：

```jsonl
{"question":"文档的核心主题是什么？","expected_sources":["paper.pdf"],"expected_keywords":["retrieval"]}
```

不使用 reranker 的快速评测：

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --eval-file ./examples/eval_questions.jsonl --skip-rerank
```

使用完整精排的评测：

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --eval-file ./examples/eval_questions.jsonl
```

评测报告包含 `pass_rate`、`source_hit`、`keyword_hit` 和返回片段的来源元数据。

## 注意事项

- 首次加载 embedding/reranker 模型可能需要联网。
- 默认 Hugging Face endpoint 设置为 `https://hf-mirror.com`，便于国内环境下载模型。
- 不要提交 API key、本地 PDF、生成的向量库或 `.venv`。
- 日常问答不要加 `--rebuild`，否则会重新计算所有 chunk 的 embedding，耗时较长。
