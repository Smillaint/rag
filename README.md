# TraceRAG

基于本地 PDF 文档的 RAG（Retrieval-Augmented Generation，检索增强生成）问答系统。通过 collection 隔离、RRF 混合检索、HyDE 查询改写、CrossEncoder 精排和 IR 评测框架，为 LLM Agent 提供高质量、可追溯、可评测的知识检索能力。

## 功能特性

- 按 collection（子目录）隔离语料，检索时可限定领域，避免跨域误召回。
- 按 PyMuPDF 页面抽取文本，保留 `source`、`source_path`、`collection`、`page`、`chunk_id`、`char_start`、`char_end` 等可追溯元数据。
- 基于段落、换行、中文标点和重叠窗口进行递归分片，为每个 chunk 记录 `chunk_index`、`chunk_in_page`、`content_hash`、`chunk_length`。
- chunk 缓存到 `.rag_cache/`，通过 sha256 + mtime 文件指纹做增量索引，新增/修改/删除文档时只同步受影响 chunk。
- 使用 **bge-m3**（1024 维，多语言）构建 Chroma 向量库，embedding 指纹自动检测模型变更并重建。
- **加权 RRF 混合检索**：向量检索 + BM25 关键词检索通过 Reciprocal Rank Fusion 融合，无需分数归一化，重叠命中的文档自动提权。
- **HyDE 查询改写**：检索前用 LLM 生成假想答案文档，用于向量检索缩小问题-答案语义鸿沟；BM25 和 reranker 仍使用原始 query，失败自动降级。
- 使用 **bge-reranker-v2-m3**（多语言 CrossEncoder）对候选片段精排，辅以词面匹配加权修正中文技术词排序。
- 针对代码解释类问题自动扩大检索窗口，并引入相邻 chunk，避免代码块被分片切断。
- 通过提示词约束模型只基于检索片段回答，并输出 chunk 引用，降低幻觉风险；API 不可用时自动降级为本地抽取式回答。
- **IR 评测框架**：Recall@K、Precision@K、MRR、NDCG@K（多级相关性），支持五配置消融对比（vector-only / BM25-only / hybrid-RRF / +rerank / full+HyDE）。
- 问答接口返回完整 trace：HyDE 状态、RRF 融合分数、rerank 分数、各阶段耗时，按日期写入 JSONL 调用链日志。

## 技术栈

### 检索引擎

| 环节 | 技术选型 | 设计要点 |
|---|---|---|
| Embedding | BAAI/bge-m3（1024 维） | 多语言支持，L2 归一化，中文语义优于 MiniLM 系列；embedding 指纹（`.embedding_meta.json`）自动检测模型变更并重建向量库，避免维度不匹配 |
| 向量库 | Chroma | 本地持久化，惰性分批写入（generator 逐批 yield），按 collection 元数据过滤检索域 |
| 关键词检索 | rank_bm25 + jieba | 按 collection 懒加载 BM25 索引，首次查询时构建并缓存，避免全量预建内存压力 |
| 混合融合 | 加权 Reciprocal Rank Fusion (RRF) | rank-based 融合，无需向量距离与 BM25 分数的归一化；重叠命中文档累加分数自动提权；向量检索失败自动降级为纯 BM25 |
| 精排 | BAAI/bge-reranker-v2-m3 | 多语言 CrossEncoder 成对打分；辅以词面匹配加权（coverage × 0.15）修正中文技术术语排序；仅对 top candidates 精排控制延迟 |

### 查询理解与生成

| 环节 | 技术选型 | 设计要点 |
|---|---|---|
| 查询改写 | HyDE (Hypothetical Document Embeddings) | 检索前用 LLM 生成假想答案文档，用于向量检索缩小问题-答案语义鸿沟；BM25 和 reranker 仍使用原始 query 保留精确词匹配；LLM 调用失败自动降级到原 query |
| 代码查询检测 | 关键词触发 + 动态 top_k | 检测到代码类问题时自动扩大召回（top_k=12）和精排（top_k=6），并引入命中片段前后的相邻 chunk（前 2 后 3），避免代码块被分片切断 |
| 答案生成 | DeepSeek（OpenAI 兼容接口） | 提示词约束模型只基于检索片段回答并输出 chunk 引用；API 不可用时降级为本地抽取式回答（基于 query term overlap 的 passage 排序） |

### 数据管线

| 环节 | 技术选型 | 设计要点 |
|---|---|---|
| PDF 解析 | PyMuPDF (fitz) | 按页抽取文本，保留页码和字符偏移元数据 |
| 文本分片 | LangChain `RecursiveCharacterTextSplitter` | 中文标点感知（。？！；，），chunk_size=900，overlap=120 |
| 增量索引 | sha256 + mtime 文件指纹 | 只比较稳定字段（name/source_path/collection/size/mtime_ns/sha256），不比较运行时 path 字符串；新增/修改/删除文档时只同步受影响 chunk |
| Collection 隔离 | 子目录即 collection | `data/algorithm/`、`data/tcpip/` 等自然分区，检索时可限定领域避免跨域误召回 |

### 评测与可观测性

| 环节 | 技术选型 | 设计要点 |
|---|---|---|
| IR 指标 | 纯标准库实现 | Recall@K / Precision@K / MRR / NDCG@K，多级相关性（source 匹配=1，source+page 匹配=2）；零额外依赖 |
| 消融框架 | 五配置对比 | vector-only / BM25-only / hybrid-RRF / +rerank / full+HyDE，输出对比表格和 JSON 报告 |
| 调用链日志 | JSONL 按日期轮转 | `logs/YYYY-MM-DD.jsonl`，记录 query、collection、top-k、answer 预览、citations、完整 trace |
| Trace 诊断 | 结构化 trace dict | HyDE 状态、RRF 融合分数、rerank 分数、各阶段耗时（hyde / retrieval / rerank / generation / total），便于定位检索质量和性能瓶颈 |

### 服务与部署

| 环节 | 技术选型 | 设计要点 |
|---|---|---|
| 公网网关 | Cloudflare Worker（`smillick-org`） | 路由 `smillick.org/*`，`run_worker_first` 让 Worker 在静态资源之前接管 `/api/v1/rag/*`；公开 health，Bearer 鉴权 + 两级 Rate Limit + 安全头 + 上游错误脱敏 |
| 源站服务 | FastAPI + Uvicorn | lifespan 管理单例 pipeline，请求复用内存中的 chunk、BM25 索引、向量库和 reranker；监听 `127.0.0.1:8000` |
| 公网接入 | Cloudflare Tunnel（cloudflared） | 出站 QUIC 隧道，无需开入站端口；将 `rag.smillick.org` 转发到本机 `127.0.0.1:8000` |
| 前端控制台 | 原生 HTML/CSS/JS | 项目进度、collection 列表、源码浏览、运行状态；由 Worker `ASSETS` 绑定托管 |
| 模型管理 | HF mirror + 离线模式 | `hf-mirror.com` 镜像下载，`HF_HUB_OFFLINE=1` 跳过启动时 HEAD 请求；断点续传下载脚本 |

## 系统架构

```text
客户端（任意网络）
  -> Cloudflare Worker smillick.org/api/v1/rag/*
     ├─ GET/HEAD /health            公开，透传源站
     ├─ GET /collections            Bearer 鉴权 + API Rate Limit
     ├─ GET /stats                  Bearer 鉴权 + API Rate Limit
     └─ POST /ask                  Bearer 鉴权 + Ask Rate Limit + 65536 字节限制
  -> Worker 以独立 origin key 回源 https://rag.smillick.org
  -> Cloudflare Tunnel（cloudflared，出站 QUIC）
  -> 127.0.0.1:8000 FastAPI
     -> 代码查询检测（扩 top_k）
     -> HyDE 生成假想文档
     -> 向量检索(假想文档) + BM25(原query) -> RRF 融合
     -> bge-reranker 精排(原query)
     -> 代码查询邻居扩展
     -> DeepSeek 生成答案
     -> JSONL 调用链日志
```

客户端应始终访问 `https://smillick.org/api/v1/rag/*`。`rag.smillick.org` 是 Worker 回源专用的源站地址，普通客户端不应直连。

## 目录结构

```text
src/
  loader.py          PDF 读取、chunk 分片、元数据分配
  retriever.py       bge-m3 embedding、Chroma 向量库、BM25、RRF 混合检索、惰性分批写入
  reranker.py        bge-reranker-v2-m3 精排、词面匹配加权
  query_rewrite.py   HyDE 假想文档生成
  generator.py       OpenAI 兼容接口答案生成、抽取式降级
  chain.py           端到端 RAGPipeline 编排
  index_cache.py     sha256 文件指纹、增量索引、chunk 缓存
  query_log.py       JSONL 调用链日志
  corpus_stats.py    语料统计（页数、chunk 分布、模型信息）
  metrics.py         IR 评测指标（Recall@K / MRR / NDCG@K / Precision@K）
  evaluate.py        消融评测命令行工具
main.py              CLI 入口
server.py            FastAPI 服务入口
frontend/            前端控制台（HTML/CSS/JS）
worker/
  index.js           Cloudflare Worker（网关鉴权、限流、安全头、回源代理）
scripts/             模型下载、向量库重建、smoke 测试、Tunnel 安装
examples/            评测问题（JSONL）
tests/               pytest 测试套件 + Worker vitest 测试
docs/                架构设计、部署指南、大型语料说明
.wrangler / node_modules/   Worker 本地开发依赖（gitignore）
```

## 环境安装

### Python 检索引擎

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Cloudflare Worker 网关

```powershell
npm install
```

把需要问答的 PDF 按子目录放到 `data/` 目录，子目录名即为 collection 名：

```text
data/
  algorithm/
    第1章-基础.pdf
    第3章-图论基础.pdf
  netsec_lab/
    实验1 TCP协议漏洞利用指导手册.pdf
  tcpip/
    TCPIP详解 卷1：协议.pdf
```

`data/`、`vectorstore/`、`.rag_cache/`、`logs/`、`.env`、`.wrangler/`、`node_modules/` 已加入 `.gitignore`，不会上传到 GitHub。

## 模型下载

项目使用两个 Hugging Face 模型，首次运行需下载（约 4.4GB）：

| 模型 | 用途 | 大小 |
|---|---|---|
| BAAI/bge-m3 | Embedding（1024 维） | ~2.2GB |
| BAAI/bge-reranker-v2-m3 | CrossEncoder 精排 | ~2.2GB |

默认使用 `https://hf-mirror.com` 镜像。如果 `huggingface_hub` 下载失败，可使用脚本手动下载：

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
```

下载完成后设置离线模式避免重复请求：

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

## API Key 配置

项目涉及三层密钥，均通过环境变量注入，不写入代码或配置文件：

| 变量 | 用途 | 配置位置 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek LLM 答案生成 | `.env` |
| `RAG_API_KEY` | FastAPI 源站应用层鉴权 | `.env` |
| `RAG_GATEWAY_API_KEY` | Worker 网关层 Bearer 鉴权（客户端携带） | Wrangler secret |
| `RAG_ORIGIN_API_KEY` | Worker 回源时使用的 origin key | Wrangler secret |

**密钥关系**：

- `RAG_ORIGIN_API_KEY` **必须**与源站 `.env` 中的 `RAG_API_KEY` 完全相同——Worker 用它重建 `Authorization` 头回源，FastAPI 用 `RAG_API_KEY` 校验。两者不一致则回源 401。
- `RAG_GATEWAY_API_KEY` 是客户端访问网关的密钥，应与 `RAG_ORIGIN_API_KEY` 分离。
- 快速上线时三个 key 可以先用同一个值，但生产环境建议 `RAG_GATEWAY_API_KEY` 与 `RAG_ORIGIN_API_KEY` 分离并独立轮换；`RAG_ORIGIN_API_KEY` 与 `RAG_API_KEY` 始终成对相同。

在本地 `.env` 文件中配置 DeepSeek 和源站密钥：

```text
DEEPSEEK_API_KEY="<your_deepseek_key>"
RAG_API_KEY="<must_match_RAG_ORIGIN_API_KEY>"
```

`.env` 已被 `.gitignore` 忽略，不会提交到仓库。Worker 密钥通过 `wrangler secret put` 注入，不落盘到配置文件。也支持其他 OpenAI 兼容接口（Qwen、GLM 等），修改 `base_url` 和 `model` 即可。

## 本地运行

首次运行、PDF 变化、分片参数变化时，需要重建向量库：

```powershell
.\.venv\Scripts\python.exe main.py --rebuild "这份文档主要讲了什么？"
```

日常问答直接复用已有向量库：

```powershell
.\.venv\Scripts\python.exe main.py "Dijkstra算法适合解决什么问题？" --collection algorithm
```

进入交互模式：

```powershell
.\.venv\Scripts\python.exe main.py
```

列出可用 collection：

```powershell
.\.venv\Scripts\python.exe main.py --list-collections
```

调整检索和分片参数：

```powershell
.\.venv\Scripts\python.exe main.py --retrieve-top-k 8 --rerank-top-k 5 --chunk-size 768 --chunk-overlap 96 "你的问题"
```

修改 chunk 参数后需要重建向量库：

```powershell
.\.venv\Scripts\python.exe main.py --rebuild --chunk-size 768 --chunk-overlap 96 "你的问题"
```

如果切换了 embedding 模型，向量库会通过 `.embedding_meta.json` 指纹自动检测并重建，无需手动加 `--rebuild`。

## 在线 API 调用

生产环境客户端应通过 `https://smillick.org/api/v1/rag/*` 访问，携带 `Authorization: Bearer $RAG_GATEWAY_API_KEY`：

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---|---|
| GET / HEAD | `/api/v1/rag/health` | 公开 | 源站健康检查 |
| GET | `/api/v1/rag/collections` | Bearer | 可用 collection 列表 |
| GET | `/api/v1/rag/stats` | Bearer | 文档、页数、chunk、模型统计 |
| POST | `/api/v1/rag/ask` | Bearer | 问答接口，`Content-Type: application/json`，body ≤ 65536 字节 |

调用示例：

```powershell
$base = "https://smillick.org/api/v1/rag"
$key = $env:RAG_GATEWAY_API_KEY

# 健康检查（公开）
curl "$base/health"

# 问答（需鉴权）
curl "$base/ask" `
    -Method POST `
    -Headers @{Authorization = "Bearer $key"} `
    -ContentType "application/json" `
    -Body '{"query":"Dijkstra算法适合解决什么问题？","collection":"algorithm"}'
```

`/ask` 接口返回 `answer`、`citations`、`sources` 和 `trace`：

- `citations`：答案引用的 chunk 编号、来源、页码。
- `sources`：检索片段的完整元数据（source、page、chunk_id、char_start、char_end、preview）。
- `trace`：HyDE 状态、RRF 融合分数、rerank 分数、各阶段耗时（hyde / retrieval / rerank / generation / total）。

## FastAPI 本地服务

启动本地服务（开发或源站）：

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

服务启动时会加载一次 `RAGPipeline`，后续请求复用内存中的 PDF 分片、BM25 索引、向量库和 Reranker。

本地接口与 Worker 网关路径一致（`/health`、`/collections`、`/stats`、`/ask`），但无 Rate Limit 和安全头。详见 `docs/SERVER_DEPLOY.md`。

## 代码问答增强

当问题中包含 `代码`、`程序`、`函数`、`main.c`、`demo`、`编程`、`解释` 等关键词时，系统会自动进入代码问答模式：

- 自动提高召回数量（top_k=12）和精排数量（top_k=6）。
- 基于 `chunk_index` 引入命中片段前后的相邻 chunk（前 2 后 3）。
- 支持代码跨页时的上下文拼接。
- 生成阶段会重点解释代码目的、主要变量、函数调用、执行流程和资源释放。

示例：

```powershell
.\.venv\Scripts\python.exe main.py "解释一下sm4实验给的代码" --collection netsec_lab
```

## 检索评测

在 `examples/eval_questions.jsonl` 中配置评测问题，支持 source 和 page 级相关性标注：

```jsonl
{"question":"Dijkstra算法适合解决什么问题？","collection":"algorithm","relevant_sources":["第3章-图论基础.pdf","第7章-图论提高.pdf"],"relevant_source_pages":[{"source":"第3章-图论基础.pdf","pages":[]}],"any_keywords":["Dijkstra","最短路径"]}
```

运行五配置消融评测：

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --eval-file ./examples/eval_questions.jsonl --output ./eval_report.json
```

只运行部分配置：

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --eval-file ./examples/eval_questions.jsonl --configs vector_only hybrid_rrf
```

评测输出消融对比表：

```text
Config              R@3      P@3      N@3      R@5      P@5      N@5      MRR      ms
--------------------------------------------------------------------------------------------
vector_only         0.610    0.833    0.872    1.000    0.829    0.945    0.929    1525
bm25_only           0.539    0.571    0.653    0.929    0.557    0.783    0.779    6112
hybrid_rrf          1.000    0.786    0.947    1.000    0.471    0.947    0.952    1173
hybrid_rrf_rerank   1.000    0.833    0.967    1.000    0.500    0.967    0.964    84919
full_pipeline       1.000    0.857    0.972    1.000    0.514    0.972    0.964    139388
```

## 测试

### Worker 网关测试（vitest，workerd 运行时）

```powershell
npm test
```

### Python 检索引擎测试（pytest）

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

### 全部测试

```powershell
npm test
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

测试覆盖：

| 文件 | 覆盖内容 |
|---|---|
| test/worker.test.js | Worker 网关 9 项：公开 health、鉴权 401、Bearer 代理 + origin key 替换、405 + Allow、415 非 JSON、413 超限、上游 502 脱敏、成功响应 + 安全头、X-Request-ID 替换 |
| test_loader_metadata.py | 分片元数据完整性（chunk_id、char_start/end、content_hash） |
| test_generator_fallback.py | 抽取式降级回答生成 |
| test_chain_context_expansion.py | 代码查询邻居扩展不跨 source 边界 |
| test_hybrid_rrf.py | RRF 公式、去重、重叠提权、向量降级、embedding 指纹 |
| test_hyde.py | HyDE 生成、异常降级、空响应降级、vector_query 路由 |
| test_metrics.py | Recall@K、Precision@K、MRR、NDCG@K、多级相关性、聚合 |
| test_batched.py | 惰性分批生成器、批次边界、无效 batch_size、validate_batch_size |
| test_rebuild_vectorstore.py | argparse 默认值/override、模式判定、增量/全量 sync 参数 wiring |

当前 Worker 测试 9 项，Python 测试 71 项，合计 80 项。

## 注意事项

- 首次加载 bge-m3 / bge-reranker-v2-m3 需要下载约 4.4GB 模型。
- 默认 Hugging Face endpoint 设置为 `https://hf-mirror.com`，便于国内环境下载。
- 模型下载完成后建议设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，避免每次启动向 huggingface.co 发 HEAD 请求。
- 切换 embedding 模型后，向量库会通过 `.embedding_meta.json` 指纹自动检测并重建，无需手动 `--rebuild`。
- 不要提交 API key、本地 PDF、生成的向量库或 `.venv`。
- `.rag_cache/` 是本地生成的 chunk 缓存，也不需要提交。
- 日常问答不要加 `--rebuild`，否则会重新计算所有 chunk 的 embedding，耗时较长。
- bge-m3 在 CPU 上 embed 大量 chunk 耗时较长（具体取决于 chunk 数量、CPU 核数和语料规模，需在目标机器上 benchmark），建议首次重建时耐心等待或使用 GPU。
- bge-reranker-v2-m3 在 CPU 上单次精排有一定延迟（取决于候选数量和 CPU，需在目标机器上 benchmark），对延迟敏感的场景可考虑 ONNX 量化或缓存热门 query。
- 客户端应始终访问 `smillick.org/api/v1/rag/*`，`rag.smillick.org` 仅供 Worker 回源，不应直连。
