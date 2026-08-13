# RAG 项目面试深挖稿

## 1. 项目一句话介绍

我做的是一个面向本地 PDF 文档的 RAG 问答系统。系统会把 `data/` 下的 PDF 按子目录划分成 collection，例如 `algorithm`、`netsec_lab`、`tcpip`、`pcb`，解析后按页切分 chunk，并保留 `source_path`、`collection`、`page`、`chunk_id`、`char_start`、`char_end` 等元数据。查询时先按 collection 限定检索域，再融合 Chroma 向量检索和 BM25 关键词检索，用 CrossEncoder 加词面加权精排，最后把片段交给大模型生成带引用答案，并记录 JSONL 调用链日志。

## 2. RAG 是什么，为什么不用直接把文档塞给模型

RAG 是 Retrieval-Augmented Generation，核心是先检索外部知识，再让大模型基于检索结果回答。它解决三个问题：

- 模型参数知识不包含本地 PDF。
- 长文档超出上下文窗口，不能一次性塞给模型。
- 回答需要可追溯，需要知道答案来自哪个文件、哪一页、哪个 chunk。

我的项目中，PDF 被切成 chunk 后存入向量库，同时保留页码和字符偏移。回答时模型只能基于检索片段生成，并输出引用。这样比直接上传整本书更可控，也便于调试和评测。

## 3. 为什么要用向量库

向量库解决的是“语义相似检索”。用户提问和原文不一定字面一致，例如用户问“TCP 建连流程”，原文可能写“三次握手、connect 处理过程、SYN/ACK”。BM25 只看关键词时可能漏召回，而 embedding 可以把语义接近的内容拉近。

我用 Chroma 的原因是它适合本地持久化、部署简单、和 LangChain/HuggingFace embedding 集成方便。对简历项目来说，Chroma 可以验证完整 RAG 链路；如果到百万级 chunk 或多用户并发，我会迁移到 Qdrant、Milvus 或 Elasticsearch/OpenSearch kNN。

面试回答可以说：

> 向量库不是为了替代数据库，而是为了支持语义召回。普通数据库擅长精确字段查询，BM25 擅长关键词匹配，向量库擅长语义相似。我的项目把 Chroma 用作本地向量索引，结合 BM25 提升中文和中英混合文档的召回稳定性。

## 4. 为什么还要 BM25

纯向量检索对技术文档不够稳，尤其是这些内容：

- 函数名：`sniff()`、`sendp()`。
- 协议名：TCP、ARP、HTTP。
- 算法名：Dijkstra、Floyd。
- 文件名、实验编号、错误码。

这些词具有强字面匹配特征，BM25 更可靠。所以我的检索是 hybrid retrieval：Chroma 召回语义相关片段，BM25 召回关键词精确命中的片段，然后合并去重，再交给 reranker。

## 5. Rerank 怎么实现

项目里 rerank 分两层：

1. CrossEncoder 对 `(query, chunk)` 成对打分。
2. 再加一层词面匹配加权，避免中文技术词、英文缩写、函数名被英文 reranker 压下去。

代码层面是：

- `src/retriever.py` 负责 Chroma + BM25 混合召回。
- `src/reranker.py` 负责 CrossEncoder + lexical bonus 精排。
- `src/chain.py` 负责编排 retrieve -> rerank -> generate。

为什么要 rerank：

> 向量检索和 BM25 是召回阶段，目标是尽量别漏；rerank 是精排阶段，目标是把最能回答问题的片段排到前面。CrossEncoder 比 embedding 相似度更准，因为它能同时看 query 和 passage 的交互关系，但计算更慢，所以只对 top candidates 做精排。

如果被问“为什么加词面加权”：

> 我自测时发现，大混合库里问 Dijkstra，英文 MS MARCO CrossEncoder 有时会把中文技术片段排偏。后来我加了 query term overlap bonus，对 Dijkstra、ARP、TCP、sniff 这类关键术语命中的片段加权，top1 就能稳定回到原文。这是一次根据调试结果做的工程修正。

## 6. Chunk 怎么切，怎么调参

切 chunk 的目标是平衡三件事：

- 太小：上下文不完整，代码块和解释容易被切断。
- 太大：检索粒度粗，噪声多，占用上下文窗口。
- overlap 太小：跨段信息断裂。
- overlap 太大：重复内容多，索引变大。

项目早期默认：

```text
chunk_size = 512
chunk_overlap = 64
```

后来面对大 PDF，我改成建议：

```text
chunk_size = 900
chunk_overlap = 120
vectorstore_batch_size = 128
```

原因是百兆级文档如果 chunk 太小，chunk 数会暴涨，embedding 和 Chroma 写入都变慢。增大 chunk 可以减少索引规模，同时保留足够上下文。

面试回答：

> 我不是固定迷信某个 chunk_size，而是根据文档类型调。技术教材和 PDF 书籍段落较长，适合 800 到 1000 字左右；实验指导书和代码解释需要适当 overlap，避免步骤或代码被切断。我还针对代码类问题扩大检索窗口，引入相邻 chunk，降低代码块断裂的影响。

## 7. 项目怎么做调试

我主要看四类信息：

- `sources`：答案来自哪个 source/page/chunk。
- `trace.rerank_scores`：精排分数和排序是否合理。
- `trace.timing_ms`：检索、精排、生成耗时。
- `logs/YYYY-MM-DD.jsonl`：每轮 query、collection、top-k、引用、生成模式。

实际调试例子：

- TCP 三次握手：命中 `tcpip` collection 下的 TCPIP 详解，相关。
- ARP 缓存中毒实验目的：命中 `netsec_lab` 下实验指导书第 2 页，相关。
- Scapy sniff：命中实验 PPT 和 Scapy 用法文档，相关。
- Dijkstra：初始大混合库中被 rerank 带偏，后来通过 collection 过滤和词面加权修正。

这个例子可以重点讲，因为它体现了你真的调过系统，而不是只调包。

## 8. 怎么减少幻觉

减少幻觉不是只靠 prompt，而是多层控制：

- collection 过滤，先避免拿错领域文档。
- hybrid retrieval，提高召回质量。
- rerank，把真正相关的片段排前面。
- prompt 限制模型只能基于检索片段回答。
- citations 返回 source/page/chunk_id。
- 当没有相关片段或生成 API 失败时，不硬编；本地 fallback 只做抽取式回答。
- JSONL 日志保留每次调用链，便于回放和定位问题。

面试回答：

> RAG 的幻觉很多时候不是模型凭空产生，而是检索阶段给错了上下文。所以我重点做了检索域隔离、混合召回、精排和引用溯源。prompt 是最后一道约束，不是唯一手段。

## 9. 大文本能不能承载

现在测试过约 45MB PDF，切成 3546 个 chunk，重建 Chroma 约 90 秒。为适配大文本，我做了两点优化：

- Chroma 分批写入，避免一次性 add 全部 chunk。
- BM25 懒加载，按 collection 第一次查询时才建索引。

边界回答：

> 百兆级文本可以作为本地 RAG 项目承载，但索引应提前离线完成，不能现场第一次启动才建库。如果到百万级 chunk 或多用户并发，就应该把向量检索迁到 Qdrant/Milvus，把关键词检索迁到 Elasticsearch/OpenSearch。

## 10. Agent、Skill、MCP 怎么和 RAG 结合

### Agent

Agent 是“能规划和调用工具的 LLM 应用”。普通 RAG 是 query -> retrieve -> answer；Agentic RAG 则可以先判断问题类型，再选择工具，例如：

- 查 PDF：调用 RAG retriever。
- 查日志：调用 query log reader。
- 需要计算：调用 Python 工具。
- 需要外部资料：调用 Web/API。
- 多轮任务：拆解成多个子问题检索后汇总。

面试说法：

> 我的项目目前是标准 RAG pipeline，还不是完整 Agent。但它已经有 Agent 化的基础：collection 是工具选择前的路由信息，trace/log 是可观测性，后续可以让 Agent 根据问题自动选择 collection、决定是否扩大 top_k、是否调用评测脚本。

### Skill

Skill 可以理解为封装好的领域能力或操作流程。放到 RAG 中，skill 可以是：

- PDF ingestion skill：解析、切分、入库。
- retrieval evaluation skill：跑 JSONL 测评。
- answer grounding skill：检查答案引用是否存在。
- code explanation skill：检测代码问题并扩大相邻 chunk。

面试说法：

> Skill 更像可复用能力模块。我的项目里代码类问题扩大窗口、JSONL 评测、collection 过滤，都可以抽象成 RAG skills，由 Agent 在不同任务中调用。

### MCP

MCP 是 Model Context Protocol，用来标准化模型和外部工具/数据源之间的连接。和 RAG 结合时，可以把本地 PDF RAG 服务暴露成 MCP tool：

- `list_collections`
- `retrieve_chunks`
- `ask_pdf`
- `evaluate_retrieval`
- `read_query_logs`

这样 IDE、Agent 或其他模型客户端都可以通过统一协议调用本地知识库。

面试说法：

> MCP 的价值是把工具调用标准化。我的 RAG 系统已经有 FastAPI 接口，后续可以包装成 MCP server，把 collection 查询、检索、问答和日志读取暴露成工具，让 Agent 不直接耦合具体 HTTP API。

## 11. 常见追问短答

**问：为什么不用纯向量？**  
答：技术文档里函数名、协议名、实验编号很重要，BM25 对精确词更稳，所以我用向量 + BM25 混合召回。

**问：为什么不用纯 BM25？**  
答：用户表达和原文不一定一致，向量检索能补语义相似召回。

**问：rerank 慢怎么办？**  
答：只对召回后的 top 50 或 top 100 做 rerank，不对全库做；高并发可以换轻量 reranker 或缓存热门 query。

**问：chunk_size 怎么确定？**  
答：按文档类型调。短 FAQ 可以小，教材/电子书可以大；我用日志和检索命中情况看是否切断上下文，再调 chunk_size 和 overlap。

**问：如何评估 RAG？**  
答：检索评估看 Recall@K、MRR、source_hit、keyword_hit；生成评估看答案是否引用正确、是否基于原文、是否拒答无依据问题。

**问：你的项目最大亮点？**  
答：不是单纯调包，而是做了 collection 隔离、可追溯元数据、混合检索、rerank 加权、调用链日志和大文本分批索引。

**问：最大不足？**  
答：当前还是本地单机 RAG，Chroma 和内存 BM25 适合个人知识库；企业级需要服务化向量库、ES/OpenSearch、权限控制、并发和更完整评测。

## 12. 面试收束表达

可以用这段总结：

> 这个项目我主要关注三个点：第一是可追溯，所有 chunk 都保留 source、page、chunk_id 和字符偏移；第二是检索稳定性，通过 collection 过滤、Chroma 向量检索、BM25 和 CrossEncoder 精排降低误召回；第三是工程可观测性，每次问答都有 trace 和 JSONL 日志。我也做过实际调试，比如 Dijkstra 在大混合库里被精排带偏，后来通过 collection 隔离和词面加权修正。这个项目目前适合本地 PDF 知识库，后续如果要企业级扩展，会把向量库和关键词检索迁到专门服务。
