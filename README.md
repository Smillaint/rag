# rag

一个本地 PDF RAG 问答示例。流程包括 PDF 读取、文本切块、Chroma 向量检索、BM25 关键词检索、CrossEncoder 精排，以及 DeepSeek/OpenAI 兼容接口生成答案。

## 安装

建议先创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 本地数据

把 PDF 文件放到本地 `data/` 目录。`data/` 已经加入 `.gitignore`，不会提交到 GitHub。

```text
data/
  your-document.pdf
```

## API Key 配置

不要把 API key、token、secret 等敏感信息提交到 Git。仓库已经通过 `.gitignore` 忽略 `.env`、`key.txt`、`*.key` 等本地密钥文件。

运行前在本地环境变量中配置 API key：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key_here"
```

如果 API key 已经上传到 GitHub，请立即在对应平台作废/轮换这个 key。`.gitignore` 只能防止后续继续提交，不能从 GitHub 历史记录中移除已经泄露的密钥。

## 运行

首次运行或 PDF 变化后，重建向量库：

```powershell
python main.py --rebuild "你的问题"
```

后续可以直接复用本地 `vectorstore/`：

```powershell
python main.py "联邦学习如何保护数据隐私？"
```

不传问题会进入交互模式：

```powershell
python main.py
```

可选参数：

```powershell
python main.py --data-dir ./data --vectorstore-dir ./vectorstore --model deepseek-chat "你的问题"
```

## 目录结构

```text
src/
  loader.py       # PDF 按页读取，切分 chunk，并保留 source/page/chunk_id
  retriever.py    # Chroma 向量库、BM25Index、混合检索
  reranker.py     # CrossEncoder 精排
  generator.py    # OpenAI 兼容接口生成答案
  chain.py        # 端到端 RAGPipeline
main.py           # 命令行入口
test_*.py         # 手动验证脚本
```

## 注意

- `vectorstore/` 是本地生成的向量库，已被 Git 忽略。
- `data/` 是本地文档目录，已被 Git 忽略。
- 首次下载 embedding/reranker 模型需要联网；代码默认设置了 `HF_ENDPOINT=https://hf-mirror.com`。
