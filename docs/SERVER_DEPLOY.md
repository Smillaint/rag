# TraceRAG 公网部署指南（Cloudflare Worker 网关 + Tunnel + Windows 服务）

本文档说明把 TraceRAG 从本机开发模式升级到「公网跨网 + 常驻后台」的完整步骤。生产主路径已切换为 Cloudflare Worker 网关，Tunnel 和 Windows 服务化作为源站运维层保留。

## 架构

```text
客户端（任意网络）
   │
   │  HTTPS  https://smillick.org/api/v1/rag/*
   │  Authorization: Bearer $RAG_GATEWAY_API_KEY
   ▼
Cloudflare Edge → Worker smillick-org
   │  公开 health / Bearer 鉴权 / 两级 Rate Limit / 安全头 / 上游错误脱敏
   │  以独立 origin key 回源，不转发客户端凭证
   ▼
https://rag.smillick.org  （源站，客户端不应直连）
   │
   │  Cloudflare Tunnel（cloudflared，出站 QUIC，无需开公网端口）
   ▼
cloudflared.exe（本机 Windows，NSSM 守护）
   │
   │  HTTP 127.0.0.1:8000（仅本地回环）
   ▼
uvicorn + FastAPI（应用层 API Key 中间件）
   │
   ▼
TraceRAG（bge-m3 + bge-reranker + Chroma + BM25 + DeepSeek）
```

**关键安全特性**：

- 客户端只访问 `smillick.org`，Worker 处理鉴权、限流、安全头和错误脱敏。
- `rag.smillick.org` 是源站地址，仅供 Worker 回源，普通客户端不应直连。
- Worker 丢弃客户端凭证，以独立 `RAG_ORIGIN_API_KEY` 回源。
- 无需开防火墙入站端口，cloudflared 通过出站连接到 CF 边缘。
- uvicorn 只监听 `127.0.0.1`，本机其他进程也无法绕过认证直接访问。

---

## 前置准备

### 1. Python 环境与依赖

```powershell
cd C:\Users\31854\rag_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Node.js 与 Worker 依赖

```powershell
npm install
```

### 3. 模型下载（首次，约 4.4 GB）

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
```

### 4. 配置 .env（源站侧密钥）

```powershell
Copy-Item .env.example .env
notepad .env
```

必填项：

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `RAG_API_KEY` | FastAPI 源站应用层鉴权密钥。生成方法：`python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `RAG_HOST` | 保持 `127.0.0.1`，由 cloudflared 反代 |
| `RAG_PORT` | 默认 `8000` |

### 5. 准备语料

把 PDF 放到 `data/` 子目录（参见 README）。

---

## 步骤 1：本地启动验证

先确保 uvicorn 能正常起，再走网关和隧道。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_server.ps1
```

看到 `Uvicorn running on http://127.0.0.1:8000` 即成功。模型首次加载需要 1-2 分钟。

**新开一个 PowerShell 窗口验证**：

```powershell
# 健康检查（不需要 API Key）
curl http://127.0.0.1:8000/health
# 预期：{"status":"ok","ready":true}

# 无 key 调用 /ask，应该返回 401
curl http://127.0.0.1:8000/ask -Method POST -ContentType "application/json" -Body '{"query":"test"}'
# 预期：{"detail":"Missing or malformed Authorization header..."}
```

验证通过后 `Ctrl+C` 停止。

---

## 步骤 2：部署 Cloudflare Worker 网关

### 2.1 配置 Worker 密钥

Worker 需要两个密钥，通过 `wrangler secret put` 注入（不写入 `wrangler.jsonc`）：

```powershell
# 客户端鉴权密钥（客户端携带此 key 访问 smillick.org）
npx wrangler secret put RAG_GATEWAY_API_KEY

# Worker 回源时使用的独立 origin key
npx wrangler secret put RAG_ORIGIN_API_KEY
```

每个命令会交互式提示输入值。密钥生成方法：

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

> **密钥关系**：
> - `RAG_ORIGIN_API_KEY`（Worker secret）**必须**与源站 `.env` 中的 `RAG_API_KEY` 完全相同——Worker 用它重建 `Authorization` 头回源，FastAPI 用 `RAG_API_KEY` 校验。两者不一致则回源 401。
> - `RAG_GATEWAY_API_KEY`（Worker secret）是客户端鉴权密钥，应与 `RAG_ORIGIN_API_KEY` 分离。
> - 快速上线时三个 key 可以先用同一个值，但生产环境建议 `RAG_GATEWAY_API_KEY` 与 `RAG_ORIGIN_API_KEY` 分离并独立轮换；`RAG_ORIGIN_API_KEY` 与 `RAG_API_KEY` 始终成对相同，修改时需同步更新两端。

`wrangler.jsonc` 中 `vars.RAG_ORIGIN_URL` 已固定为 `https://rag.smillick.org`，Rate Limit 绑定（`RAG_API_RATE_LIMITER` 120/60s、`RAG_ASK_RATE_LIMITER` 10/60s）已在配置中声明，无需手动设置。

### 2.2 验证配置

```powershell
npm run check
```

输出应显示所有绑定（ASSETS、两个 Rate Limit、RAG_ORIGIN_URL），无 unsafe 警告，`--dry-run` 退出。

### 2.3 部署到生产

```powershell
npm run deploy
```

部署后 Worker 接管 `smillick.org/*`，`/api/v1/rag/*` 路由由 Worker 处理，其余请求回退到静态资源。

### 2.4 验证线上网关

```powershell
$base = "https://smillick.org/api/v1/rag"
$key = "<你的 RAG_GATEWAY_API_KEY>"

# 公开健康检查
curl "$base/health"

# 鉴权接口（无 key 应返回 401）
curl "$base/collections"
# 预期：{"error":"unauthorized","request_id":"..."}

# 带 key 调用
curl "$base/collections" -Headers @{Authorization = "Bearer $key"}
```

---

## 步骤 3：Cloudflare Tunnel 配置（源站接入）

Worker 通过 `https://rag.smillick.org` 回源，该地址由 Cloudflare Tunnel 转发到本机 `127.0.0.1:8000`。

### 3.1 快速隧道联调（一次性测试，无需域名）

开两个 PowerShell 窗口：

**窗口 A**：启动 uvicorn

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_server.ps1
```

**窗口 B**：启动快速隧道

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_tunnel_quick.ps1
```

输出会包含一行 `https://<random>.trycloudflare.com`，当天有效，仅用于联调。

### 3.2 命名隧道 + 自定义域名（生产）

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_cloudflared.ps1 -Domain "rag.smillick.org"
```

脚本会引导你：

1. 下载 `cloudflared.exe` 到 `.bin/`
2. 浏览器登录 Cloudflare 选域名（生成 `cert.pem`）
3. 创建命名隧道 `tracerag`（生成 credentials JSON）
4. 绑定 DNS：`rag.smillick.org` → 隧道
5. 生成 `cloudflared/config.yml`

完成后手动测试隧道：

```powershell
.\.bin\cloudflared.exe tunnel run tracerag --config .\cloudflared\config.yml
```

从任意设备验证 `https://rag.smillick.org/health`（此地址仅供 Worker 回源，客户端应通过 `smillick.org` 访问）。

---

## 步骤 4：服务化（NSSM 守护 + 开机自启）

```powershell
# 确保步骤 1-3 全部跑通
powershell -ExecutionPolicy Bypass -File .\scripts\install_services.ps1
```

脚本会：

1. 下载 `nssm.exe` 到 `.bin/`
2. 注册 `TraceRAG` 服务（调用 `start_server.ps1`，自动加载 `.env`）
3. 注册 `cloudflared` 服务（用 cloudflared 自带的 `service install`）
4. 启动两个服务
5. 健康检查 `http://127.0.0.1:8000/health`

完成后：

```powershell
Get-Service TraceRAG, cloudflared
# Status   Name               DisplayName
# ------   ----               -----------
# Running  TraceRAG           TraceRAG Server
# Running  cloudflared        Cloudflare Tunnel
```

**管理命令**：

```powershell
Stop-Service  TraceRAG
Start-Service TraceRAG
Restart-Service TraceRAG
```

**查看日志**：

```powershell
# TraceRAG 服务日志
Get-Content .\logs\nssm_stderr.log -Tail 50 -Wait

# cloudflared 日志（Windows 事件查看器 → Windows 日志 → 应用程序 → 来源 cloudflared）
Get-EventLog -LogName Application -Source "cloudflared" -Newest 20
```

**修改 .env 后**：

```powershell
Restart-Service TraceRAG
```

---

## 步骤 5：验证完整链路

```powershell
$base = "https://smillick.org/api/v1/rag"
$key = "<你的 RAG_GATEWAY_API_KEY>"

# 1. 公开健康检查（Worker → Tunnel → FastAPI）
curl "$base/health"

# 2. 鉴权接口
curl "$base/collections" -Headers @{Authorization = "Bearer $key"}

# 3. 问答
curl "$base/ask" `
    -Method POST `
    -Headers @{Authorization = "Bearer $key"} `
    -ContentType "application/json" `
    -Body '{"query":"Dijkstra算法适合解决什么问题？","collection":"algorithm"}'
```

---

## 客户端配置

在客户端应用中填入：

| 字段 | 值 |
|---|---|
| RAG 服务地址 | `https://smillick.org/api/v1/rag` |
| RAG API Key | 与 Worker `RAG_GATEWAY_API_KEY` 一致 |
| 超时 | 60 秒（推荐 120 秒，bge-reranker CPU 精排有一定延迟，需在目标机器上 benchmark） |

客户端调用示例：

```python
import urllib.request, json

req = urllib.request.Request(
    "https://smillick.org/api/v1/rag/ask",
    data=json.dumps({"query": "Dijkstra算法适合解决什么问题？", "collection": "algorithm"}).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read())
print(result["answer"])
print(result["citations"])
```

---

## 防火墙

**不需要任何入站规则**。cloudflared 是出站连接，不需要在 Windows 防火墙开放任何端口。

可选的出站限制（增强安全，让 cloudflared 只能连 Cloudflare IP）：

```powershell
# Cloudflare 官方 IP 段见 https://www.cloudflare.com/ips/
# 这里只放行 cloudflared.exe，不限制目标 IP
New-NetFirewallRule -DisplayName "Allow cloudflared outbound" `
    -Direction Outbound `
    -Program "C:\Users\31854\rag_project\.bin\cloudflared.exe" `
    -Action Allow
```

---

## 故障排查

### Worker 部署问题

```powershell
# 验证配置和绑定
npm run check

# 查看 Worker 实时日志
npx wrangler tail
```

### 源站服务启动失败

```powershell
# 查看错误日志
Get-Content .\logs\nssm_stderr.log -Tail 100
Get-Content .\logs\nssm_stdout.log -Tail 100
```

### 模型未下载

```
RuntimeError: bge-m3 model files not found
```

→ 执行 `scripts\download_models.py`。

### `cloudflared` 服务找不到 config.yml

`cloudflared` 服务以 SYSTEM 身份运行，会查找 `C:\Windows\System32\config\systemprofile\.cloudflared\config.yml`。`install_services.ps1` 已经自动复制，如果手动改了 `cloudflared\config.yml`，需要：

```powershell
$systemDir = "$env:SystemRoot\System32\config\systemprofile\.cloudflared"
Copy-Item .\cloudflared\config.yml $systemDir -Force
Restart-Service cloudflared
```

### 503 Service Unavailable

uvicorn 还在加载模型（首次启动 1-2 分钟）。等 30 秒后重试 `/health`。

### 401 / 403 / 429

- 401 `unauthorized`：客户端没带 `Authorization: Bearer <key>` 头，或 key 与 Worker `RAG_GATEWAY_API_KEY` 不一致。
- 429 `rate_limited`：超过 Rate Limit（API 120/60s，ask 10/60s），等待 `Retry-After` 秒后重试。
- 503 `rate_limiter_unavailable`：Rate Limit 绑定缺失或异常，检查 `wrangler.jsonc` 中的 `ratelimits` 配置。

### 502 upstream_error / upstream_unavailable

Worker 已收到请求但回源失败。检查 Tunnel 服务是否运行、FastAPI 是否监听 `127.0.0.1:8000`。

```powershell
Restart-Service cloudflared
Restart-Service TraceRAG
```

### 隧道连不上

```powershell
# 看 cloudflared 日志
Get-EventLog -LogName Application -Source "cloudflared" -Newest 50 | Format-List

# 重启
Restart-Service cloudflared
```

---

## 安全建议

1. **密钥不要硬编码到客户端代码或 git 仓库**。`.env` 已被 `.gitignore` 忽略；Worker 密钥通过 `wrangler secret put` 注入，不落盘到配置文件。
2. **密钥分离与轮换**：`RAG_ORIGIN_API_KEY`（Worker secret）必须与源站 `.env` 的 `RAG_API_KEY` 完全相同，否则回源 401；两者始终成对更新。`RAG_GATEWAY_API_KEY` 应与 `RAG_ORIGIN_API_KEY` 分离并独立轮换。快速上线时三个 key 可以相同，但生产环境建议 gateway 和 origin 分离。
3. **定期轮换密钥**：改 Worker secret 后无需重新部署（下次请求生效）；改 `.env` 后 `Restart-Service TraceRAG`。
4. **cloudflared 配置文件路径白名单**：`config.yml` 的 `path` 字段限制只暴露必要路径。
5. **如果担心 CF 隧道被滥用**，可在 Cloudflare Zero Trust 控制台启用 Access Policy，加一层邮件 OTP 验证。
6. **`rag.smillick.org` 不应暴露给客户端**：它是 Worker 回源专用地址。即使被直连，FastAPI 的 `RAG_API_KEY` 中间件仍会鉴权，但客户端将失去网关层 Bearer 鉴权、Rate Limit、统一错误脱敏、安全头和 request-id 追踪。客户端应始终通过 `smillick.org/api/v1/rag/*` 访问。

---

## 卸载

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_services.ps1 -Uninstall
```

会停止并删除 `TraceRAG` 和 `cloudflared` 两个服务。`.bin/`、`logs/`、`cloudflared/` 目录保留。

Worker 部署可通过 `npx wrangler delete` 删除（会同时删除路由和 Rate Limit 绑定）。

---

## 文件清单

```
rag_project/
├─ .env.example                       源站配置模板
├─ .env                               实际配置（gitignore）
├─ wrangler.jsonc                     Worker 配置（路由、绑定、Rate Limit）
├─ worker/
│  └─ index.js                        Cloudflare Worker 网关
├─ server.py                          FastAPI 源站入口
├─ cloudflared/
│  └─ config.yml.example              隧道路由模板
├─ scripts/
│  ├─ start_server.ps1                手动启动 uvicorn
│  ├─ rebuild_vectorstore.py          向量库重建/增量同步
│  ├─ setup_cloudflared.ps1           命名隧道引导
│  ├─ start_tunnel_quick.ps1          快速隧道测试
│  └─ install_services.ps1            NSSM 服务化
```

运行时生成的目录：

```
rag_project/
├─ .bin/                              cloudflared.exe + nssm.exe
├─ .wrangler/                         Wrangler 本地状态（gitignore）
├─ node_modules/                      Worker 依赖（gitignore）
├─ logs/
│  ├─ nssm_stdout.log                 TraceRAG 服务 stdout
│  ├─ nssm_stderr.log                 TraceRAG 服务 stderr
│  └─ YYYY-MM-DD.jsonl                RAG 调用链日志
└─ cloudflared/
   ├─ cert.pem                        Cloudflare 账户证书
   ├─ <uuid>.json                     隧道 credentials
   └─ config.yml                      实际使用的隧道配置
```
