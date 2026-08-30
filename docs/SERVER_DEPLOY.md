# TraceRAG 公网部署指南（Cloudflare Tunnel + Windows 服务）

本文档说明把 TraceRAG 从本机开发模式升级到「公网跨网 + 常驻后台」的完整步骤。

## 架构

```text
TaffySearchTool 客户端（任意网络）
   │
   │  HTTPS（API Key 通过 Authorization 头传输）
   ▼
Cloudflare Edge（自动 TLS / DDoS 防护 / 全球加速）
   │
   │  出站加密隧道（QUIC 协议，无需开公网端口）
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

- 无需开防火墙入站端口，cloudflared 通过出站连接到 CF 边缘
- 无需公网 IP / 域名证书，CF 自动处理 TLS
- 应用层 API Key 中间件防止隧道被滥用
- uvicorn 只监听 `127.0.0.1`，本机其他进程也无法绕过认证直接访问

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

### 2. 模型下载（首次，约 4.4 GB）

```powershell
.\.venv\Scripts\python.exe scripts\download_models.py
```

### 3. 配置 .env

```powershell
Copy-Item .env.example .env
notepad .env
```

必填项：

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `RAG_API_KEY` | **生成方法**：`python -c "import secrets; print(secrets.token_urlsafe(32))"`，客户端需要带这个 key |
| `RAG_HOST` | 保持 `127.0.0.1`，由 cloudflared 反代 |
| `RAG_PORT` | 默认 `8000` |

### 4. 准备语料

把 PDF 放到 `data/` 子目录（参见 README）。

---

## 步骤 1：本地启动验证

先确保 uvicorn 能正常起，再走隧道。

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

# 带 key 调用
$env:RAG_API_KEY = "你的 key"
curl http://127.0.0.1:8000/ask `
    -Method POST `
    -Headers @{Authorization = "Bearer $env:RAG_API_KEY"} `
    -ContentType "application/json" `
    -Body '{"query":"Dijkstra算法适合解决什么问题？","collection":"algorithm"}'
```

验证通过后 `Ctrl+C` 停止。

---

## 步骤 2：快速隧道联调（一次性测试，无需域名）

开两个 PowerShell 窗口：

**窗口 A**：启动 uvicorn

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_server.ps1
```

**窗口 B**：启动快速隧道

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_tunnel_quick.ps1
```

输出会包含一行：

```
https://<random>.trycloudflare.com
```

从任意设备（手机、其他电脑）用这个 URL 测试：

```powershell
$base = "https://<random>.trycloudflare.com"
curl "$base/health"
curl "$base/ask" `
    -Method POST `
    -Headers @{Authorization = "Bearer $env:RAG_API_KEY"} `
    -ContentType "application/json" `
    -Body '{"query":"test"}'
```

**说明**：快速隧道生成的 URL 当天有效，重启 cloudflared 后失效。仅用于联调，不可用于生产。

---

## 步骤 3：命名隧道 + 自定义域名（生产）

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_cloudflared.ps1 -Domain "rag.yourdomain.com"
```

脚本会引导你：

1. 下载 `cloudflared.exe` 到 `.bin/`
2. 浏览器登录 Cloudflare 选域名（生成 `cert.pem`）
3. 创建命名隧道 `tracerag`（生成 credentials JSON）
4. 绑定 DNS：`rag.yourdomain.com` → 隧道
5. 生成 `cloudflared/config.yml`

完成后**手动测试隧道**（让 cloudflared 读 config.yml）：

```powershell
.\.bin\cloudflared.exe tunnel run tracerag --config .\cloudflared\config.yml
```

从任意设备验证 `https://rag.yourdomain.com/health`。

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

## 客户端配置

在 TaffySearchTool 设置页（或配置文件）填入：

| 字段 | 值 |
|---|---|
| RAG 服务地址 | `https://rag.yourdomain.com` |
| RAG API Key | 与服务端 `RAG_API_KEY` 一致 |
| 超时 | 60 秒（推荐 120 秒，bge-reranker CPU 精排约 6 秒） |

客户端调用示例：

```python
import urllib.request, json

req = urllib.request.Request(
    "https://rag.yourdomain.com/ask",
    data=json.dumps({"query": "塔菲在哪天玩了空洞骑士", "collection": "taffy_replays"}).encode(),
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

### 服务启动失败

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

### 401 / 403

- 401 `Missing Authorization header`：客户端没带 `Authorization: Bearer <key>` 头
- 403 `Invalid API key`：客户端 key 与服务端 `.env` 中的 `RAG_API_KEY` 不一致

### 隧道连不上

```powershell
# 看 cloudflared 日志
Get-EventLog -LogName Application -Source "cloudflared" -Newest 50 | Format-List

# 重启
Restart-Service cloudflared
```

---

## 安全建议

1. **API Key 不要硬编码到客户端代码或 git 仓库**。TaffySearchTool 的 `settings.ini` 已被 `.gitignore` 忽略。
2. **`.env` 不要提交 git**。已加 `.gitignore`。
3. **定期轮换 API Key**：改 `.env` 后 `Restart-Service TraceRAG`。
4. **cloudflared 配置文件路径白名单**：`config.yml` 的 `path` 字段限制只暴露 `/health`、`/ask`、`/collections`、`/stats`、`/ingest`，其他路径在 CF 边缘就 404。
5. **如果担心 CF 隧道被滥用**，可在 Cloudflare Zero Trust 控制台启用 Access Policy，加一层邮件 OTP 验证。

---

## 卸载

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_services.ps1 -Uninstall
```

会停止并删除 `TraceRAG` 和 `cloudflared` 两个服务。`.bin/`、`logs/`、`cloudflared/` 目录保留。

---

## 文件清单

本次部署新增/修改的文件：

```
rag_project/
├─ .env.example                       完整配置模板
├─ server.py                          加 API Key 中间件
├─ cloudflared/
│  └─ config.yml.example              隧道路由模板
└─ scripts/
   ├─ start_server.ps1                手动启动 uvicorn
   ├─ setup_cloudflared.ps1           命名隧道引导
   ├─ start_tunnel_quick.ps1          快速隧道测试
   └─ install_services.ps1            NSSM 服务化
```

运行时生成的目录：

```
rag_project/
├─ .bin/                              cloudflared.exe + nssm.exe
├─ logs/
│  ├─ nssm_stdout.log                 TraceRAG 服务 stdout
│  ├─ nssm_stderr.log                 TraceRAG 服务 stderr
│  └─ YYYY-MM-DD.jsonl                RAG 调用链日志（原有）
└─ cloudflared/
   ├─ cert.pem                        Cloudflare 账户证书
   ├─ <uuid>.json                     隧道 credentials
   └─ config.yml                      实际使用的隧道配置
```
