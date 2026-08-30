# =============================================================================
# TraceRAG 服务启动脚本
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\scripts\start_server.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\start_server.ps1 -Dev
#   powershell -ExecutionPolicy Bypass -File .\scripts\start_server.ps1 -Workers 2
#
# - 默认生产模式：监听 RAG_HOST:RAG_PORT（默认 127.0.0.1:8000），不启用 reload
# - -Dev：启用 reload，单 worker，适合本地调试
# - -Workers N：生产模式下指定 worker 数（注意：bge-m3 模型占内存大，
#   多 worker 会复制模型实例，建议保持默认 1）
# =============================================================================
param(
    [switch]$Dev,
    [ValidateRange(1, 8)]
    [int]$Workers = 1
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

# -----------------------------------------------------------------------------
# 加载 .env 到当前进程环境变量
# -----------------------------------------------------------------------------
function Invoke-LoadEnv {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Warning ".env not found at $Path; relying on shell environment."
        return
    }
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#") -or $trimmed.StartsWith("#")) {
            continue
        }
        if ($trimmed -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') {
            continue
        }
        $key = $Matches[1]
        $value = $Matches[2].Trim().Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

Invoke-LoadEnv -Path (Join-Path $ProjectRoot ".env")

# -----------------------------------------------------------------------------
# 强制模型离线模式（避免每次启动向 huggingface.co 发 HEAD 请求）
# -----------------------------------------------------------------------------
[Environment]::SetEnvironmentVariable("HF_HUB_OFFLINE", "1", "Process")
[Environment]::SetEnvironmentVariable("TRANSFORMERS_OFFLINE", "1", "Process")
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "Process")

# -----------------------------------------------------------------------------
# 解析监听参数
# -----------------------------------------------------------------------------
$ListenHost = [Environment]::GetEnvironmentVariable("RAG_HOST")
if ([string]::IsNullOrWhiteSpace($ListenHost)) {
    $ListenHost = "127.0.0.1"
}

$ListenPort = [Environment]::GetEnvironmentVariable("RAG_PORT")
if ([string]::IsNullOrWhiteSpace($ListenPort)) {
    $ListenPort = "8000"
}

$ApiKey = [Environment]::GetEnvironmentVariable("RAG_API_KEY")
if ([string]::IsNullOrWhiteSpace($ApiKey)) {
    Write-Warning "RAG_API_KEY 未设置；服务将以无认证模式运行。仅适用于本机开发，公网部署必须设置。"
} else {
    Write-Host "API Key 已加载"
}

if ($Dev) {
    $Workers = 1
    Write-Host "[Dev] 启用自动重载，单 worker"
}

# -----------------------------------------------------------------------------
# 定位 Python 解释器
# -----------------------------------------------------------------------------
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $VenvPython) {
    $PythonExe = $VenvPython
} else {
    $PythonExe = (Get-Command python.exe -ErrorAction Stop).Source
    Write-Warning "未找到 .venv，使用系统 Python：$PythonExe"
}

Write-Host "项目根目录：$ProjectRoot"
Write-Host "Python 解释器：$PythonExe"
Write-Host "监听地址：$ListenHost`:$ListenPort"
Write-Host "Worker 数：$Workers"
Write-Host ""

# -----------------------------------------------------------------------------
# 启动 uvicorn
# -----------------------------------------------------------------------------
$args = @(
    "-m", "uvicorn",
    "server:app",
    "--host", $ListenHost,
    "--port", $ListenPort,
    "--workers", $Workers
)
if ($Dev) {
    $args += "--reload"
    $args += "--reload-dir", (Join-Path $ProjectRoot "src")
}
$args += "--log-level", "info"
$args += "--access-log"

# PYTHONPATH 让 uvicorn 能找到 src 包（main.py 已经把 src 加到 sys.path 之外）
$env:PYTHONPATH = $ProjectRoot

& $PythonExe @args
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Error "uvicorn 退出码：$exitCode"
    exit $exitCode
}
