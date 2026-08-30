# =============================================================================
# Cloudflare 快速隧道（一次性测试用，无需登录/域名/证书）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\scripts\start_tunnel_quick.ps1
#
# 输出会包含一个 https://xxx-xxx.trycloudflare.com 的临时 URL，
# 当天有效，重启后失效。仅用于联调，不可用于生产。
#
# 前提：本地 uvicorn 已经在 127.0.0.1:8000 起来了。
# =============================================================================
param(
    [int]$LocalPort = 8000
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BinDir = Join-Path $ProjectRoot ".bin"
$CloudflaredExe = Join-Path $BinDir "cloudflared.exe"

if (-not (Test-Path -LiteralPath $CloudflaredExe)) {
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    Write-Host "下载 cloudflared.exe ..."
    $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $url -OutFile $CloudflaredExe -UseBasicParsing
}

Write-Host "启动快速隧道 -> http://127.0.0.1:$LocalPort"
Write-Host "等待 URL 输出（形如 https://xxx.trycloudflare.com）..."
Write-Host "Ctrl+C 退出。"
Write-Host ""

& $CloudflaredExe tunnel --url "http://127.0.0.1:$LocalPort"
