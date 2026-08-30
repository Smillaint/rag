# =============================================================================
# TraceRAG + Cloudflared Windows 服务化脚本
#
# 注册两个服务（自动启动 + 崩溃自动重启）：
#   1. TraceRAG     - uvicorn，监听 127.0.0.1:8000
#   2. cloudflared  - 通过 cloudflared 自带的 service install 注册
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_services.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\install_services.ps1 -Uninstall
#
# 前提：
#   - 已运行 setup_cloudflared.ps1，cloudflared/config.yml 已就绪
#   - .env 已配置 DEEPSEEK_API_KEY 和 RAG_API_KEY
# =============================================================================
param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BinDir = Join-Path $ProjectRoot ".bin"
$LogsDir = Join-Path $ProjectRoot "logs"
$NssmExe = Join-Path $BinDir "nssm.exe"
$CloudflaredExe = Join-Path $BinDir "cloudflared.exe"
$StartScript = Join-Path $PSScriptRoot "start_server.ps1"

New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null

# -----------------------------------------------------------------------------
# 下载 NSSM（一次性）
# -----------------------------------------------------------------------------
function Get-Nssm {
    if (Test-Path -LiteralPath $NssmExe) {
        return
    }
    Write-Host "下载 NSSM 到 $BinDir ..."
    $zipUrl = "https://nssm.cc/release/nssm-2.24.zip"
    $zipPath = Join-Path $BinDir "nssm.zip"
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
    Expand-Archive -LiteralPath $zipPath -DestinationPath (Join-Path $BinDir "nssm-extract") -Force
    $extracted = Get-ChildItem -Path (Join-Path $BinDir "nssm-extract") -Recurse -Filter "nssm.exe" |
        Select-Object -First 1
    if (-not $extracted) {
        throw "nssm.exe 未在解压结果中找到"
    }
    Copy-Item -LiteralPath $extracted.FullName -Destination $NssmExe -Force
    Remove-Item -LiteralPath $zipPath -Force
    Remove-Item -LiteralPath (Join-Path $BinDir "nssm-extract") -Recurse -Force
}

# -----------------------------------------------------------------------------
# 卸载模式
# -----------------------------------------------------------------------------
if ($Uninstall) {
    Write-Host "停止并删除服务..." -ForegroundColor Yellow
    foreach ($svcName in @("TraceRAG", "cloudflared")) {
        $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
        if ($svc) {
            Write-Host "  停止 $svcName"
            Stop-Service -Name $svcName -Force -ErrorAction SilentlyContinue
            if ($svcName -eq "cloudflared" -and (Test-Path -LiteralPath $CloudflaredExe)) {
                & $CloudflaredExe service uninstall 2>&1 | Out-Null
            } else {
                & $NssmExe remove $svcName confirm 2>&1 | Out-Null
            }
            Start-Sleep -Seconds 2
            $stillThere = Get-Service -Name $svcName -ErrorAction SilentlyContinue
            if ($stillThere) {
                Write-Warning "$svcName 仍存在，可能需要 sc.exe delete $svcName"
            }
        } else {
            Write-Host "  $svcName 未安装"
        }
    }
    Write-Host "卸载完成。"
    exit 0
}

# -----------------------------------------------------------------------------
# 安装模式
# -----------------------------------------------------------------------------
Get-Nssm
Write-Host "NSSM: $NssmExe"

if (-not (Test-Path -LiteralPath $StartScript)) {
    throw "找不到 $StartScript"
}

$pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = (Get-Command python.exe -ErrorAction Stop).Source
    Write-Warning "未找到 .venv，使用系统 Python：$pythonExe"
}

# -----------------------------------------------------------------------------
# 1. 注册 TraceRAG 服务
# -----------------------------------------------------------------------------
$ServiceName = "TraceRAG"
$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "服务 $ServiceName 已存在，先停止并删除..."
    Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
    & $NssmExe remove $ServiceName confirm 2>&1 | Out-Null
    Start-Sleep -Seconds 2
}

# NSSM 调用 powershell 运行 start_server.ps1
# start_server.ps1 负责加载 .env 并启动 uvicorn
Write-Host "注册服务 $ServiceName ..."
& $NssmExe install $ServiceName "powershell.exe" "-NoProfile -ExecutionPolicy Bypass -File `"$StartScript`""
if ($LASTEXITCODE -ne 1) {
    # nssm install 在服务已存在时返回 1，在成功时返回 0
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "nssm install 返回 $LASTEXITCODE，可能服务已存在"
    }
}

& $NssmExe set $ServiceName AppDirectory $ProjectRoot
& $NssmExe set $ServiceName AppStdout (Join-Path $LogsDir "nssm_stdout.log")
& $NssmExe set $ServiceName AppStderr (Join-Path $LogsDir "nssm_stderr.log")
& $NssmExe set $ServiceName AppRotateFiles 1
& $NssmExe set $ServiceName AppRotateBytes 10485760
& $NssmExe set $ServiceName AppRotateOnline 1
& $NssmExe set $ServiceName AppStdoutCreationDisposition 4   # AppendOrCreate
& $NssmExe set $ServiceName AppStderrCreationDisposition 4
# 崩溃后 10 秒重启，无限重启
& $NssmExe set $ServiceName AppExit Default Restart
& $NssmExe set $ServiceName AppRestartDelay 10000
& $NssmExe set $ServiceName Start SERVICE_AUTO_START
& $NssmExe set $ServiceName Description "TraceRAG FastAPI 服务（uvicorn + bge-m3 + reranker）"
& $NssmExe set $ServiceName DisplayName "TraceRAG Server"

Write-Host "TraceRAG 服务已注册" -ForegroundColor Green

# -----------------------------------------------------------------------------
# 2. 注册 cloudflared 服务（用 cloudflared 自带的 service install）
# -----------------------------------------------------------------------------
$cloudflaredConfig = Join-Path $ProjectRoot "cloudflared\config.yml"
if (-not (Test-Path -LiteralPath $CloudflaredExe)) {
    Write-Warning "cloudflared.exe 未安装，跳过 cloudflared 服务注册"
    Write-Warning "请先运行 scripts\setup_cloudflared.ps1"
} elseif (-not (Test-Path -LiteralPath $cloudflaredConfig)) {
    Write-Warning "cloudflared\config.yml 不存在，跳过 cloudflared 服务注册"
    Write-Warning "请先运行 scripts\setup_cloudflared.ps1"
} else {
    $cloudflaredSvc = Get-Service -Name "cloudflared" -ErrorAction SilentlyContinue
    if ($cloudflaredSvc) {
        Write-Host "cloudflared 服务已存在，先卸载..."
        Stop-Service -Name "cloudflared" -Force -ErrorAction SilentlyContinue
        & $CloudflaredExe service uninstall 2>&1 | Out-Null
        Start-Sleep -Seconds 2
    }

    # cloudflared service install 默认读取 System32\config\systemprofile\.cloudflared\config.yml
    $systemConfigDir = "$env:SystemRoot\System32\config\systemprofile\.cloudflared"
    New-Item -ItemType Directory -Path $systemConfigDir -Force | Out-Null

    # 复制 cert.pem、config.yml、credentials.json 到系统目录
    $certPem = Join-Path $ProjectRoot "cloudflared\cert.pem"
    $creds = Get-ChildItem -Path (Join-Path $ProjectRoot "cloudflared") -Filter "*.json" |
        Where-Object { $_.Name -match '^[0-9a-fA-F-]{36}\.json$' } | Select-Object -First 1
    if (-not $certPem -or -not (Test-Path -LiteralPath $certPem)) {
        Write-Warning "cert.pem 未找到（$certPem），cloudflared 服务可能无法启动"
    } else {
        Copy-Item -LiteralPath $certPem -Destination $systemConfigDir -Force
    }
    if (-not $creds) {
        Write-Warning "credentials JSON 未找到，请确认 setup_cloudflared.ps1 已执行"
    } else {
        Copy-Item -LiteralPath $creds.FullName -Destination $systemConfigDir -Force
    }
    Copy-Item -LiteralPath $cloudflaredConfig -Destination $systemConfigDir -Force

    Write-Host "注册 cloudflared 服务..."
    & $CloudflaredExe service install
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "cloudflared service install 返回 $LASTEXITCODE"
    } else {
        Write-Host "cloudflared 服务已注册" -ForegroundColor Green
    }
}

# -----------------------------------------------------------------------------
# 3. 启动服务
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "启动服务..." -ForegroundColor Cyan
foreach ($svcName in @("TraceRAG", "cloudflared")) {
    $svc = Get-Service -Name $svcName -ErrorAction SilentlyContinue
    if (-not $svc) {
        Write-Host "  $svcName 未注册，跳过"
        continue
    }
    Write-Host "  启动 $svcName ..."
    Start-Service -Name $svcName -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
    $refreshed = Get-Service -Name $svcName
    Write-Host "  状态：$($refreshed.Status) ($($refreshed.StartType))"
}

# -----------------------------------------------------------------------------
# 4. 健康检查
# -----------------------------------------------------------------------------
Write-Host ""
Write-Host "=== 健康检查 ===" -ForegroundColor Cyan
Start-Sleep -Seconds 10   # 等待 uvicorn 起来加载模型
try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -TimeoutSec 15 -UseBasicParsing
    Write-Host "本地 /health 状态：$($resp.StatusCode)"
    Write-Host "响应：$($resp.Content)"
} catch {
    Write-Warning "本地 /health 访问失败：$($_.Exception.Message)"
    Write-Host "首次启动可能需要 1-2 分钟加载 bge-m3 模型，请稍后重试："
    Write-Host "  Invoke-WebRequest http://127.0.0.1:8000/health"
}

Write-Host ""
Write-Host "完成。" -ForegroundColor Green
Write-Host "管理命令："
Write-Host "  Stop-Service  TraceRAG"
Write-Host "  Start-Service TraceRAG"
Write-Host "  Restart-Service TraceRAG"
Write-Host "  Get-Service TraceRAG"
Write-Host ""
Write-Host "查看日志："
Write-Host "  Get-Content $LogsDir\nssm_stderr.log -Tail 50 -Wait"
Write-Host ""
Write-Host "卸载："
Write-Host "  powershell -File $PSScriptRoot\install_services.ps1 -Uninstall"
