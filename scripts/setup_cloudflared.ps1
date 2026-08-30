# =============================================================================
# Cloudflare Tunnel 一键引导脚本
#
# 执行的步骤：
#   1. 检查/下载 cloudflared.exe 到项目 .bin 目录
#   2. 提示用户浏览器登录授权（一次性）
#   3. 创建命名隧道 tracerag（生成 credentials JSON）
#   4. 让用户填写域名，绑定 DNS
#   5. 由 config.yml.example 生成 config.yml
#   6. （可选）注册为 Windows 服务
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_cloudflared.ps1
# =============================================================================
param(
    [string]$Domain = "",
    [switch]$RegisterService
)

$ErrorActionPreference = "Stop"
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CloudflaredDir = Join-Path $ProjectRoot "cloudflared"
$BinDir = Join-Path $ProjectRoot ".bin"
$CloudflaredExe = Join-Path $BinDir "cloudflared.exe"

# -----------------------------------------------------------------------------
# 1. 准备 cloudflared.exe
# -----------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath $CloudflaredExe)) {
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    Write-Host "下载 cloudflared.exe 到 $BinDir ..."
    $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $url -OutFile $CloudflaredExe -UseBasicParsing
    if (-not (Test-Path -LiteralPath $CloudflaredExe)) {
        throw "cloudflared 下载失败，请手动下载并放到 $CloudflaredExe"
    }
}
Write-Host "cloudflared: $CloudflaredExe"
$version = & $CloudflaredExe --version 2>&1
Write-Host "版本：$version"

# -----------------------------------------------------------------------------
# 2. 登录授权（如果还没有 cert.pem）
# cloudflared tunnel login 不接受 --origincert，会自动生成到 ~/.cloudflared/cert.pem
# 脚本在登录完成后复制到项目目录
# -----------------------------------------------------------------------------
$CertPem = Join-Path $CloudflaredDir "cert.pem"
$defaultCert = Join-Path $env:USERPROFILE ".cloudflared\cert.pem"
if (-not (Test-Path -LiteralPath $CertPem) -and -not (Test-Path -LiteralPath $defaultCert)) {
    Write-Host ""
    Write-Host "=== 浏览器登录授权 ===" -ForegroundColor Cyan
    Write-Host "即将打开浏览器选择 Cloudflare 账户和域名。"
    Write-Host "在浏览器中授权后，本机即可管理该域名的隧道。"
    Write-Host "按 Enter 继续，或 Ctrl+C 退出..."
    [void](Read-Host)
    & $CloudflaredExe tunnel login
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "cloudflared tunnel login 返回 $LASTEXITCODE"
    }
}

# 复制 cert.pem 到项目目录
if ((Test-Path -LiteralPath $defaultCert) -and -not (Test-Path -LiteralPath $CertPem)) {
    Copy-Item -LiteralPath $defaultCert -Destination $CertPem -Force
    Write-Host "已复制 cert.pem：$defaultCert -> $CertPem"
}
if (-not (Test-Path -LiteralPath $CertPem)) {
    Write-Warning "cert.pem 仍未生成。请手动执行：$CloudflaredExe tunnel login"
    exit 1
}

# -----------------------------------------------------------------------------
# 3. 创建命名隧道（如果还没有）
# -----------------------------------------------------------------------------
$TunnelName = "tracerag"
$existingTunnels = & $CloudflaredExe tunnel list 2>&1
$hasTunnel = $existingTunnels -match $TunnelName
$TunnelUuid = ""
if (-not $hasTunnel) {
    Write-Host ""
    Write-Host "=== 创建隧道 $TunnelName ===" -ForegroundColor Cyan
    $createOutput = & $CloudflaredExe tunnel create $TunnelName 2>&1
    Write-Host ($createOutput -join "`n")
    # 解析 UUID
    $uuidMatch = [regex]::Match(($createOutput -join " "), '([0-9a-fA-F-]{36})')
    if (-not $uuidMatch.Success) {
        throw "无法从输出中解析隧道 UUID：$($createOutput -join ' ')"
    }
    $TunnelUuid = $uuidMatch.Groups[1].Value
    Write-Host "隧道 UUID: $TunnelUuid"
} else {
    # 从 list 输出解析现有 UUID
    $uuidMatch = [regex]::Match(($existingTunnels -join " "), "$TunnelName\s+([0-9a-fA-F-]{36})")
    if ($uuidMatch.Success) {
        $TunnelUuid = $uuidMatch.Groups[1].Value
    } else {
        # 直接 grep ID 列
        $lines = $existingTunnels -split "`n" | Where-Object { $_ -match $TunnelName }
        foreach ($line in $lines) {
            $cols = ($line -split '\s+') | Where-Object { $_ }
            foreach ($col in $cols) {
                if ($col -match '^[0-9a-fA-F-]{36}$') {
                    $TunnelUuid = $col
                    break
                }
            }
            if ($TunnelUuid) { break }
        }
    }
    if (-not $TunnelUuid) {
        throw "隧道 $TunnelName 已存在但无法解析 UUID，请手动执行 cloudflared tunnel list"
    }
    Write-Host "现有隧道 UUID: $TunnelUuid"
}

# 复制 credentials JSON 到项目 cloudflared 目录（默认在 ~/.cloudflared/<uuid>.json）
$defaultCreds = Join-Path $env:USERPROFILE ".cloudflared\$TunnelUuid.json"
$localCreds = Join-Path $CloudflaredDir "$TunnelUuid.json"
if ((Test-Path -LiteralPath $defaultCreds) -and -not (Test-Path -LiteralPath $localCreds)) {
    Copy-Item -LiteralPath $defaultCreds -Destination $localCreds -Force
    Write-Host "复制 credentials：$defaultCreds -> $localCreds"
}
if (-not (Test-Path -LiteralPath $localCreds)) {
    throw "找不到 credentials 文件：$localCreds。请检查 cloudflared 是否在 $env:USERPROFILE\.cloudflared 下生成了 $TunnelUuid.json"
}

# -----------------------------------------------------------------------------
# 4. 域名绑定
# -----------------------------------------------------------------------------
if (-not $Domain) {
    $Domain = Read-Host "请输入对外域名（如 rag.example.com，留空跳过 DNS 绑定）"
}
if ($Domain) {
    Write-Host ""
    Write-Host "=== 绑定 DNS $Domain -> 隧道 $TunnelName ===" -ForegroundColor Cyan
    & $CloudflaredExe tunnel route dns $TunnelName $Domain
    Write-Host "DNS 已绑定。生效后即可通过 https://$Domain 访问"
}

# -----------------------------------------------------------------------------
# 5. 生成 config.yml
# -----------------------------------------------------------------------------
$configPath = Join-Path $CloudflaredDir "config.yml"
$configTemplate = Join-Path $CloudflaredDir "config.yml.example"
$template = Get-Content -LiteralPath $configTemplate -Raw
$config = $template `
    -replace '<TUNNEL_UUID>', $TunnelUuid `
    -replace '<YOUR_DOMAIN>', $(if ($Domain) { $Domain } else { 'rag.example.com' })
$config = $config -replace 'C:\\Users\\31854\\rag_project\\cloudflared\\<TUNNEL_UUID>\.json', $localCreds
[System.IO.File]::WriteAllText($configPath, $config, [System.Text.UTF8Encoding]::new($false))
Write-Host "已生成：$configPath"

# -----------------------------------------------------------------------------
# 6. 注册 Windows 服务（可选）
# -----------------------------------------------------------------------------
if ($RegisterService) {
    Write-Host ""
    Write-Host "=== 注册 cloudflared 为 Windows 服务 ===" -ForegroundColor Cyan
    & $CloudflaredExe service install
    Write-Host "cloudflared 服务已安装。首次会以系统身份运行，cert.pem 和 config.yml 默认查找路径是 C:\Windows\System32\config\systemprofile\.cloudflared\"
    Write-Host "需要把这两个文件复制过去："
    $systemCloudflared = "$env:SystemRoot\System32\config\systemprofile\.cloudflared"
    New-Item -ItemType Directory -Path $systemCloudflared -Force | Out-Null
    Copy-Item -LiteralPath $CertPem -Destination $systemCloudflared -Force
    Copy-Item -LiteralPath $configPath -Destination $systemCloudflared -Force
    Copy-Item -LiteralPath $localCreds -Destination $systemCloudflared -Force
    Write-Host "已复制到 $systemCloudflared"
    Write-Host "启动：Start-Service cloudflared"
}

Write-Host ""
Write-Host "完成！" -ForegroundColor Green
Write-Host "下一步："
Write-Host "  1. 启动 uvicorn 服务（见 start_server.ps1）"
Write-Host "  2. 启动隧道（手动测试）："
Write-Host "     & '$CloudflaredExe' tunnel run $TunnelName --config '$configPath'"
if ($Domain) {
    Write-Host "  3. 客户端 base_url = https://$Domain"
}
