# push.ps1 — 智能提交：调用大模型自动生成 commit message

# 0. 强制 TLS 1.2
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# 1. 读取 .env
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Error "找不到 .env 文件: $envFile"
    exit 1
}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.+?)\s*$') {
        $val = $Matches[2].Trim('"').Trim("'")
        [Environment]::SetEnvironmentVariable($Matches[1], $val, "Process")
    }
}
$apiKey  = $env:api_key
$baseUrl = $env:base_url
$model   = $env:base_model

# 如果 base_url 以 /chat/completions 结尾，去掉后缀，保留纯 base
$baseUrl = $baseUrl -replace '/chat/completions$', ''
$endpoint = "$baseUrl/chat/completions"

# 2. 验证模型是否可用
try {
    $modelsJson = curl.exe -s --max-time 15 "$baseUrl/models" -H "Authorization: Bearer $apiKey" 2>$null
    $modelsResp = $modelsJson | ConvertFrom-Json
    $availableModels = $modelsResp.data | ForEach-Object { $_.id }
    if ($model -notin $availableModels) {
        Write-Warning "模型 '$model' 不可用。"
        $chatModels = $availableModels | Where-Object { $_ -match 'step|flash|mini|large|chat' } | Sort-Object
        Write-Host "可选模型:" -ForegroundColor Cyan
        for ($i = 0; $i -lt $chatModels.Count; $i++) {
            Write-Host "  [$($i+1)] $($chatModels[$i])"
        }
        $choice = Read-Host "请选择模型编号 (直接回车用第1个)"
        if ([string]::IsNullOrWhiteSpace($choice)) { $choice = 1 }
        $model = $chatModels[[int]$choice - 1]
        Write-Host "已切换到: $model" -ForegroundColor Green
    } else {
        Write-Host "模型验证通过: $model" -ForegroundColor Green
    }
} catch {
    Write-Warning "无法验证模型，将直接尝试调用: $_"
}

# 3. 收集 git 信息
git add .

$diff = git diff --cached --stat
if ([string]::IsNullOrWhiteSpace($diff)) {
    Write-Host "没有需要提交的更改。"
    exit 0
}

$diffDetail = git diff --cached --no-color 2>$null
if ($diffDetail.Length -gt 4000) {
    $diffDetail = $diffDetail.Substring(0, 4000) + "`n... (diff truncated)"
}

# 3. 构造 prompt
$prompt = @"
你是一个 git commit message 生成器。根据以下信息生成一条简洁的中文 commit message。

规则:
- 只输出 commit message 本身，不要任何解释、前缀或引号
- 第一行不超过 50 个字符，概括改动
- 如果需要，空一行后加简短说明（英文或中文均可）

本次改动文件:
$diff

详细 diff:
$diffDetail
"@

# 4. 调用大模型 (用临时文件传 body 给 curl)
$body = @{
    model       = $model
    messages    = @(@{ role = "user"; content = $prompt })
    temperature = 0.3
} | ConvertTo-Json -Depth 5

$tmpFile = Join-Path $PSScriptRoot ".push_body_tmp.json"
[System.IO.File]::WriteAllText($tmpFile, $body)

Write-Host "正在生成 commit message..."
try {
    $tmpFileUrl = $tmpFile -replace '\\', '/'
    $respJson = curl.exe -s --max-time 30 -X POST "$endpoint" `
        -H "Content-Type: application/json" `
        -H "Authorization: Bearer $apiKey" `
        --data-binary "@$tmpFileUrl" 2>$null
    Remove-Item -Path $tmpFile -ErrorAction SilentlyContinue
    if ([string]::IsNullOrWhiteSpace($respJson)) {
        throw "curl 返回空响应"
    }
    $response = $respJson | ConvertFrom-Json
    $commitMsg = $response.choices[0].message.content.Trim()
} catch {
    Write-Warning "大模型调用失败，使用默认消息: $_"
    $commitMsg = "add"
}

Write-Host "生成的 commit message: $commitMsg"
Write-Host ""

# 5. 提交并推送
git commit -m $commitMsg
git push origin main
