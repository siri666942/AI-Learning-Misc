# push.ps1 — 智能提交：调用大模型自动生成 commit message

# 1. 读取 .env
$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Error "找不到 .env 文件: $envFile"
    exit 1
}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.+?)\s*$') {
        [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}
$apiKey  = $env:api_key
$baseUrl = $env:base_url
$model   = $env:base_model

# 如果 base_url 以 /chat/completions 结尾，去掉后缀，保留纯 base
$baseUrl = $baseUrl -replace '/chat/completions$', ''
$endpoint = "$baseUrl/chat/completions"

# 2. 收集 git 信息
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

# 4. 调用大模型
$body = @{
    model    = $model
    messages = @(
        @{ role = "user"; content = $prompt }
    )
    temperature = 0.3
} | ConvertTo-Json -Depth 5

$headers = @{
    "Content-Type"  = "application/json"
    "Authorization" = "Bearer $apiKey"
}

Write-Host "正在生成 commit message..."
try {
    $response = Invoke-RestMethod -Uri $endpoint -Method Post -Headers $headers -Body $body -TimeoutSec 30
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
