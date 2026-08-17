# 一键备份：知识库 / 记忆库 / 配置 / 上传 / Redis
$ErrorActionPreference = "Stop"
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$src = "D:\ai_learning"
$dstRoot = "D:\ai_learning_backups"
$dst = Join-Path $dstRoot ("backup_" + $ts)
New-Item -ItemType Directory -Force $dst | Out-Null

Write-Host "[1/5] 备份知识库..." -ForegroundColor Cyan
Copy-Item -Path "$src\data\chroma_db" -Destination "$dst\chroma_db" -Recurse -Force

Write-Host "[2/5] 备份记忆库..." -ForegroundColor Cyan
Copy-Item -Path "$src\data\memory_db" -Destination "$dst\memory_db" -Recurse -Force

Write-Host "[3/5] 备份配置(.env)..." -ForegroundColor Cyan
Copy-Item -Path "$src\.env" -Destination "$dst\.env" -Force

Write-Host "[4/5] 备份上传文件..." -ForegroundColor Cyan
if (Test-Path "$src\uploads") {
    Copy-Item -Path "$src\uploads" -Destination "$dst\uploads" -Recurse -Force
}

Write-Host "[5/5] 备份 Redis(可选)..." -ForegroundColor Cyan
try {
    $cid = docker ps --filter "name=ai_learning-redis" --format "{{.ID}}"
    if ($cid) {
        docker exec $cid redis-cli SAVE | Out-Null
        docker cp "$cid`:/data/dump.rdb" "$dst\redis_dump.rdb" 2>$null
        Write-Host "Redis 快照已备份" -ForegroundColor Green
    } else {
        Write-Host "Redis 容器未运行，跳过" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Redis 备份跳过: $_" -ForegroundColor Yellow
}

# 只保留最近 7 份
$old = Get-ChildItem $dstRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -Skip 7
foreach ($d in $old) { Remove-Item $d -Recurse -Force }

Write-Host ""
Write-Host "OK 备份完成: $dst" -ForegroundColor Green
