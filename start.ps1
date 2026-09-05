# start.ps1 — Single command startup for Breakout Training Studio
# Usage: .\start.ps1

Write-Host "Breakout Training Studio" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

# 1. Install backend deps if needed
Write-Host "`n[1/3] Installing Python dependencies..." -ForegroundColor Yellow
Set-Location "$PSScriptRoot\backend"
pip install -r requirements.txt -q

# 2. Start backend in background
Write-Host "[2/3] Starting backend (http://localhost:8000)..." -ForegroundColor Yellow
$backend = Start-Process -FilePath "python" -ArgumentList "main.py" `
    -WorkingDirectory "$PSScriptRoot\backend" `
    -PassThru -WindowStyle Minimized

Start-Sleep -Seconds 2

# 3. Install frontend deps if needed and start
Write-Host "[3/3] Starting frontend (http://localhost:5173)..." -ForegroundColor Yellow
Set-Location "$PSScriptRoot\frontend"
if (-not (Test-Path "node_modules")) {
    npm install -q
}

Write-Host "`nOpening http://localhost:5173 ..." -ForegroundColor Green
Start-Process "http://localhost:5173"

npm run dev

# Cleanup on exit
$backend | Stop-Process -ErrorAction SilentlyContinue
