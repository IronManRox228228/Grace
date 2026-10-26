# scripts/app_run.ps1
#
# Starts the Grace app (Tauri overlay + Python backend) with run capture on.
# A dev build loads the UI from the Vite dev server on :5173, so this starts
# that first and stops it again when the shell exits.
#
# Captures land in runs/<timestamp>/ under the repo root (gitignored).

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

$env:GRACE_BACKEND = "python"
$env:GRACE_CAPTURE_RUN = "1"

$vite = $null
$viteUp = $false
# Vite binds to "localhost", which can be IPv6-only ([::1]), so try both.
function Test-Vite {
    foreach ($addr in @("::1", "127.0.0.1")) {
        try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect($addr, 5173); $c.Close(); return $true } catch {}
    }
    return $false
}
$viteUp = Test-Vite
if (-not $viteUp) {
    Write-Host "Starting the UI dev server on :5173..." -ForegroundColor Cyan
    $vite = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev:renderer" `
        -WorkingDirectory (Join-Path $repoRoot "frontend") -WindowStyle Hidden -PassThru
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        if (Test-Vite) { $viteUp = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $viteUp) { throw "The UI dev server did not come up on :5173 within 60 s." }
}

try {
    Write-Host "Starting Grace (Python backend, run capture on)..." -ForegroundColor Cyan
    Push-Location $repoRoot
    cargo run -p grace-shell
} finally {
    Pop-Location
    if ($vite -ne $null -and -not $vite.HasExited) {
        # npm spawns node as a child; stop the whole tree it started.
        & taskkill /PID $vite.Id /T /F | Out-Null
    }
}
