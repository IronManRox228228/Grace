# scripts/real_run.ps1 "<request>"
#
# Thin wrapper to run one Grace turn against the real desktop and capture run artifacts.
# Uses Grace's in-process --capture-run feature.
#
# Output lands in runs/<timestamp>/ under the repo root.
# runs/ is gitignored.

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Request
)

$ErrorActionPreference = "Stop"

# Resolve repo root correctly (one level up from scripts/)
$repoRoot = Split-Path -Parent $PSScriptRoot

$graceExe = Join-Path $repoRoot "venv\Scripts\grace.exe"
if (-not (Test-Path $graceExe)) {
    $graceExe = "grace.exe" # fall back to PATH
}

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$runDir = Join-Path (Join-Path $repoRoot "runs") $timestamp

Write-Host "Grace real run: $Request" -ForegroundColor Cyan
Write-Host "Output dir: $runDir"

# Run grace.exe in the foreground with --capture-run so console output is live
& $graceExe --text $Request --no-tts --capture-run $runDir

Write-Host "Run complete: $runDir" -ForegroundColor Cyan
