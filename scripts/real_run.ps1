# scripts/real_run.ps1 "<request>"
#
# Runs one Grace turn against the real desktop and saves everything to
# runs/<timestamp>/ so the user can review it and Claude can read it:
#
#   * the full log                        -> runs/<timestamp>/log.txt
#   * each step's screenshot              -> runs/<timestamp>/step_NNN/screenshot.png
#   * each step's raw planner output      -> runs/<timestamp>/step_NNN/planner.json
#   * each step's element count           -> runs/<timestamp>/step_NNN/element_count.txt
#
# The per-step data is captured live from Grace's WebSocket event stream
# (ws://127.0.0.1:8765 by default) while grace.exe runs. That stream is best
# effort: if Grace is not emitting a particular field, the corresponding file
# is simply not written; the full log is always captured.
#
# runs/ is gitignored - the user reviews these, and Claude reads them.

param(
    [Parameter(Mandatory = $true)]
    [string]$Request
)

$ErrorActionPreference = "Continue"

# Locate grace.exe next to this script (repo root) so the script works from
# any working directory.
$scriptDir = Split-Path -Parent $PSScriptRoot
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$graceExe = Join-Path $repoRoot "venv\Scripts\grace.exe"
if (-not (Test-Path $graceExe)) {
    $graceExe = "grace.exe" # fall back to PATH
}

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$runDir = Join-Path (Join-Path $repoRoot "runs") $timestamp
$null = New-Item -ItemType Directory -Path $runDir -Force

$logPath = Join-Path $runDir "log.txt"
$exitCodePath = Join-Path $runDir "exit_code.txt"

Write-Host "Grace real run: $Request" -ForegroundColor Cyan
Write-Host "Output dir: $runDir"

# Capture every byte grace.exe writes (stdout + stderr) into the full log.
# Start-Process buffers to the log file; we also mirror to the console.
$logStream = [System.IO.File]::Create($logPath)
try {
    $proc = Start-Process -FilePath $graceExe -ArgumentList @("--text", $Request, "--no-tts") -NoNewWindow -PassThru
} catch {
    "Failed to start grace.exe: $_" | Out-File -FilePath $logPath -Encoding utf8
    Write-Host "Failed to start grace.exe: $_" -ForegroundColor Red
    exit 1
}

# --- Per-step capture from the WebSocket event stream (best effort) --------
$wsUrl = "ws://127.0.0.1:8765"
$step = 0
$ws = $null
try {
    # Wait for Grace's WebSocket server to come up (it starts during startup).
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        if ($proc.HasExited) { break }
        try {
            $ws = New-Object System.Net.WebSockets.ClientWebSocket
            $cts = New-Object System.Threading.CancellationTokenSource(800)
            $connectTask = $ws.ConnectAsync((New-Object System.Uri($wsUrl)), $cts.Token)
            if ($connectTask.Wait(800) -and $ws.State -eq 'Open') {
                break
            }
        } catch {
            # Ignore - retry.
        }
        if ($ws) { try { $ws.Dispose() } catch {}; $ws = $null }
        Start-Sleep -Milliseconds 400
    }

    if ($ws -and $ws.State -eq 'Open') {
        $buffer = New-Object System.Byte[] 16384
        while ($proc.HasExited -eq $false -and $ws.State -eq 'Open') {
            $cts = New-Object System.Threading.CancellationTokenSource(1200)
            $recv = $ws.ReceiveAsync((New-Object System.ArraySegment[byte]($buffer)), $cts.Token)
            if (-not $recv.Wait(1200)) { continue }
            if ($recv.Result.MessageType -eq 'Close') { break }
            $text = [System.Text.Encoding]::UTF8.GetString($buffer, 0, $recv.Result.Count)
            if ([string]::IsNullOrWhiteSpace($text)) { continue }

            $step++
            $stepDir = Join-Path $runDir ("step_{0:D3}" -f $step)
            $null = New-Item -ItemType Directory -Path $stepDir -Force

            # Full event, exactly as Grace sent it.
            $text | Out-File -FilePath (Join-Path $stepDir "event.json") -Encoding utf8

            try {
                $event = $text | ConvertFrom-Json -Depth 10
            } catch {
                $event = $null
            }

            if ($event) {
                # Raw planner output, if the event carries one.
                foreach ($prop in @("planner", "planner_output", "raw_planner_output", "raw")) {
                    if ($event.PSObject.Properties.Name -contains $prop -and $event.$prop -ne $null) {
                        $event.$prop | Out-File -FilePath (Join-Path $stepDir ("planner_" + $prop + ".json")) -Encoding utf8
                    }
                }
                # Element count, if the event carries one.
                foreach ($prop in @("element_count", "elements", "count")) {
                    if ($event.PSObject.Properties.Name -contains $prop -and $event.$prop -ne $null) {
                        "$($event.$prop)" | Out-File -FilePath (Join-Path $stepDir "element_count.txt") -Encoding utf8
                    }
                }
                # Screenshot, if the event carries a base64 PNG.
                foreach ($prop in @("screenshot", "png_b64", "image", "image_b64")) {
                    if ($event.PSObject.Properties.Name -contains $prop -and $event.$prop) {
                        try {
                            $bytes = [System.Convert]::FromBase64String([string]$event.$prop)
                            [System.IO.File]::WriteAllBytes((Join-Path $stepDir "screenshot.png"), $bytes)
                        } catch {
                            # Not valid base64 - skip.
                        }
                    }
                }
            }
        }
    } else {
        Write-Warning "Could not connect to Grace's WebSocket ($wsUrl); per-step capture skipped."
    }
} catch {
    Write-Warning "Per-step capture encountered an error: $_"
} finally {
    if ($ws) {
        try { $ws.CloseAsync().Wait(1000) } catch {}
        $ws.Dispose()
    }
}

# Wait for grace.exe to finish and record its exit code.
$proc.WaitForExit()
"$($proc.ExitCode)" | Out-File -FilePath $exitCodePath -Encoding utf8

# Mirror the captured log to the console so the user sees the outcome.
try { Get-Content $logPath } catch {}

Write-Host "Run complete: $runDir (exit code $($proc.ExitCode), $step step events captured)" -ForegroundColor Cyan
