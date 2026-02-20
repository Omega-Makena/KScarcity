$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$SessionConfig = if ($env:X_SESSION_CONFIG) { $env:X_SESSION_CONFIG } else { Join-Path $RootDir "config/x_sessions.json" }
$ProxyFile = if ($env:X_PROXY_FILE) { $env:X_PROXY_FILE } else { Join-Path $RootDir "config/x_proxies.txt" }
$CheckpointPath = if ($env:X_CHECKPOINT_PATH) { $env:X_CHECKPOINT_PATH } else { Join-Path $RootDir "data/pulse/x_scraper_checkpoint.json" }

$ArgsList = @(
  (Join-Path $RootDir "scripts/scrape_x_kenya.py"),
  "--resume",
  "--conservative-mode",
  "--detection-cooldown-hours", "24",
  "--checkpoint", $CheckpointPath
)

if (Test-Path $SessionConfig) {
  $ArgsList += @("--session-config", $SessionConfig)
}

if (Test-Path $ProxyFile) {
  $ArgsList += @("--proxy-file", $ProxyFile)
}

if ($env:X_WAIT_COOLDOWN -eq "1") {
  $ArgsList += @("--wait-cooldown")
}

$ArgsList += $args

& $PythonBin @ArgsList
