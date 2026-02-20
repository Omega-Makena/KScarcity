#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SESSION_CONFIG="${X_SESSION_CONFIG:-$ROOT_DIR/config/x_sessions.json}"
PROXY_FILE="${X_PROXY_FILE:-$ROOT_DIR/config/x_proxies.txt}"
CHECKPOINT_PATH="${X_CHECKPOINT_PATH:-$ROOT_DIR/data/pulse/x_scraper_checkpoint.json}"

CMD=(
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/scrape_x_kenya.py"
  "--resume"
  "--conservative-mode"
  "--detection-cooldown-hours"
  "24"
  "--checkpoint"
  "$CHECKPOINT_PATH"
)

if [[ -f "$SESSION_CONFIG" ]]; then
  CMD+=("--session-config" "$SESSION_CONFIG")
fi

if [[ -f "$PROXY_FILE" ]]; then
  CMD+=("--proxy-file" "$PROXY_FILE")
fi

if [[ "${X_WAIT_COOLDOWN:-0}" == "1" ]]; then
  CMD+=("--wait-cooldown")
fi

CMD+=("$@")

exec "${CMD[@]}"
