#!/bin/bash
set -euo pipefail

# Yui Voice Agent
# Uses the local virtualenv when present, otherwise falls back to python3.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

echo "🎤 Starting Yui Voice Agent..."
echo ""

exec "$PYTHON" "$SCRIPT_DIR/yui.py" "$@"
