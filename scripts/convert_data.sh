#!/usr/bin/env bash
# Wrapper: `python scripts/convert_data.py …` from any cwd.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export REPO_ROOT
exec python3 "$SCRIPT_DIR/convert_data.py" "$@"
