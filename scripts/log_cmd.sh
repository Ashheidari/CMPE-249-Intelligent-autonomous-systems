#!/usr/bin/env bash
# Usage: scripts/log_cmd.sh <command...>
# Appends a timestamped copy of the command to logs/commands.log, then runs it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$ROOT/logs/commands.log"
exec "$@"
