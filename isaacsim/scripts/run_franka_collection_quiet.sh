#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

exec "$ISAAC_ROOT/python.sh" "$SCRIPT_DIR/franka_data_collecter.py" "$@"
