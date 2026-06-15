#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_ISAAC_ROOT="${REAL_ISAAC_ROOT:-/home/james/isaacsim}"

if [[ ! -x "$REAL_ISAAC_ROOT/python.sh" ]]; then
    printf 'error: Isaac Sim python.sh not found or not executable: %s\n' "$REAL_ISAAC_ROOT/python.sh" >&2
    exit 1
fi

if (($# > 0)) && [[ "$1" != -* && "$1" != /* && ! -e "$1" && -e "$SCRIPT_DIR/$1" ]]; then
    set -- "$SCRIPT_DIR/$1" "${@:2}"
fi

exec "$REAL_ISAAC_ROOT/python.sh" "$@"
