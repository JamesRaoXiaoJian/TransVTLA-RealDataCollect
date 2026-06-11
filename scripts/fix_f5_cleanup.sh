#!/bin/bash
# F5: Remove Baidu Yun sync config files from dataset
#
# Usage:
#   bash scripts/fix_f5_cleanup.sh [--dry-run]

set -euo pipefail

SESSIONS_ROOT="dataset/phase2_realdata_sessions/sessions"

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "DRY RUN MODE — no files will be deleted"
    echo ""
    find "$SESSIONS_ROOT" -name "*.baiduyun.uploading.cfg" -type f
    COUNT=$(find "$SESSIONS_ROOT" -name "*.baiduyun.uploading.cfg" -type f | wc -l)
    echo ""
    echo "Would delete $COUNT files"
else
    echo "Removing Baidu Yun config files..."
    find "$SESSIONS_ROOT" -name "*.baiduyun.uploading.cfg" -type f -print -delete
    echo "Cleanup complete."
fi
