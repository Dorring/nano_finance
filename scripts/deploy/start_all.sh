#!/usr/bin/env bash
# Start all Phase 7 services (model -> backend -> frontend), failing fast on error.
set -euo pipefail

__DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${__DIR}/load_env.sh"

echo "=== Starting all services (model -> backend -> frontend) ==="
"${__DIR}/start_model.sh"
"${__DIR}/start_backend.sh"
"${__DIR}/start_frontend.sh"
echo "=== All services started ==="

read_status() {
    local f="$1"
    if [[ -f "$f" ]]; then
        cat "$f"
    else
        echo "UNKNOWN"
    fi
}

echo
printf 'Model service: %s\n'    "$(read_status "${STATUS_DIR}/model.status")"
printf 'Backend service: %s\n'  "$(read_status "${STATUS_DIR}/backend.status")"
printf 'Frontend service: %s\n' "$(read_status "${STATUS_DIR}/frontend.status")"
