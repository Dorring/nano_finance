#!/usr/bin/env bash
# Shared environment loader for Phase 7 deployment scripts.
# Source this from other scripts: . "$(dirname "$0")/load_env.sh"
set -euo pipefail

# Resolve repo root relative to this script's location (scripts/deploy/).
__DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${__DEPLOY_DIR}/../.." && pwd)"

# Choose config file: prefer online.env, fall back to the committed example.
__ENV_FILE="${REPO_ROOT}/config/deployment/online.env"
if [[ ! -f "${__ENV_FILE}" ]]; then
    __ENV_FILE="${REPO_ROOT}/config/deployment/online.env.example"
    echo "[load_env] config/deployment/online.env not found; using online.env.example" >&2
fi

# Load and export all variables from the chosen env file.
set -a
# shellcheck disable=SC1090
. "${__ENV_FILE}"
set +a

# Runtime directory layout.
RUNTIME_DIR="${REPO_ROOT}/runtime/phase7"
LOG_DIR="${RUNTIME_DIR}/logs"
PID_DIR="${RUNTIME_DIR}/pids"
STATUS_DIR="${RUNTIME_DIR}/status"
mkdir -p "${LOG_DIR}" "${PID_DIR}" "${STATUS_DIR}"

# tmux session names (defaults if not provided by the env file).
: "${TMUX_SESSION_MODEL:=nano-finance-model}"
: "${TMUX_SESSION_BACKEND:=nano-finance-backend}"
: "${TMUX_SESSION_FRONTEND:=nano-finance-frontend}"

export REPO_ROOT RUNTIME_DIR LOG_DIR PID_DIR STATUS_DIR
export TMUX_SESSION_MODEL TMUX_SESSION_BACKEND TMUX_SESSION_FRONTEND

# ---------------------------------------------------------------------------
# Helpers usable by any script that sources this file.
# ---------------------------------------------------------------------------

# Print a single-quoted, safely-escaped form of a value for shell interpolation.
shell_squote() {
    local s="$1"
    s="${s//\'/\'\\\'\'}"
    printf "'%s'" "$s"
}

# Wait until an HTTP endpoint returns 200, or until timeout (seconds).
# Optionally fail fast if the PID recorded in a pid file dies.
# Usage: wait_for_http_checked <url> <timeout_seconds> [pid_file]
# Returns: 0 healthy, 1 timeout, 2 process died.
wait_for_http_checked() {
    local url="$1"
    local timeout="$2"
    local pid_file="${3:-}"
    local elapsed=0 code p
    while [[ "${elapsed}" -lt "${timeout}" ]]; do
        if [[ -n "${pid_file}" && -f "${pid_file}" ]]; then
            p="$(cat "${pid_file}" 2>/dev/null || true)"
            if [[ -n "${p}" ]] && ! kill -0 "${p}" 2>/dev/null; then
                return 2
            fi
        fi
        code="$(curl -s -o /dev/null -w '%{http_code}' "${url}" 2>/dev/null || true)"
        if [[ "${code}" == "200" ]]; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

# Return 0 if a TCP port is free, 1 if in use. Best-effort across ss/netstat.
port_is_free() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
            return 1
        fi
    elif command -v netstat >/dev/null 2>&1; then
        if netstat -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
            return 1
        fi
    fi
    return 0
}
