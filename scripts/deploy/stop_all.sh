#!/usr/bin/env bash
# Stop all Phase 7 services in reverse order (frontend -> backend -> model).
# Only stops processes we started: by tmux session name and by recorded PID.
# Never uses pkill -f, killall, or killing by process name.
set -euo pipefail

__DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${__DIR}/load_env.sh"

# stop_service <name> <session> <pid_file> <status_file>
stop_service() {
    local name="$1" session="$2" pid_file="$3" status_file="$4"
    echo "[stop] ${name} ..."

    # Kill the tmux session first; this takes the whole process tree with it.
    if tmux has-session -t "${session}" 2>/dev/null; then
        tmux kill-session -t "${session}"
        echo "[stop] ${name}: killed tmux session '${session}'."
    else
        echo "[stop] ${name}: no tmux session '${session}'."
    fi

    # Kill by the PID we recorded (only this PID, never by process name).
    if [[ -f "${pid_file}" ]]; then
        local pid
        pid="$(cat "${pid_file}" 2>/dev/null || true)"
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            echo "[stop] ${name}: sending TERM to PID ${pid}."
            kill -TERM "${pid}" 2>/dev/null || true
            local waited=0
            while [[ "${waited}" -lt 10 ]] && kill -0 "${pid}" 2>/dev/null; do
                sleep 1
                waited=$((waited + 1))
            done
            if kill -0 "${pid}" 2>/dev/null; then
                echo "[stop] ${name}: still alive after 10s; sending KILL to PID ${pid}."
                kill -9 "${pid}" 2>/dev/null || true
            fi
        else
            echo "[stop] ${name}: PID ${pid:-<none>} not running."
        fi
        rm -f "${pid_file}"
    else
        echo "[stop] ${name}: no PID file."
    fi

    rm -f "${status_file}"
}

stop_service "Frontend" "${TMUX_SESSION_FRONTEND}" "${PID_DIR}/frontend.pid" "${STATUS_DIR}/frontend.status"
stop_service "Backend"  "${TMUX_SESSION_BACKEND}"  "${PID_DIR}/backend.pid"  "${STATUS_DIR}/backend.status"
stop_service "Model"    "${TMUX_SESSION_MODEL}"    "${PID_DIR}/model.pid"    "${STATUS_DIR}/model.status"

echo "[stop] Done."
