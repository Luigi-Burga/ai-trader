#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="/home/luigi/ai-trader"
TOOLS_DIR="/home/luigi/tools"
LOGFILE="${TOOLS_DIR}/run_ai_trader.log"
LOCKFILE="/var/lock/ai-trader-operation.lock"
mkdir -p "${TOOLS_DIR}"
log() { printf '%s - %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" >> "${LOGFILE}"; }

# Same lock as updater: scans cannot overlap scans or builds.
exec 9>"${LOCKFILE}"
if ! flock -n 9; then
    log "Scan skipped: another AI Trader operation (scan/update) is active."
    exit 0
fi

cd "${REPO_DIR}"
COMMIT="$(git rev-parse HEAD)"
log "Production scan started. Git commit=${COMMIT}"
if docker compose run --rm ai-trader >> "${LOGFILE}" 2>&1; then
    EXIT_CODE=0
else
    EXIT_CODE=$?
fi
log "Production scan finished. Git commit=${COMMIT}; exit_code=${EXIT_CODE}"
exit "${EXIT_CODE}"
