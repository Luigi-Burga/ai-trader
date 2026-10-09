#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="/home/luigi/ai-trader"
TOOLS_DIR="/home/luigi/tools"
LOGFILE="${TOOLS_DIR}/update_ai_trader.log"
LOCKFILE="/var/lock/ai-trader-operation.lock"
BUILT_COMMIT_FILE="${TOOLS_DIR}/ai-trader-built-commit"
BRANCH="main"
mkdir -p "${TOOLS_DIR}"
log() { printf '%s - %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" >> "${LOGFILE}"; }

# Shared lock: the scan holds this for its full duration.
exec 9>"${LOCKFILE}"
if ! flock -n 9; then
    log "Update skipped: another AI Trader operation is active."
    exit 0
fi

cd "${REPO_DIR}"
log "Updater started."
git fetch origin >> "${LOGFILE}" 2>&1
LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git rev-parse "origin/${BRANCH}")"
UPDATED=0

if [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]]; then
    REMOTE_PATHS="$(git diff --name-only "${LOCAL_HEAD}" "${REMOTE_HEAD}")"
    while IFS= read -r path; do
        [[ -z "${path}" ]] && continue
        if [[ -n "$(git status --porcelain --untracked-files=all -- "${path}")" ]]; then
            log "Update aborted: incoming path has local changes: ${path}"
            exit 2
        fi
    done <<< "${REMOTE_PATHS}"
    log "Fast-forwarding ${LOCAL_HEAD} -> ${REMOTE_HEAD}."
    if ! git merge --ff-only "origin/${BRANCH}" >> "${LOGFILE}" 2>&1; then
        log "Update aborted: fast-forward failed. No reset/clean was attempted."
        exit 3
    fi
    UPDATED=1
else
    log "Repository already at origin/${BRANCH}: ${LOCAL_HEAD}."
fi

CURRENT_COMMIT="$(git rev-parse HEAD)"
BUILT_COMMIT=""
[[ -f "${BUILT_COMMIT_FILE}" ]] && BUILT_COMMIT="$(cat "${BUILT_COMMIT_FILE}")"

# Retry failed builds on the next run; marker changes only after a successful build.
if [[ "${CURRENT_COMMIT}" != "${BUILT_COMMIT}" ]]; then
    log "Building ai-trader image for commit ${CURRENT_COMMIT}."
    if ! docker compose build ai-trader >> "${LOGFILE}" 2>&1; then
        log "Build failed. Marker unchanged; next run will retry."
        exit 4
    fi
    MARKER_TMP="${BUILT_COMMIT_FILE}.tmp"
    printf '%s\n' "${CURRENT_COMMIT}" > "${MARKER_TMP}"
    mv "${MARKER_TMP}" "${BUILT_COMMIT_FILE}"
    log "Build successful. Marker set to ${CURRENT_COMMIT}."
else
    log "Image already marked as built for ${CURRENT_COMMIT}; no rebuild required."
fi

if [[ "${UPDATED}" -eq 1 ]]; then
    send_telegram "AI Trader repository updated and image build verified. Commit: ${CURRENT_COMMIT}"
fi
log "Updater finished successfully."
