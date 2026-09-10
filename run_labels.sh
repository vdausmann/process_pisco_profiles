#! /usr/bin/env bash
#
# Hub wrapper for ecotaxa_fetch_labels.py. Runs on the EcoTaxa machine, where
# the instance is reachable on loopback and the credentials file lives.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-}"

if [[ -z "$CONFIG_FILE" || ! -f "$CONFIG_FILE" ]]; then
    echo "Error: settings file '${CONFIG_FILE:-<none>}' not found." >&2
    exit 1
fi

cfg() {
    grep -i "^[[:space:]]*$1[[:space:]]*=" "$CONFIG_FILE" \
        | head -1 | sed 's/^[^=]*=//' | sed 's/[[:space:]]*#.*//' \
        | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' || true
}

PROJECT_IDS="$(cfg projectIds)"
TARGET_PATH="$(cfg targetPath)"
STATUS="$(cfg statusFilter)"
URL="$(cfg url)"

for required in PROJECT_IDS TARGET_PATH; do
    if [[ -z "${!required}" ]]; then
        echo "Error: $required is required." >&2
        exit 1
    fi
done

# "23022, 23052 23196" -> separate arguments, in the order given: priority
# order matters, the last project wins on a duplicate crop.
read -r -a PROJECTS <<< "${PROJECT_IDS//,/ }"

ARGS=("--projects" "${PROJECTS[@]}" "-o" "$TARGET_PATH")
[[ -n "$STATUS" ]] && ARGS+=("--status" "$STATUS")
[[ -n "$URL" ]] && ARGS+=("--url" "$URL")

unset PYTHONPATH PYTHONHOME
PYTHON="${PISCO_ECOTAXA_PYTHON:-/opt/pisco/venvs/ecotaxa/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
fi

echo "=== EcoTaxa label fetch ==="
echo "Interpreter: $PYTHON"
echo "Projects:    ${PROJECTS[*]}"
echo "Status:      ${STATUS:-V}"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/ecotaxa_fetch_labels.py" "${ARGS[@]}"
