#! /usr/bin/env bash
#
# Hub wrapper for ecotaxa_import.py. Runs on the EcoTaxa machine, where the
# instance is reachable on loopback and the credentials file lives.

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

ARCHIVE="$(cfg archivePath)"
TARGET="$(cfg targetPath)"
PROJECT_ID="$(cfg projectId)"
PROJECT_TITLE="$(cfg projectTitle)"
INSTRUMENT="$(cfg instrument)"
VISIBLE="$(cfg visible)"
ACCESS="$(cfg access)"
UPDATE_MODE="$(cfg updateMode)"
SKIP_EXISTING="$(cfg skipExistingObjects)"
URL="$(cfg url)"

for required in ARCHIVE TARGET; do
    if [[ -z "${!required}" ]]; then
        echo "Error: $required is required." >&2
        exit 1
    fi
done

ARGS=("$ARCHIVE" "-o" "$TARGET")
[[ -n "$PROJECT_ID" ]] && ARGS+=("--project-id" "$PROJECT_ID")
[[ -n "$PROJECT_TITLE" ]] && ARGS+=("--project-title" "$PROJECT_TITLE")
[[ -n "$INSTRUMENT" ]] && ARGS+=("--instrument" "$INSTRUMENT")
[[ -n "$UPDATE_MODE" ]] && ARGS+=("--update-mode" "$UPDATE_MODE")
[[ -n "$URL" ]] && ARGS+=("--url" "$URL")
[[ -n "$ACCESS" ]] && ARGS+=("--access" "${ACCESS,,}")
case "${SKIP_EXISTING,,}" in 1|true|yes|on) ARGS+=("--skip-existing-objects") ;; esac
# visible defaults to true; only an explicit falsey value turns it off.
case "${VISIBLE,,}" in 0|false|no|off) ARGS+=("--no-visible") ;; esac

unset PYTHONPATH PYTHONHOME
PYTHON="${PISCO_ECOTAXA_PYTHON:-/opt/pisco/venvs/ecotaxa/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Warning: $PYTHON is not executable, falling back to the system python." >&2
    PYTHON="$(command -v python3)"
fi

echo "=== EcoTaxa import ==="
echo "Interpreter: $PYTHON"
echo "Archive:     $ARCHIVE"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/ecotaxa_import.py" "${ARGS[@]}"
