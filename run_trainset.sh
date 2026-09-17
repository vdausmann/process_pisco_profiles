#! /usr/bin/env bash
#
# Hub wrapper for training/build_trainset.py. Runs where the crops are: it
# reads the segmenter's result.h5 through the same code path inference uses.

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

LABELS_PATH="$(cfg labelsPath)"
CROPS_PATH="$(cfg cropsPath)"
TARGET_PATH="$(cfg targetPath)"
ISOLATE="$(cfg isolateCrops)"
NONLIVING="$(cfg nonLiving)"
EXCLUDE="$(cfg excludeMulticlass)"
MIN_PER_CLASS="$(cfg minPerClass)"

for required in LABELS_PATH CROPS_PATH TARGET_PATH; do
    if [[ -z "${!required}" ]]; then
        echo "Error: $required is required." >&2
        exit 1
    fi
done

# Comma-separated lists become separate arguments. Class names may contain
# spaces ("multiple species"), so only commas separate them.
split_list() {
    local IFS=','
    read -r -a PARTS <<< "$1"
    for part in "${PARTS[@]}"; do
        part="${part#"${part%%[![:space:]]*}"}"
        part="${part%"${part##*[![:space:]]}"}"
        [[ -n "$part" ]] && printf '%s\n' "$part"
    done
}

ARGS=("--labels" "$LABELS_PATH" "--out" "$TARGET_PATH" "--crops-root")
while IFS= read -r root; do ARGS+=("$root"); done < <(split_list "$CROPS_PATH")

if [[ -n "$NONLIVING" ]]; then
    ARGS+=("--nonliving")
    while IFS= read -r cls; do ARGS+=("$cls"); done < <(split_list "$NONLIVING")
fi
if [[ -n "$EXCLUDE" ]]; then
    ARGS+=("--exclude-multiclass")
    while IFS= read -r cls; do ARGS+=("$cls"); done < <(split_list "$EXCLUDE")
fi
[[ -n "$MIN_PER_CLASS" ]] && ARGS+=("--min-per-class" "$MIN_PER_CLASS")
case "${ISOLATE,,}" in 1|true|yes|on) ARGS+=("--isolate-crops") ;; esac

unset PYTHONPATH PYTHONHOME
PYTHON="${PISCO_PYTHON:-/opt/pisco/venvs/torch/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Warning: $PYTHON is not executable, falling back to the system python." >&2
    PYTHON="$(command -v python3)"
fi
if ! "$PYTHON" -c "import h5py, numpy, PIL, torch" 2>/dev/null; then
    echo "Error: $PYTHON cannot import h5py, numpy, PIL, torch. Set PISCO_PYTHON to an interpreter that can." >&2
    exit 1
fi

echo "=== Training set build ==="
echo "Interpreter: $PYTHON"
echo "Labels:      $LABELS_PATH"
echo "Crops:       $CROPS_PATH"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/training/build_trainset.py" "${ARGS[@]}"
