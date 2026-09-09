#! /usr/bin/env bash
#
# Hub wrapper for export_hdf5_to_ecotaxa.py.
#
# The Hub renders an INI settings file and passes its path as the single
# argument. The exporter itself is an ordinary CLI tool, so this wrapper's
# only job is to translate INI keys into flags — keeping the script usable
# by hand and the Hub integration thin.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-}"

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: no settings file given." >&2
    echo "Usage: $0 <settings.ini>" >&2
    exit 1
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: settings file '$CONFIG_FILE' not found." >&2
    exit 1
fi

# Read one key from the INI. Values may carry inline comments and whitespace.
# A missing key must yield an empty string, not fail: the Hub omits optional
# settings entirely, and under `set -e` a non-matching grep would otherwise
# abort the script before it printed anything.
cfg() {
    grep -i "^[[:space:]]*$1[[:space:]]*=" "$CONFIG_FILE" \
        | head -1 \
        | sed 's/^[^=]*=//' \
        | sed 's/[[:space:]]*#.*//' \
        | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
        || true
}

SOURCE_PATH="$(cfg sourcePath)"
TARGET_PATH="$(cfg targetPath)"
PROFILE_NAME="$(cfg profileName)"
MAX_ZIP_SIZE="$(cfg maxZipSizeMb)"
MIN_SIZE="$(cfg minSize)"
GAUGE_PRESSURE="$(cfg gaugePressure)"
LINK_BASE="$(cfg linkBase)"
SCALE_BAR="$(cfg scaleBar)"
PIXEL_SIZE="$(cfg pixelSizeUm)"

if [[ -z "$SOURCE_PATH" ]]; then
    echo "Error: sourcePath is required." >&2
    exit 1
fi
if [[ -z "$TARGET_PATH" ]]; then
    echo "Error: targetPath is required." >&2
    exit 1
fi

ARGS=("$SOURCE_PATH" "-o" "$TARGET_PATH")
[[ -n "$PROFILE_NAME" ]] && ARGS+=("--profile-name" "$PROFILE_NAME")
[[ -n "$MAX_ZIP_SIZE" ]] && ARGS+=("--max-zip-size" "$MAX_ZIP_SIZE")
[[ -n "$MIN_SIZE" ]] && ARGS+=("--min-size" "$MIN_SIZE")
[[ -n "$LINK_BASE" ]] && ARGS+=("--link-base" "$LINK_BASE")
[[ -n "$PIXEL_SIZE" ]] && ARGS+=("--pixel-size" "$PIXEL_SIZE")
# The scale bar is on by default, so only an explicit falsey value turns it off.
case "${SCALE_BAR,,}" in
    0|false|no|off) ARGS+=("--no-scale-bar") ;;
esac
# The Hub renders booleans as 0/1; accept the usual spellings too.
case "${GAUGE_PRESSURE,,}" in
    1|true|yes|on) ARGS+=("--gauge-pressure") ;;
esac

# h5py, skimage and PIL live in the torch venv on the workers. The Hub server
# runs its own (nix) Python; leaking it in produces confusing import failures.
unset PYTHONPATH PYTHONHOME
PYTHON="${PISCO_PYTHON:-/opt/pisco/venvs/torch/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(command -v python3)"
fi

echo "=== EcoTaxa export ==="
echo "Interpreter: $PYTHON"
echo "Source:      $SOURCE_PATH"
echo "Target:      $TARGET_PATH"
echo ""

exec "$PYTHON" "$SCRIPT_DIR/export_hdf5_to_ecotaxa.py" "${ARGS[@]}"
