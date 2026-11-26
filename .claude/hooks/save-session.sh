#!/usr/bin/env bash
# Copies the finished session's transcript into the repo.
src=$(jq -r '.transcript_path')
dest_dir="$CLAUDE_PROJECT_DIR/.claude/sessions"
mkdir -p "$dest_dir"
dest="$dest_dir/$(basename "$src")"
# When the session was resumed straight from the repo file, src and dest are the same
[ "$src" -ef "$dest" ] || cp "$src" "$dest"
