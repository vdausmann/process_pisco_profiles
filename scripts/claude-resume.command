#!/usr/bin/env bash
# Pulls the latest changes and resumes the most recently committed Claude session.
cd "$(dirname "$0")/.."
git pull -q --rebase --autostash || echo "(Couldn't pull from GitLab, using the local copy.)"
latest=$(git log -1 --name-only --format= -- .claude/sessions | grep '\.jsonl$' | head -1)
if [ -z "$latest" ]; then
  echo "No saved sessions yet. Starting a new one."
  claude
else
  claude --resume "$PWD/$latest"
fi
