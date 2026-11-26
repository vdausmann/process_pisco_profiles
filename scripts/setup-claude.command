#!/usr/bin/env bash
# Sets up a git-versioned Claude Code project, connected to GitLab.
#
# First time on a computer (file downloaded or received by mail/chat):
#   open Terminal, type "bash " (with a space), drag this file into the window,
#   press Enter. This avoids macOS's warnings for downloaded files.
# Copies inside a project (scripts/setup-claude.command) come via git and can
# simply be double-clicked.
#
# Where the project ends up:
#   - Script inside an existing project (e.g. scripts/setup-claude.command):
#       re-runs the setup for that project.
#   - Script anywhere else: asks for the project's GitLab SSH URL and creates
#       the project folder next to this script, named after the GitLab project.
#       If the script's folder already holds files, you can instead make that
#       folder itself the project.
#
# What it does:
#   - checks tools (git, jq), your git name/email and your SSH access to GitLab
#   - creates the repo or joins the GitLab project's existing history
#   - auto memory in .claude/memory/, session transcripts in .claude/sessions/
#   - gitignores machine-specific files and files over 50 MB
#   - commits and pushes
#
# Safe to run more than once.
set -euo pipefail

script_path="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
script_dir="$(dirname "$script_path")"
remote="${1:-}"
LARGE_FILE_MB=50

# Linux: a double-click from the file manager may start the script without a
# terminal window. Reopen it in one so its questions can be answered.
if [ ! -t 0 ] && [ "$(uname)" = Linux ] && [ -z "${SETUP_CLAUDE_IN_TERMINAL:-}" ] \
   && { [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; }; then
  export SETUP_CLAUDE_IN_TERMINAL=1
  for t in x-terminal-emulator gnome-terminal konsole xfce4-terminal kgx xterm; do
    command -v "$t" >/dev/null || continue
    case "$t" in
      gnome-terminal|kgx) exec "$t" -- bash "$script_path" "$@" ;;
      *)                  exec "$t" -e bash "$script_path" "$@" ;;
    esac
  done
fi

# Keep the Terminal window open after a double-click so messages can be read
case "$0" in
  *.command) trap 'echo; printf "Press Enter to close this window..."; read -r _ || true' EXIT ;;
esac

ok()   { printf '✓ %s\n' "$*"; }
warn() { printf '! %s\n' "$*"; }
die()  { printf '\nError: %s\n' "$*" >&2; exit 1; }
ask()  {  # ask "question" "default"
  local a=""
  printf '%s ' "$1" >&2
  read -r a || true
  printf '%s' "${a:-${2:-}}"
}
yes_no() { case "$(ask "$1 [y/N]" n)" in y|Y|yes|Yes|j|J|ja) return 0;; *) return 1;; esac; }
yes_no_y() { case "$(ask "$1 [Y/n]" y)" in n|N|no|No|nein) return 1;; *) return 0;; esac; }

# --- Tools ------------------------------------------------------------------
command -v git >/dev/null || die "git is not installed (Mac: run  xcode-select --install)."
if ! command -v jq >/dev/null; then
  if command -v brew >/dev/null && yes_no "jq is required. Install it now with Homebrew?"; then
    brew install jq
  else
    die "jq is required.  Mac: brew install jq   Linux: sudo apt install jq  (Fedora: sudo dnf install jq)"
  fi
fi

# --- Which folder is the project? ----------------------------------------------
cwd="$(pwd)"
root=""
if top="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null)"; then
  # The script's copy inside a project (scripts/setup-claude.command)
  root="$top"
elif [ "$cwd" != "$HOME" ] && ! [ "$cwd" -ef "$script_dir" ]; then
  # Started from a terminal inside an existing project folder
  cand="$(git -C "$cwd" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$cwd")"
  echo "Current folder: $cand"
  if yes_no_y "Set up Claude for this existing project? (n = create a new project instead)"; then
    root="$cand"
  fi
fi

if [ -n "$root" ]; then
  # Existing project without GitLab connection: ask for it so everything gets pushed
  if [ -z "$remote" ] && ! git -C "$root" remote get-url origin >/dev/null 2>&1; then
    echo "This project isn't connected to GitLab yet."
    echo "Paste its SSH URL from GitLab (Code / Clone → Clone with SSH),"
    remote="$(ask "or press Enter to keep it local only:" "")"
  fi
else
  if [ -z "$remote" ]; then
    echo "Paste the project's SSH URL from GitLab (Code / Clone → Clone with SSH),"
    remote="$(ask "e.g. git@git.geomar.de:group/project.git :" "")"
  fi
  [ -n "$remote" ] || die "No GitLab URL given."
  name="${remote%/}"; name="${name##*/}"; name="${name##*:}"; name="${name%.git}"
  [ -n "$name" ] || die "Can't read a project name from: $remote"

  # Run straight from Downloads or the Desktop: ask where projects should live
  base="$script_dir"
  case "$script_dir" in
    "$HOME/Downloads"|"$HOME/Downloads/"*|"$HOME/Desktop"|"$HOME/Desktop/"*)
      echo
      base="$(ask "Create the project folder '$name' in which folder? [~/projects]:" "$HOME/projects")"
      base="${base/#\~/$HOME}"
      base="${base%/}"
      mkdir -p "$base" || die "Can't create the folder: $base"
      base="$(cd "$base" && pwd)" ;;
  esac

  choice=1
  others=""
  if [ "$base" = "$script_dir" ]; then
    others="$(ls -A "$script_dir" | grep -vxF -e "$(basename "$script_path")" -e .DS_Store || true)"
  fi
  if [ -n "$others" ]; then
    echo
    echo "The folder this script is in already contains files:"
    echo "  $script_dir"
    echo "  1) Create a new project folder '$name' in it (default)"
    echo "  2) Make this folder itself the project, including its files"
    choice="$(ask "Choose 1 or 2:" 1)"
  fi
  if [ "$choice" = 2 ]; then root="$script_dir"; else root="$base/$name"; fi

  if [ -d "$root" ] && [ "$root" != "$script_dir" ] && [ -n "$(ls -A "$root")" ] \
     && ! git -C "$root" rev-parse --show-toplevel >/dev/null 2>&1; then
    die "'$root' already exists and isn't a git project. Move or rename it, then run this again."
  fi
  mkdir -p "$root"
fi
cd "$root"
echo
echo "Setting up Claude Code project in: $root"

# --- Cloud-synced folders ----------------------------------------------------
case "$root" in
  *Nextcloud*|*Dropbox*|*OneDrive*|*"Google Drive"*|*"Mobile Documents"*)
    warn "This folder is synced by a cloud service. Syncing a git repository that way"
    warn "can corrupt it when you work on several machines. Better: put the script"
    warn "in e.g. ~/projects/ and run it there."
    yes_no "Continue here anyway?" || exit 1 ;;
esac

# --- Git identity -------------------------------------------------------------
if [ -z "$(git config user.name || true)" ]; then
  n="$(ask "Your name for git commits:" "")"
  [ -n "$n" ] || die "git needs a name for commits."
  git config --global user.name "$n"
fi
if [ -z "$(git config user.email || true)" ]; then
  e="$(ask "Your email for git commits:" "")"
  [ -n "$e" ] || die "git needs an email for commits."
  git config --global user.email "$e"
fi

# --- Git repository -------------------------------------------------------------
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  top="$(git rev-parse --show-toplevel)"
  [ "$top" -ef "$root" ] || die "this folder is inside another git repository ($top)."
else
  git init -q
  git symbolic-ref HEAD refs/heads/main
  ok "Created git repository"
fi
has_commits() { git rev-parse -q --verify HEAD >/dev/null; }

# --- SSH access ---------------------------------------------------------------
ssh_ok() {  # ssh_ok target port
  local out
  out="$(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
         ${2:+-p "$2"} "$1" </dev/null 2>&1 || true)"
  printf '%s' "$out" | grep -qi "welcome"
}

setup_ssh() {  # setup_ssh target port
  local target="$1" port="$2" host="${1#*@}" key="" k tries=0
  if ssh_ok "$target" "$port"; then ok "SSH access to $host"; return; fi
  warn "No SSH access to $host yet. Setting up an SSH key."

  for k in id_ed25519 id_ecdsa id_rsa; do
    [ -f "$HOME/.ssh/$k.pub" ] && { key="$HOME/.ssh/$k"; break; }
  done
  if [ -z "$key" ]; then
    key="$HOME/.ssh/id_ed25519"
    mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
    echo "Creating a new SSH key. You can set a passphrase or press Enter twice for none."
    ssh-keygen -t ed25519 -C "$(git config user.email)" -f "$key"
  fi
  if [ "$(uname)" = Darwin ]; then
    ssh-add --apple-use-keychain "$key" 2>/dev/null || ssh-add -K "$key" 2>/dev/null || true
  else
    ssh-add "$key" 2>/dev/null || true
  fi

  echo
  echo "Add this public key to your GitLab account:"
  echo "  https://$host  →  your avatar  →  Edit profile  →  SSH Keys  →  Add new key"
  echo
  cat "$key.pub"
  echo
  copied=""
  if command -v pbcopy >/dev/null; then pbcopy < "$key.pub" && copied=1
  elif [ -n "${WAYLAND_DISPLAY:-}" ] && command -v wl-copy >/dev/null; then wl-copy < "$key.pub" && copied=1
  elif command -v xclip >/dev/null; then xclip -selection clipboard < "$key.pub" 2>/dev/null && copied=1
  fi
  [ -n "$copied" ] && echo "(The key is already copied to your clipboard.)"

  while ! ssh_ok "$target" "$port"; do
    tries=$((tries + 1))
    [ "$tries" -le 3 ] || die "Still no SSH access to $host. Check the key in GitLab and run the script again."
    ask "Press Enter once you've added the key in GitLab..." "" >/dev/null
  done
  ok "SSH access to $host"
}

remote_branch=""
if [ -n "$remote" ]; then
  case "$remote" in
    http://*|https://*)
      die "Use the SSH URL (git@...:group/project.git) from GitLab's Clone button, not the https one." ;;
    ssh://*)
      hostpart="${remote#ssh://}"; hostpart="${hostpart%%/*}"
      target="${hostpart%%:*}"; port=""
      [ "$hostpart" != "$target" ] && port="${hostpart##*:}"
      setup_ssh "$target" "$port" ;;
    /*|file://*) : ;;  # local path, no SSH needed
    *@*:*)
      setup_ssh "${remote%%:*}" "" ;;
    *) die "Don't recognise the remote URL: $remote" ;;
  esac

  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$remote"
  else
    git remote add origin "$remote"
  fi
  ok "Remote: $remote"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git fetch -q origin || die "Could not reach the GitLab project. Check the URL and that you're a member of the project or its group."
  remote_branch="$(git ls-remote --symref origin HEAD 2>/dev/null | awk '/^ref:/ {sub("refs/heads/", "", $2); print $2}')"

  # New folder or brand-new repo: take over the project's existing history,
  # keeping any files that are already here.
  if [ -n "$remote_branch" ] && ! has_commits; then
    local_gitignore=""
    [ -f .gitignore ] && local_gitignore="$(cat .gitignore)"
    git reset -q "origin/$remote_branch"
    git branch -M "$remote_branch"
    git branch -q --set-upstream-to "origin/$remote_branch"
    if [ -n "$(git ls-files -d)" ]; then git ls-files -d -z | xargs -0 git checkout -- ; fi
    if [ -n "$local_gitignore" ] && git cat-file -e "origin/$remote_branch:.gitignore" 2>/dev/null; then
      { git show "origin/$remote_branch:.gitignore"; printf '%s\n' "$local_gitignore"; } | awk '!seen[$0]++' > .gitignore
    fi
    ok "Downloaded the GitLab project ($remote_branch)"
  fi
fi

# --- Folders and a copy of this script --------------------------------------
mkdir -p .claude/memory .claude/sessions .claude/hooks scripts
[ "$script_path" -ef scripts/setup-claude.command ] || cp "$script_path" scripts/setup-claude.command
chmod +x scripts/setup-claude.command
# A downloaded copy sitting in the project folder itself now lives in scripts/
if [ "$script_dir" -ef "$root" ]; then
  rm -f "$script_path"
  ok "Moved this script to scripts/setup-claude.command"
fi
# Remove the earlier version's file names
for old in scripts/setup-claude.sh scripts/claude-resume.sh; do
  [ -f "$old" ] && ! [ "$script_path" -ef "$old" ] && { git rm -q -f --cached "$old" 2>/dev/null || true; rm -f "$old"; }
done

# --- .gitignore -------------------------------------------------------------
touch .gitignore
add_ignore() {
  if [ -s .gitignore ] && [ -n "$(tail -c1 .gitignore)" ]; then echo >> .gitignore; fi
  grep -qxF "$1" .gitignore || echo "$1" >> .gitignore
}
for line in "CLAUDE.local.md" ".claude/settings.local.json" ".claude/worktrees/" ".DS_Store"; do
  add_ignore "$line"
done

# The shared Claude files must not be gitignored (some projects ignore .claude/
# completely, in their own .gitignore or in a global git ignore file).
claude_ignored() {
  local p
  for p in .claude/settings.json .claude/hooks/save-session.sh .claude/memory/MEMORY.md \
           .claude/sessions/x.jsonl scripts/setup-claude.command; do
    git check-ignore -q --no-index "$p" 2>/dev/null && return 0
  done
  return 1
}
if claude_ignored; then
  rule="$(git check-ignore -v --no-index .claude/settings.json .claude/memory/MEMORY.md \
          .claude/sessions/x.jsonl scripts/setup-claude.command 2>/dev/null | head -1 | cut -f1)"
  echo
  warn "Your git ignore rules exclude the .claude folder (rule: $rule)."
  echo "  To version Claude's memory and sessions it has to be included. Personal files"
  echo "  (.claude/settings.local.json, CLAUDE.local.md) stay excluded."
  yes_no_y "Include the shared .claude files in git?" \
    || die "Stopped. Nothing was committed. Remove the .claude rule from your gitignore to continue."
  # Re-include .claude, then put the personal exclusions after it (later rules win)
  tmp="$(mktemp)"
  grep -vxF -e ".claude/settings.local.json" -e ".claude/worktrees/" -e "!/.claude/" -e "!/.claude/**" .gitignore > "$tmp" || true
  mv "$tmp" .gitignore
  for line in "!/.claude/" "!/.claude/**" ".claude/settings.local.json" ".claude/worktrees/" "!/scripts/" "!/scripts/*.command"; do
    if [ -s .gitignore ] && [ -n "$(tail -c1 .gitignore)" ]; then echo >> .gitignore; fi
    echo "$line" >> .gitignore
  done
  if claude_ignored; then
    die "The .claude folder is still ignored. Check this rule and remove it by hand:
$(git check-ignore -v --no-index .claude/settings.json .claude/memory/MEMORY.md 2>/dev/null | head -1)"
  fi
  ok "Included .claude in git (personal settings stay excluded)"
fi

big_count=0
while IFS= read -r f; do
  [ -n "$f" ] || continue
  f="${f#./}"
  git check-ignore -q "$f" 2>/dev/null && continue
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 && continue  # already tracked
  add_ignore "/$f"
  big_count=$((big_count + 1))
done <<EOF
$(find . -type f -size +"${LARGE_FILE_MB}"M -not -path './.git/*' 2>/dev/null)
EOF
ok ".gitignore"
[ "$big_count" -gt 0 ] && warn "$big_count file(s) over ${LARGE_FILE_MB} MB added to .gitignore (not uploaded)"

# --- Auto memory location (machine-specific, gitignored) ---------------------
local_settings=".claude/settings.local.json"
tmp="$(mktemp)"
if [ -f "$local_settings" ]; then
  jq --arg d "$root/.claude/memory" '.autoMemoryDirectory = $d' "$local_settings" > "$tmp"
else
  jq -n --arg d "$root/.claude/memory" '{autoMemoryDirectory: $d}' > "$tmp"
fi
mv "$tmp" "$local_settings"
ok "Auto memory -> .claude/memory/"

# --- Import what Claude Code stored for this folder before the setup ----------
# Claude Code keeps memory and transcripts in <config>/projects/<encoded path>/,
# where the path's non-alphanumeric characters become "-".
cfg="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
old_dir=""
imported=""
for p in "$root" "$(pwd -P)"; do
  d="$cfg/projects/$(printf '%s' "$p" | sed 's/[^a-zA-Z0-9]/-/g')"
  [ -d "$d" ] && { old_dir="$d"; break; }
done

if [ -n "$old_dir" ]; then
  # Memory: copy topic files that aren't in the repo yet, merge the index.
  # A file with the same name but different content (e.g. written on another
  # computer) is kept as <name>-from-<this computer>.md, so nothing is lost.
  mem_new=0
  mem_renamed=0
  if [ -d "$old_dir/memory" ] && ! [ "$old_dir/memory" -ef .claude/memory ]; then
    host="$(hostname -s 2>/dev/null || hostname)"
    host="$(printf '%s' "$host" | tr -c 'a-zA-Z0-9-' '-')"
    renames="$(mktemp)"
    for f in "$old_dir"/memory/*; do
      [ -f "$f" ] || continue
      b="$(basename "$f")"
      [ "$b" = MEMORY.md ] && continue
      if [ ! -e ".claude/memory/$b" ]; then
        cp "$f" ".claude/memory/$b"
        mem_new=$((mem_new + 1))
      elif ! cmp -s "$f" ".claude/memory/$b"; then
        nb="${b%.md}-from-$host.md"
        if ! cmp -s "$f" ".claude/memory/$nb" 2>/dev/null; then
          cp "$f" ".claude/memory/$nb"
          mem_new=$((mem_new + 1))
          mem_renamed=$((mem_renamed + 1))
        fi
        printf '%s\t%s\n' "$b" "$nb" >> "$renames"
      fi
    done
    if [ -f "$old_dir/memory/MEMORY.md" ]; then
      touch .claude/memory/MEMORY.md
      while IFS= read -r line || [ -n "$line" ]; do
        [ -n "$line" ] || continue
        # Point index lines at the renamed copies
        while IFS="$(printf '\t')" read -r o n; do
          line="$(printf '%s' "$line" | awk -v o="($o)" -v n="($n)" '{
            out = ""; s = $0
            while ((i = index(s, o)) > 0) { out = out substr(s, 1, i - 1) n; s = substr(s, i + length(o)) }
            print out s }')"
        done < "$renames"
        if ! grep -qxF -- "$line" .claude/memory/MEMORY.md; then
          if [ -s .claude/memory/MEMORY.md ] && [ -n "$(tail -c1 .claude/memory/MEMORY.md)" ]; then echo >> .claude/memory/MEMORY.md; fi
          printf '%s\n' "$line" >> .claude/memory/MEMORY.md
          mem_new=$((mem_new + 1))
        fi
      done < "$old_dir/memory/MEMORY.md"
    fi
    rm -f "$renames"
  fi
  if [ "$mem_renamed" -gt 0 ]; then
    warn "$mem_renamed memory file(s) existed in the project with different content."
    warn "This computer's versions were kept as *-from-$host.md. Ask Claude to merge them sometime."
  fi
  if [ "$mem_new" -gt 0 ]; then
    ok "Imported earlier Claude memory into .claude/memory/"
    imported="memory"
  fi

  # Sessions: offer to copy past transcripts that aren't in the repo yet
  list="$(mktemp)"
  for f in "$old_dir"/*.jsonl; do
    [ -f "$f" ] || continue
    [ -e ".claude/sessions/$(basename "$f")" ] || printf '%s\n' "$f" >> "$list"
  done
  if [ -s "$list" ]; then
    n_sess="$(wc -l < "$list" | tr -d ' ')"
    size="$(tr '\n' '\0' < "$list" | xargs -0 du -ck | tail -1 | awk '{printf "%.0f MB", $1/1024}')"
    echo
    echo "Found $n_sess earlier Claude Code session(s) for this folder on this computer (about $size)."
    echo "They contain everything Claude saw in them: file contents, command output, anything pasted in."
    echo "Exit any Claude session still running in this project first, or its latest part won't be included."
    if yes_no "Copy them into .claude/sessions/ so they're versioned and shared?"; then
      while IFS= read -r f; do cp "$f" .claude/sessions/; done < "$list"
      ok "Imported $n_sess session(s) into .claude/sessions/"
      imported="${imported:+$imported and }$n_sess session(s)"
    fi
  fi
  rm -f "$list"
fi

# --- Hook that saves session transcripts into the repo -----------------------
cat > .claude/hooks/save-session.sh <<'EOF'
#!/usr/bin/env bash
# Copies the finished session's transcript into the repo.
src=$(jq -r '.transcript_path')
dest_dir="$CLAUDE_PROJECT_DIR/.claude/sessions"
mkdir -p "$dest_dir"
dest="$dest_dir/$(basename "$src")"
# When the session was resumed straight from the repo file, src and dest are the same
[ "$src" -ef "$dest" ] || cp "$src" "$dest"
EOF
chmod +x .claude/hooks/save-session.sh

shared=".claude/settings.json"
hook_cmd='"$CLAUDE_PROJECT_DIR"/.claude/hooks/save-session.sh'
[ -f "$shared" ] || echo '{}' > "$shared"
if ! jq -e --arg c "$hook_cmd" '[.hooks.SessionEnd[]?.hooks[]?.command] | index($c)' "$shared" >/dev/null; then
  tmp="$(mktemp)"
  jq --arg c "$hook_cmd" '.hooks.SessionEnd += [{hooks: [{type: "command", command: $c, timeout: 15}]}]' "$shared" > "$tmp"
  mv "$tmp" "$shared"
fi
# SessionEnd hooks get only 1.5 seconds by default; give large transcripts time to copy
tmp="$(mktemp)"
jq --arg c "$hook_cmd" '(.hooks.SessionEnd[]?.hooks[]? | select(.command == $c and (.timeout // 0) < 15) | .timeout) = 15' "$shared" > "$tmp"
if cmp -s "$tmp" "$shared"; then rm -f "$tmp"; else mv "$tmp" "$shared"; fi
ok "Session hook"

# --- Resume helper (double-clickable) -------------------------------------------
cat > scripts/claude-resume.command <<'EOF'
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
EOF
chmod +x scripts/claude-resume.command
ok "scripts/claude-resume.command"

# --- Commit -------------------------------------------------------------------
add_or_die() {
  git add "$@" 2>/dev/null || die "git couldn't add: $*
Check whether a gitignore rule excludes it:  git check-ignore -v $*"
}
add_or_die .gitignore .claude/settings.json .claude/hooks scripts
if [ -n "$(ls -A .claude/memory 2>/dev/null)" ]; then add_or_die .claude/memory; fi
if [ -n "$(ls -A .claude/sessions 2>/dev/null)" ]; then add_or_die .claude/sessions; fi
if [ -f CLAUDE.md ]; then add_or_die CLAUDE.md; fi

other="$(git ls-files -o -m --exclude-standard)"
if [ -n "$other" ]; then
  n_files="$(printf '%s\n' "$other" | wc -l | tr -d ' ')"
  size="$(git ls-files -z -o -m --exclude-standard | xargs -0 du -ck 2>/dev/null | tail -1 | awk '{printf "%.0f MB", $1/1024}')"
  echo
  echo "Your other project files (new or changed, not part of the Claude setup):"
  echo "$n_files file(s), about $size:"
  git status --short --untracked-files=all | grep -v '^[AMDR] ' | head -25 | sed 's/^/    /'
  [ "$n_files" -gt 25 ] && echo "    ... and $((n_files - 25)) more (see: git status)"
  if yes_no "Commit and upload them too?"; then git add -A; fi
fi

if ! git diff --cached --quiet; then
  msg="Set up Claude Code project"
  [ -n "$imported" ] && msg="$msg, import earlier Claude $imported"
  git commit -q -m "$msg"
  ok "Committed: $msg"
fi

# --- Sync with GitLab -----------------------------------------------------------
# Sync the current branch only with its own counterpart on GitLab (its upstream,
# or the GitLab branch of the same name). A branch that doesn't exist on GitLab
# yet is simply pushed as a new branch.
cur_branch="$(git symbolic-ref --short -q HEAD || true)"
sync_ref=""
if git remote get-url origin >/dev/null 2>&1 && has_commits && [ -n "$cur_branch" ]; then
  if up="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"; then
    sync_ref="$up"
  elif git rev-parse -q --verify "refs/remotes/origin/$cur_branch" >/dev/null; then
    sync_ref="origin/$cur_branch"
  fi
fi
if [ -n "$sync_ref" ]; then
  if ! git merge-base --is-ancestor "$sync_ref" HEAD 2>/dev/null; then
    # Put local commits on top of what's on GitLab. A .gitignore conflict is
    # resolved by keeping the lines from both sides; anything else stops.
    if ! git rebase -q --autostash "$sync_ref" >/dev/null 2>&1; then
      rounds=0
      while [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; do
        rounds=$((rounds + 1))
        conflicts="$(git diff --name-only --diff-filter=U)"
        if [ "$conflicts" = ".gitignore" ] && [ "$rounds" -le 20 ]; then
          { git show :2:.gitignore 2>/dev/null || true; git show :3:.gitignore 2>/dev/null || true; } \
            | awk '!seen[$0]++' > .gitignore.merged
          mv .gitignore.merged .gitignore
          git add .gitignore
          GIT_EDITOR=true git rebase --continue >/dev/null 2>&1 || true
        else
          git rebase --abort
          die "Your files conflict with what's already in the GitLab project: $conflicts
Resolve it by hand:  git pull --rebase"
        fi
      done
    fi
    ok "Combined with the history already in GitLab"
  fi
fi

if git remote get-url origin >/dev/null 2>&1 && has_commits && [ -z "$cur_branch" ]; then
  warn "No branch is checked out (detached HEAD), so nothing was pushed. Check out a branch and run the script again."
elif git remote get-url origin >/dev/null 2>&1 && has_commits; then
  git push -q -u origin HEAD || die "Push to GitLab failed. Your work is committed locally; fix the problem above and run the script again."
  ok "Pushed to GitLab"
else
  warn "Not connected to GitLab, so nothing was pushed. Run the script again with the project's SSH URL to connect it."
fi

echo
echo "Done. Your project is in:"
echo "  $root"
echo "Start Claude there with:  cd \"$root\" && claude"
echo "Continue your latest session (also on another machine): double-click scripts/claude-resume.command"
