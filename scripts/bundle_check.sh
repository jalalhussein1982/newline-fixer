#!/usr/bin/env bash
# Produce the submission bundle and prove it is self-contained: verify, clone, check, build, serve.
set -euo pipefail
OUT_DIR="${1:-${TMPDIR:-/tmp}}"
BUNDLE="$OUT_DIR/jalal-hussein.bundle"
CLONE="$(mktemp -d "${TMPDIR:-/tmp}/newline-fixer-bundle-check.XXXXXX")"
trap 'rm -rf "$CLONE"' EXIT

git bundle create "$BUNDLE" --all
git bundle verify "$BUNDLE"
git clone --quiet "$BUNDLE" "$CLONE/repo"
cd "$CLONE/repo"
echo "branches in the bundle:"; git branch -a
echo "first commits:"; git log --reverse --format='%h %s' | head -3
first_paths="$(git log --reverse --format='%h' | head -3 | xargs -I{} git show --name-only --format= {} | sort -u)"
# the very first commit also adds the root README.md (a planning-era stub), so it is allowed beside docs/
if echo "$first_paths" | grep -vE '^(docs/|README\.md$)' ; then echo "FAIL: the first commits touch files outside docs/ and README.md"; exit 1; fi
uv sync --all-extras
make check
docker build -t newline-fixer:bundle-check .
uv run python scripts/container_check.py --image newline-fixer:bundle-check --no-build
docker rmi newline-fixer:bundle-check >/dev/null
echo "PASS: $BUNDLE ($(du -h "$BUNDLE" | cut -f1)) clones, checks, builds and serves"
