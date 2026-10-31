#!/usr/bin/env bash
# Classify a GitHub event diff; errors leave the workflow's test job enabled.
set -euo pipefail

: "${BASE_SHA:?}" "${HEAD_SHA:?}" "${GITHUB_OUTPUT:?}"

if [[ "$BASE_SHA" =~ ^0+$ ]]; then
  : "${DEFAULT_BRANCH:?}" "${GITHUB_REF:?}"
  # An initial default-branch push has no prior baseline to classify safely.
  if [[ "$GITHUB_REF" == "refs/heads/$DEFAULT_BRANCH" ]]; then
    echo "run_tests=true" >> "$GITHUB_OUTPUT"
    exit 0
  fi
  # New feature branches have no before SHA; compare their actual branch changes.
  BASE_SHA=$(git merge-base "$HEAD_SHA" "origin/$DEFAULT_BRANCH")
fi

# Include both paths when code is renamed to a documentation suffix.
changed_files=$(git diff --no-renames --name-only "$BASE_SHA" "$HEAD_SHA" --)
run_tests=false
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  case "$path" in
    *.md|*.pdf|*.typ|*.tex|*.latex|*.ltx) ;;
    *) run_tests=true; break ;;
  esac
done <<< "$changed_files"
echo "run_tests=$run_tests" >> "$GITHUB_OUTPUT"
