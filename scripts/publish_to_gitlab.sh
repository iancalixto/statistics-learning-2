#!/usr/bin/env bash
# Publish Gitlab_Lecture to FH Kufstein GitLab.
# Requires: glab (brew install glab), GITLAB_TOKEN with api scope.
#
# Usage:
#   export GITLAB_TOKEN="glpat-..."
#   ./scripts/publish_to_gitlab.sh

set -euo pipefail

HOST="gitlab.web.fh-kufstein.ac.at"
API="https://${HOST}/api/v4"
USER="calixtoian"
PROJECT_NAME="Statistics Learning 2 - Semester 2"
PROJECT_PATH="statistics-learning-2-semester-2"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

GLAB="$(command -v glab || echo /opt/homebrew/bin/glab)"

if [[ -z "${GITLAB_TOKEN:-}" ]]; then
  echo "ERROR: Set GITLAB_TOKEN (Personal Access Token from ${HOST})."
  echo "  Settings → Access Tokens → api, read_repository, write_repository"
  exit 1
fi

cd "$REPO_ROOT"

echo "==> Authenticating glab for ${HOST}"
"$GLAB" auth login --hostname "$HOST" --token "$GITLAB_TOKEN" 2>/dev/null || true

echo "==> Ensuring git repository"
if [[ ! -d .git ]]; then
  git init -b main
fi

if ! git rev-parse HEAD >/dev/null 2>&1; then
  git add -A
  git commit -m "$(cat <<'EOF'
Initial commit: Statistics Learning 2 course materials.

Notebooks, lecture PDFs, requirements, and probabilistic view tutorial.
EOF
)"
fi

PROJECT_ID=""
if curl -sf --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
  "${API}/projects/${USER}%2F${PROJECT_PATH}" >/dev/null 2>&1; then
  echo "==> Project already exists: ${USER}/${PROJECT_PATH}"
  PROJECT_ID=$(curl -sf --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    "${API}/projects/${USER}%2F${PROJECT_PATH}" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
else
  echo "==> Creating project: ${PROJECT_NAME}"
  JSON_BODY=$(PROJECT_NAME="$PROJECT_NAME" PROJECT_PATH="$PROJECT_PATH" python3 -c '
import json, os
print(json.dumps({
    "name": os.environ["PROJECT_NAME"],
    "path": os.environ["PROJECT_PATH"],
    "visibility": "private",
    "initialize_with_readme": False,
    "default_branch": "main",
}))
')
  RESPONSE=$(curl -sf --request POST --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
    --header "Content-Type: application/json" \
    --data "$JSON_BODY" \
    "${API}/projects")
  PROJECT_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
  echo "    Created project id=${PROJECT_ID}"
fi

ENC_PASS=$(python3 -c "import os, urllib.parse; print(urllib.parse.quote(os.environ['GITLAB_TOKEN'], safe=''))")
ENC_USER=$(python3 -c "import os, urllib.parse; print(urllib.parse.quote(os.environ.get('GITLAB_USER', '${USER}'), safe=''))")
REMOTE_URL="https://${ENC_USER}:${ENC_PASS}@${HOST}/${USER}/${PROJECT_PATH}.git"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

echo "==> Pushing main"
git push -u origin main

echo "==> Creating and pushing development branch"
git checkout -B development
git push -u origin development

echo "==> Protecting main branch (no direct push)"
curl -sf --request POST --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"name":"main","push_access_level":0,"merge_access_level":40,"allow_force_push":false,"code_owner_approval_required":false}' \
  "${API}/projects/${PROJECT_ID}/protected_branches" 2>/dev/null \
  || curl -sf --request PATCH --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
  "${API}/projects/${PROJECT_ID}/protected_branches/main" 2>/dev/null \
  || echo "    (main may already be protected — check GitLab UI)"

echo "==> Setting default branch to main"
curl -sf --request PUT --header "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
  --header "Content-Type: application/json" \
  --data '{"default_branch":"main"}' \
  "${API}/projects/${PROJECT_ID}" >/dev/null

echo ""
echo "Done."
echo "  Project: https://${HOST}/${USER}/${PROJECT_PATH}"
echo "  main:        protected (push disabled)"
echo "  development: active working branch (checked out locally)"
echo ""
echo "Work on development:  git checkout development"
