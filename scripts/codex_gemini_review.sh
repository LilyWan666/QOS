#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
SKILL_SCRIPT="${GEMINI_REVIEW_SKILL_SCRIPT:-${HOME}/.agents/skills/gemini-review/scripts/gemini_review.sh}"

if [[ ! -x "${SKILL_SCRIPT}" ]]; then
  echo "[ERR] Gemini review skill script is missing or not executable: ${SKILL_SCRIPT}" >&2
  exit 127
fi

export GEMINI_REVIEW_OUT="${GEMINI_REVIEW_OUT:-${REPO_ROOT}/temp/gemini_review.md}"
if [[ -z "${GEMINI_KEY_FILE:-}" ]]; then
  echo "[ERR] Set GEMINI_KEY_FILE to a local Gemini API key file before running review." >&2
  exit 2
fi
export GEMINI_KEY_FILE
export GEMINI_REVIEW_MODEL="${GEMINI_REVIEW_MODEL:-gemini-3-flash-preview}"

mkdir -p "$(dirname "${GEMINI_REVIEW_OUT}")"
cd "${REPO_ROOT}"

"${SKILL_SCRIPT}"
