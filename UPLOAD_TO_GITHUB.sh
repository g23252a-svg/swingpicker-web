#!/usr/bin/env bash
set -euo pipefail

# SwingPicker v26 Loss Defense — safe GitHub upload
# Run from Git Bash with: bash UPLOAD_TO_GITHUB.sh

cd "$(dirname "$0")"

BRANCH="codex/v26-loss-defense"
REMOTE="origin"

PATCH_FILES=(
  .gitignore
  README.md
  UPLOAD_TO_GITHUB.sh
  components/decision_center.py
  components/page_briefing.py
  components/tab_market.py
  daily_briefing.py
  docs/LOSS_DEFENSE_V26.md
  main.py
  ml_engine.py
  pipeline_finalize.py
  pytest.ini
  scoring_engine.py
  scripts/analyze_realized_edge.py
  services/data_store.py
  services/portfolio_swap.py
  services/recommendation_quality.py
  telegram_sender.py
  telegram_sender_v2.py
  tests/test_decision_center.py
  tests/test_ml_reliability.py
  tests/test_recommendation_quality.py
  tests/test_safe_notifications.py
  version_info.py
)

echo "[1/5] Checking repository..."
git rev-parse --is-inside-work-tree >/dev/null
git remote get-url "$REMOTE"

echo "[2/5] Refreshing GitHub main..."
git fetch "$REMOTE" main

# Local UI/ML checks may update this tracked sidecar's final newline. It is not
# part of the production patch, so restore it before staging the explicit list.
git restore -- data/feature_cache_schema.json 2>/dev/null || true

echo "[3/5] Creating patch branch: $BRANCH"
git switch -C "$BRANCH"

echo "[4/5] Staging only the reviewed v26 files..."
git add -- "${PATCH_FILES[@]}"

if git diff --cached --quiet; then
  echo "No new patch changes to commit. Continuing to push the branch."
else
  git commit -m "feat: add v26 loss-defense recommendation system"
fi

echo "[5/5] Pushing to GitHub..."
git push -u "$REMOTE" "$BRANCH"

echo
echo "Upload complete. Open the pull request page:"
echo "https://github.com/g23252a-svg/swingpicker-web/compare/main...codex/v26-loss-defense?expand=1"
echo
echo "After merging the PR into main, verify the Railway deployment and the Today tab."
