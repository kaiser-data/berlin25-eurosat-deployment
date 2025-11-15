#!/bin/bash

# ==========================================
# 🔒 Safe Push to zoezyn/decentralized
# ==========================================
#
# This script ONLY pushes to the model-bit-size branch
# Does NOT affect main or any other branches
#
# Usage: ./push_to_zoezyn.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

REMOTE_REPO="https://github.com/zoezyn/decentralized.git"
TARGET_BRANCH="model-bit-size"
REMOTE_NAME="zoezyn-repo"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     🔒 Safe Push to zoezyn/decentralized                  ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Safety check: confirm current repo
CURRENT_REPO=$(git remote get-url origin 2>/dev/null || echo "none")
echo -e "${YELLOW}Current repo:${NC} $CURRENT_REPO"
echo -e "${YELLOW}Target repo:${NC} $REMOTE_REPO"
echo -e "${YELLOW}Target branch:${NC} $TARGET_BRANCH"
echo ""

# Confirm with user
echo -e "${RED}⚠️  This will FORCE PUSH to zoezyn/decentralized:$TARGET_BRANCH${NC}"
echo -e "${YELLOW}This will OVERWRITE the remote branch with your local code.${NC}"
echo ""
read -p "Are you sure you want to continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo -e "${RED}Aborted.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}→ Adding remote if not exists...${NC}"

# Add remote if it doesn't exist
if git remote | grep -q "^${REMOTE_NAME}$"; then
    echo "  Remote '$REMOTE_NAME' already exists"
    git remote set-url $REMOTE_NAME $REMOTE_REPO
else
    git remote add $REMOTE_NAME $REMOTE_REPO
    echo "  Remote '$REMOTE_NAME' added"
fi

echo ""
echo -e "${GREEN}→ Fetching remote branches...${NC}"
git fetch $REMOTE_NAME

echo ""
echo -e "${GREEN}→ Current branch info:${NC}"
echo "  Local branch: $(git branch --show-current)"
echo "  Latest commit: $(git log -1 --oneline)"

echo ""
echo -e "${YELLOW}→ Pushing to $REMOTE_NAME:$TARGET_BRANCH...${NC}"

# Force push ONLY to the specific branch
git push $REMOTE_NAME HEAD:$TARGET_BRANCH --force

echo ""
echo -e "${GREEN}✅ Successfully pushed to $TARGET_BRANCH!${NC}"
echo ""
echo -e "${BLUE}Branch URL:${NC} https://github.com/zoezyn/decentralized/tree/$TARGET_BRANCH"
echo ""
echo -e "${YELLOW}Note: Only the '$TARGET_BRANCH' branch was affected.${NC}"
echo -e "${YELLOW}Main branch and other branches remain unchanged.${NC}"
echo ""
