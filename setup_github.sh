#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# setup_github.sh
# Run this script to initialise the git repo and push to GitHub.
#
# Usage:
#   chmod +x setup_github.sh
#   ./setup_github.sh
# ─────────────────────────────────────────────────────────────────

echo ""
echo "========================================"
echo "  Diabetes Prediction App - GitHub Setup"
echo "========================================"
echo ""

# Prompt for GitHub username and repo name
read -p "Enter your GitHub username: " GITHUB_USER
read -p "Enter your new repo name (e.g. diabetes-prediction-app): " REPO_NAME

echo ""
echo ">> Initialising git repository..."
git init
git add .
git commit -m "Initial commit: Diabetes prediction Streamlit app"
git branch -M main

echo ""
echo ">> Next steps:"
echo ""
echo "   1. Go to https://github.com/new"
echo "   2. Create a NEW repository named: $REPO_NAME"
echo "      (leave it empty - no README, no .gitignore)"
echo "   3. Come back here and run:"
echo ""
echo "      git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
echo "      git push -u origin main"
echo ""
echo "   4. To deploy for free on Streamlit Cloud:"
echo "      https://share.streamlit.io -> New app -> pick this repo -> app.py"
echo ""
echo "Done! Follow the steps above to push."
