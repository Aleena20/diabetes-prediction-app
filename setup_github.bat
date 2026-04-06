@echo off
echo.
echo ========================================
echo   Diabetes Prediction App - GitHub Setup
echo ========================================
echo.

set /p GITHUB_USER="Enter your GitHub username: "
set /p REPO_NAME="Enter your new repo name (e.g. diabetes-prediction-app): "

echo.
echo ^>^> Initialising git repository...
git init
git add .
git commit -m "Initial commit: Diabetes prediction Streamlit app"
git branch -M main

echo.
echo ^>^> Next steps:
echo.
echo    1. Go to https://github.com/new
echo    2. Create a NEW repo named: %REPO_NAME%
echo       (leave it empty - no README, no .gitignore)
echo    3. Run these two commands:
echo.
echo       git remote add origin https://github.com/%GITHUB_USER%/%REPO_NAME%.git
echo       git push -u origin main
echo.
echo    4. Deploy free on Streamlit Cloud:
echo       https://share.streamlit.io -^> New app -^> pick this repo -^> app.py
echo.
pause
