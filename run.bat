@echo off
REM MC-Seed-Search Launcher for Windows
REM Run this batch file to start the Minecraft Bedrock Structure Seed Searcher
REM Requires Python 3.12+ to be installed and in PATH

cd /d "%~dp0"
python MC_SeedSearch.py %*
pause
