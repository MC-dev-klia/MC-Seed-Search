#!/bin/bash
# MC-Seed-Search Launcher for Linux/Mac
# Run this script to start the Minecraft Bedrock Structure Seed Searcher
# Requires Python 3.12+ to be installed

cd "$(dirname "$0")"
python3 MC_SeedSearch.py "$@"
