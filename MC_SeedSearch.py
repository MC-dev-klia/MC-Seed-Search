#!/usr/bin/env python3
"""
MC-Seed-Search Launcher
Minecraft Bedrock Structure Seed Searcher

This script can be run directly if on your PATH or executed as:
    python MC_SeedSearch.py
    ./MC_SeedSearch.py (on Linux/Mac)
    python MC_SeedSearch.py (on Windows)
"""

import sys
import os

# Add the script directory to Python path to find local modules
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Now import and run the main application
if __name__ == "__main__":
    import main
