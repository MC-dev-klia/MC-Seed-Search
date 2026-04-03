# Running MC-Seed-Search

## Quick Start

### Linux/Mac Users
```bash
./run.sh
```

### Windows Users
```bash
run.bat
```

### Any Platform (with Python installed)
```bash
python3 MC_SeedSearch.py
```
or
```bash
python MC_SeedSearch.py
```

## Installation Requirements

Before running, ensure you have Python 3.12+ with the required dependencies installed:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numba numpy
```

## What You Need

- **Python 3.12+** - Download from [python.org](https://www.python.org/downloads/)
- **Numba** - JIT compiler for fast structure searching
- **NumPy** - Numerical computing library  
- **Cubiomes library** - Biome noise generation (included in the repository)

## Troubleshooting

### "Command not found: python3" (Linux/Mac)
- Install Python: `sudo apt-get install python3` (Ubuntu/Debian) or `brew install python` (Mac)

### "Python is not recognized" (Windows)
- Install Python from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation
- Restart your terminal after installation

### "ModuleNotFoundError: No module named 'numba'" 
- Install dependencies: `pip install -r requirements.txt`

### Script won't run (Permission denied on Linux/Mac)
- Make it executable: `chmod +x run.sh`
- Then run: `./run.sh`

## Files

- `MC_SeedSearch.py` - Python entry point (can be executed directly)
- `run.sh` - Shell script launcher for Linux/Mac
- `run.bat` - Batch file launcher for Windows
- `main.py` - Core application (called by entry points)
- `requirements.txt` - Python dependencies
