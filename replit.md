# Minecraft Bedrock Seed Searcher

## Overview
A brute-force structural 32-bit seed searcher for Minecraft Bedrock Edition,
with optional Java Edition biome filtering powered by the
[cubiomes](https://github.com/Cubitect/cubiomes) C library.

## Architecture
- **main.py** — entry point; interactive CLI that collects parameters, runs the
  structure search loop, and optionally applies biome filtering
- **biome.py** — Python ctypes bindings for cubiomes; exposes `BiomeGenerator`
  and interactive `prompt_biome_requirements()` helper
- **cubiomes/** — compiled C library (`libcubiomes.so`) providing fast biome
  generation for all Java Edition versions from 1.7 through 1.21

## Key Files
| File | Purpose |
|---|---|
| `main.py` | Seed search loop + biome integration |
| `biome.py` | cubiomes Python bindings & user-facing helpers |
| `cubiomes/libcubiomes.so` | Compiled cubiomes shared library |
| `cubiomes/generator.h` | cubiomes Generator API reference |

## Dependencies
Managed via Poetry (`pyproject.toml`):
- `numba` — JIT compilation for Mersenne Twister RNG (fast structure pos calc)
- `numpy` — array support for numba
- `matplotlib`, `sympy`, `pygame` — available for future visualisation work

## How to Run
The workflow runs `python3 main.py`.  The app is interactive — it prompts for:
1. Seed range (`SeedStart`, `SeedEnd`)
2. Structure RNG constants (Spacing, Separation, Salt, Linear Separation)
3. Search radius & minimum occurrence count
4. Output file name (defaults to `seed_results.txt`)
5. Optional biome filtering (MC version, dimension, coordinate/biome pairs)

## Biome Filtering Details
- Uses cubiomes' `getBiomeAt()` (C, very fast) after the structure filter narrows
  the candidate set
- Supports all biomes from MC 1.7 through 1.21 plus common aliases
- For pre-1.18 seeds: y coordinate is ignored (2D layer-based generation)
- For 1.18+ seeds: y=64 (sea level) is the default

## Structure RNG Constants Reference
| Structure | Spacing | Separation | Salt | Linear |
|---|---|---|---|---|
| Bastion/Fortress | 30 | 4 | 30084232 | 0 |
| Village | 34 | 8 | 10387312 | 1 |
| Pillager Outpost | 80 | 24 | 165745296 | 1 |
| Woodland Mansion | 80 | 20 | 10387319 | 1 |
| Ocean Monument | 32 | 5 | 10387313 | 1 |
| Shipwreck | 24 | 4 | 165745295 | 1 |
| Ruined Portal | 40 | 15 | 40552231 | 0 |
| Other Overworld | 32 | 8 | 14357617 | 0 |
