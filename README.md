# Quick Start
Go to the Releases section at https://github.com/MC-dev-klia/MC-Seed-Search/releases/tag/v2.0.0 and download the latest .exe file. No additional setup is required.
# Minecraft Bedrock Structure Seed Searcher - Technical Guide

This tool brute-forces Minecraft Bedrock seeds to find worlds where structures generate at specific coordinates. It implements the exact same RNG algorithms Minecraft uses for structure placement.

### Structure Generation Process
Minecraft uses a deterministic process to place structures:

1. **World Seed**: 64-bit integer initializing all generation
2. **Structure Seed**: Lower 32 bits of world seed used to generate structural positions
3. **Region Division**: World divided into overlapping regions of size `spacing × spacing` chunks
4. **Region Positioning**: 4 regions centered at chunk coordinates (0,0), (-spacing,0), (0,-spacing), (-spacing,-spacing)
5. **Per-Region RNG**: Each region gets unique seed: `world_seed + rx * 341873128712 + rz * 132897987541 + salt`
6. **Position Calculation**: RNG determines chunk offset within region bounds
7. **Biome Validation**: Structure generates only if biome requirements met

### RNG Constants
Each structure type has hardcoded parameters:
- **Spacing**: The width/length of a region
- **Separation**: The size of the separation zone in a region
- **Salt**: Unique RNG identifier that makes each structure type generate differently
- **Linear Separation**: Boolean controlling type of placement in region
- **Offset**: The position inside a chunk the structure starts generating from

### Region System --- MOST IMPORTANT TO UNDERSTAND
The world is divided into overlapping regions for structure placement. Each region is `spacing × spacing` chunks in size and attempts to place exactly one structure. The four regions checked are centered at chunk coordinates (0,0), (-spacing,0), (0,-spacing), and (-spacing,-spacing).

Within each region, structures can only generate in the "allowed zone" - the area that avoids the separation buffer from region edges. The "dead zone" is the separation buffer where structures cannot generate.

The four regions used to test can fit in these spaces:
AND means x and z have to both satisfy
OR means at least one of x and z has to satisfy
```
Region (0,0):  
    Overall: chunks 0 to spacing on both x and z
    Dead Zone: chunks spacing - separation to spacing on x OR z
    Allowed Zone: chunks 0 to spacing - separation on both x and z
Region (-1,0):
    Overall: chunks -spacing to 0 on x AND chunks 0 to spacing on z
    Dead Zone: chunks -separation to 0 on x OR spacing - separation to spacing on z
    Allowed Zone: chunks -spacing to -separation on x AND chunks 0 to spacing - separation on z
Region (0,-1):
    Overall: chunks 0 to spacing on x AND chunks -spacing to 0 on z
    Dead Zone: chunks -separation to 0 on z OR spacing - separation to spacing on x
    Allowed Zone: chunks 0 to spacing - separation on x AND chunks -spacing to -separation on z
Region (-1,-1):  
    Overall: chunks -spacing to 0 on both x and z
    Dead Zone: chunks -separation to 0 on x OR z
    Allowed Zone: chunks -spacing to -separation on both x and z
```

Each region attempts exactly one structure placement at:
- **Chunk coordinates**: `(rx * spacing + random_offset, rz * spacing + random_offset)` where rx and rz are region coordinates
- **Block coordinates**: `chunk_coords * 16 + offset`
- **Offset**: Usually (8,8) for chunk center placement

## Other specifics

### Bounds Specification
- **Radius**: Square search area (±N blocks from origin)
- **Box**: Rectangular search area (x1,z1 to x2,z2)
- **Closest**: Optimized search for minimum-distance structures using mathematical bounds with custom error setup
- **Region-Specific**: Targeting specific regions OR different bounding box for each region

### Occurrence Requirements
- **Min Occurrence**: Number of structures required (1-4)
- **Region Logic**: When <4, allows specifying which regions to check
- **Position Constraints**: Exact coordinates or ranges within regions

### World Coordinates
- Standard Minecraft coordinates (X, Z)
- Origin at (0, 0) - spawn chunks
- Structures generate at chunk coordinates × 16 + offset

### Chunk vs Block Coordinates
- **Chunks**: 16×16 block areas
- **Structure Position**: Chunk coordinates + offset (usually 8,8)
- **Validation**: Biome checks at block level (chunk × 16 + offset)

### Multi-Constraint Searching
Combines multiple structure requirements:
- Primary constraint uses fast JIT-compiled kernel
- Secondary constraints checked in Python (slower)
- Order constraints by selectivity for optimal performance

### Region-Specific Configuration
- **Independent Biome Filters**: Different biomes per region
- **Position Ranges**: x1,z1-x2,z2 bounding boxes per region
- **Custom Offsets**: Exact position inside a chunk

### Search Strategies
1. **Structure-Only**: Fastest, no biome validation
2. **32-bit Expansion**: Best for biome-constrained searches using a limited upper-32-bit expansion
3. **Standard Scan**: When expansion mode isn't applicable
4. **Region Specification**: When you are targeting specific regions

### Coordinate Precision
- **Chunk Accuracy**: All calculations in chunk coordinates
- **Block Conversion**: ×16 for world coordinates
- **Floating Point**: No floating point in core algorithms (integer math only)

### Validation
- All algorithms verified against Minecraft Bedrock 1.21
- Structure positions match game most of the time
- Biome compatibility tested against official generation, may fail on biome edges or due to other factors
- On edge cases, use https://www.chunkbase.com/apps/seed-map to verify false positives.

### Common Issues
- **No Seeds Found**: Increase search range or relax constraints, and check for impossibility
- **Slow Performance**: Check the probability of succeeding and if it doesn't match expected, something is wrong, otherwise this is because 2^64 is a huge search space
- **Memory Errors**: Reduce search range or use file output

## Dependencies
- **Numba JIT**: Compiles structure search kernels
- **Cubiomes Library**: Biome noise generation
- **Python 3.12+**: Modern Python features
- **MCBE-1.18-Seed-Finder**: Helped create variant filters
