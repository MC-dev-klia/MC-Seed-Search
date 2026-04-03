# Minecraft Bedrock Seed Searcher - User Guide (AI Generated)

This tool helps you find Minecraft seeds that contain specific structures (like villages, temples, or mansions) at desired locations. This guide explains all the technical terms you'll encounter when using the tool.

## Quick Start

If you're new to Minecraft world generation, here's what you need to know:

1. **Minecraft Seeds**: Every Minecraft world is generated from a number called a "seed". Different seeds create different worlds.
2. **Structures**: These are special buildings or features that generate in specific locations (villages, temples, mansions, etc.)
3. **Seed Searching**: This tool tries many different seeds to find ones where your desired structures appear where you want them.

## Key Concepts

### Seeds and Ranges

- **SeedStart/SeedEnd**: The range of seed numbers to search through. Minecraft seeds are 64-bit numbers, but this tool can search efficiently through billions of possibilities.
- **48-bit structure + 16-bit biome expansion**: An advanced mode that searches structure placement first (48 bits) then tests biome compatibility (16 bits) for much faster searching when you have biome requirements.

### Structures

Structures are special generated features in Minecraft. Each type has specific rules for where and how often they appear:

- **Village**: Small settlements with houses, farms, and villagers
- **Pillager Outpost**: Watchtowers with pillagers and illager banners
- **Woodland Mansion**: Large dark forest houses with illagers
- **Ocean Monument**: Underwater temples with guardians
- **Shipwreck**: Sunken boats, sometimes with treasure
- **Temples**: Desert Pyramids, Jungle Temples, Swamp Huts, and Igloos
- **Ruined Portal**: Nether portal ruins that can spawn anywhere
- **Bastion Remnant/Nether Fortress**: Nether structures (no biome restrictions)

### RNG Constants

Every structure type uses mathematical formulas to determine where it generates. These are the "RNG constants":

- **Spacing**: How far apart structures of this type tend to be (in chunks)
- **Separation**: Minimum distance between structures (prevents them from being too close)
- **Salt**: A unique number that makes each structure type generate differently
- **Linear Separation**: Whether the spacing follows a grid (1) or more complex pattern (0)

### Search Bounds

Where to look for your structures around the world origin (coordinates 0,0):

- **Radius**: Search in a circle of ±N blocks from origin
- **Box**: Search in a rectangle defined by coordinates (x1,z1) to (x2,z2)
- **Closest**: Search for structures as close to origin as mathematically possible

### Structure Occurrence

- **Min Occurrence**: How many of this structure type you want to find. For example:
  - "1" = find seeds with at least one village in your search area
  - "4" = find seeds with villages in all 4 quadrants of your search area

### Quadrants and Positions

Minecraft worlds are divided into 4 regions around the origin:

- **(0,0)**: Northeast quadrant (+X, +Z coordinates)
- **(-1,0)**: Northwest quadrant (-X, +Z coordinates)
- **(0,-1)**: Southeast quadrant (+X, -Z coordinates)
- **(-1,-1)**: Southwest quadrant (-X, -Z coordinates)

**Specific Quadrants**: Instead of searching all 4 quadrants, you can choose which ones to check.

**Specific Positions**: For advanced users, you can specify exact coordinates or ranges where each structure should appear within its quadrant.

### Chunk Offsets

Structures don't generate at exact block coordinates. They're offset by a few chunks:

- **Chunk Offset X/Z**: Usually 8,8 (meaning structures are offset by 8 chunks = 128 blocks from the mathematical position)
- This is automatically handled, but advanced users can modify it.

### Biome Validation

Many structures only generate in specific biomes (world types):

- **Biome Filter**: Check that structures generate in allowed biomes
- **4-Corner Biome Check**: Verify the biome at the structure position AND the 4 surrounding chunk corners (5 points total)
- **Independent Biome Checks**: Use different biome filters for each quadrant

### Common Biome Types

- **Overworld Biomes**: Plains, Desert, Forest, Taiga, Swamp, etc.
- **Nether Biomes**: Crimson Forest, Warped Forest, Soul Sand Valley, etc.
- **End Biomes**: The End dimension (not commonly searched)

## Example Usage Scenarios

### Find a Village Near Spawn
```
SeedStart: 0
SeedEnd: 1000000
Structure: village
Bounds: radius 200
Min Occurrence: 1
```

### Find Seeds with Mansions in All Quadrants
```
SeedStart: 0
SeedEnd: 1000000
Structure: mansion
Bounds: radius 1000
Min Occurrence: 4
Biome Filter: woodland_mansion
```

### Find Desert Temples with Specific Positioning
```
SeedStart: 0
SeedEnd: 1000000
Structure: desert temple
Specific Quadrants: (0,0)
Position Range: 100,100-200,200
Biome Filter: desert
```

## Tips for Success

1. **Start Small**: Begin with small seed ranges and simple constraints
2. **Use Biome Filters**: They dramatically speed up searches by eliminating impossible seeds
3. **Understand Spacing**: Larger spacing = fewer structures, smaller spacing = more common
4. **48-bit Mode**: Use when you have biome requirements - much faster for large searches
5. **Min Occurrence**: Higher values (like 4) are very restrictive and may take longer to find

## Troubleshooting

- **No Results**: Try larger seed ranges or less restrictive bounds
- **Slow Searches**: Add biome filters or use 48-bit expansion mode
- **Invalid Input**: Check coordinate ranges and make sure min occurrence ≤ 4
- **Biome Errors**: Some structures (bastion, fortress) have no biome restrictions

## Advanced Features

- **Multiple Constraints**: Search for seeds with village + outpost + mansion
- **Biome Point Checks**: Find seeds where specific coordinates have certain biomes
- **Custom RNG Values**: Enter your own spacing/separation/salt for modded structures
- **File Output**: Save results to a file for later analysis

Remember: Minecraft world generation is deterministic - the same seed always creates the same world. This tool helps you find seeds that match your criteria!