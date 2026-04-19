"""
structure_variants.py — Classification of structure variants for Bedrock 1.21+

Implements detection and classification of:
  - Bastion vs Nether Fortress (and bastion sub-types)
  - Ruined Portal variants (underground, surface, giant, normal)
  - Stronghold enumeration and placement
  - Portal properties (underground, giant, rotation, mirror)

Algorithm references: https://github.com/maehy/MCBE-1.18-Seed-Finder
"""

import math
import numpy as np
from structure import mt_init, mt_extract, N


# ---------------------------------------------------------------------------
# Village biome compatibility
# ---------------------------------------------------------------------------

# Village-compatible biomes (for stronghold validation)
VILLAGE_BIOME_IDS = frozenset({
    1,    # plains
    2,    # desert
    5,    # taiga
    30,   # snowy_taiga
    35,   # savanna
    129,  # sunflower_plains
    177,  # meadow
})


def is_village_biome(biome_id):
    """Check if a biome ID is compatible with village spawning."""
    return biome_id in VILLAGE_BIOME_IDS


# ---------------------------------------------------------------------------
# Region seed calculation (common to all structures)
# ---------------------------------------------------------------------------

def region_seed(world_seed, region_x, region_z, salt):
    """
    Calculate the region seed for a given world seed, region coordinates,
    and structure salt (used for bastion/fortress/portal).
    
    Formula: regZ * 341873128712 + regX * 132897987541 + worldSeed + salt
    """
    mixed = (world_seed + region_x * 341873128712 + region_z * 132897987541 + salt) & ((1 << 64) - 1)
    return mixed


def chunk_seed_rng(world_seed, chunk_x, chunk_z):
    """
    Generate chunk seed RNG state from world seed and chunk coordinates.
    Used for per-chunk property generation (portal properties).
    
    Returns: (xMul, zMul) pair used to compute chunk seed.
    """
    mt = mt_init(world_seed & 0xffffffff)
    idx = N
    xMul_raw, idx = mt_extract(mt, idx)
    zMul_raw, idx = mt_extract(mt, idx)
    
    # Ensure odd multipliers
    xMul = int(xMul_raw) | 1
    zMul = int(zMul_raw) | 1
    return xMul, zMul


def chunk_seed(world_seed, chunk_x, chunk_z):
    """
    Compute chunk seed from world seed and chunk coordinates.
    """
    xMul, zMul = chunk_seed_rng(world_seed, chunk_x, chunk_z)
    chunk_seed_val = (chunk_x * xMul + chunk_z * zMul) ^ world_seed
    return chunk_seed_val & 0xffffffff


# ---------------------------------------------------------------------------
# Bastion vs Nether Fortress classification
# ---------------------------------------------------------------------------

def classify_bastion_or_fortress(world_seed, region_x, region_z):
    """
    Classify a structure at (region_x, region_z) as either Bastion or Fortress.
    Both share the same salt (30084232) and placement parameters in Bedrock 1.18+.
    
    Returns: tuple (structure_type, bastion_subtype)
      - structure_type: "bastion" or "fortress"
      - bastion_subtype: 0-3 if bastion (maps to 1=bridge, 2=treasure, 3=hoglin, 4=housing)
                         None if fortress
    """
    BASTION_SALT = 30084232
    reg_seed = region_seed(world_seed, region_x, region_z, BASTION_SALT) & 0xffffffff
    mt = mt_init(reg_seed)
    idx = N
    
    # Third call determines bastion vs fortress
    x1, idx = mt_extract(mt, idx)
    y1, idx = mt_extract(mt, idx)
    check_val, idx = mt_extract(mt, idx)
    is_bastion = (check_val % 6) >= 2
    
    if is_bastion:
        # Bastion: consume rotation, then get type
        _rotation, idx = mt_extract(mt, idx)
        bastion_type_raw, idx = mt_extract(mt, idx)
        bastion_type = bastion_type_raw % 4
        return "bastion", bastion_type
    else:
        return "fortress", None


# ---------------------------------------------------------------------------
# Ruined Portal variant detection
# ---------------------------------------------------------------------------

def classify_portal_variant(world_seed, chunk_x, chunk_z):
    """
    Classify a ruined portal by its properties: underground, airpocket, rotation,
    mirror, and giant status.
    
    Returns: dict with keys:
      - underground: bool
      - airpocket: bool (if True, portal is in airpocket)
      - rotation: 0-3
      - mirror: bool
      - giant: bool
      - variant: "giant_portal_1" to "giant_portal_3", or "portal_1" to "portal_10"
      - variant_short: "giant" or "normal"
      - variant_type: summary string for user output
    """
    chunk_seed_val = chunk_seed(world_seed, chunk_x, chunk_z)
    mt = mt_init(chunk_seed_val)
    idx = N
    # Extract properties in order
    underground_raw, idx = mt_extract(mt, idx)
    underground = (underground_raw & 0xffffffff) / (2**32) < 0.5
    
    airpocket_raw, idx = mt_extract(mt, idx)
    # Airpocket only affects state; its value isn't directly used for classification
    airpocket = (airpocket_raw & 0xffffffff) / (2**32) < 0.5
    
    rotation_raw, idx = mt_extract(mt, idx)
    rotation = rotation_raw % 4
    
    mirror_raw, idx = mt_extract(mt, idx)
    mirror = (mirror_raw & 0xffffffff) / (2**32) >= 0.5
    
    giant_raw, idx = mt_extract(mt, idx)
    giant = (giant_raw & 0xffffffff) / (2**32) < 0.05
    
    if giant:
        variant_num = (mt_extract(mt, idx)[0] % 3) + 1
        variant = f"giant_portal_{variant_num}"
        variant_short = "giant"
    else:
        variant_num = (mt_extract(mt, idx)[0] % 10) + 1
        variant = f"portal_{variant_num}"
        variant_short = "normal"
    
    # Generate variant type summary
    if underground:
        variant_type = f"{variant_short}:underground"
    else:
        variant_type = f"{variant_short}:surface"
    
    return {
        "underground": underground,
        "airpocket": airpocket,
        "rotation": rotation,
        "mirror": mirror,
        "giant": giant,
        "variant": variant,
        "variant_short": variant_short,
        "variant_type": variant_type,
    }


# ---------------------------------------------------------------------------
# Village detection (for stronghold placement)
# ---------------------------------------------------------------------------

def check_village_at_chunk(world_seed, chunk_x, chunk_z):
    """
    Check if a village exists at a specific chunk coordinate.
    Used by stronghold placement algorithm.
    """
    VILLAGE_SALT = 10387312
    VILLAGE_SPACING = 34
    VILLAGE_SEPARATION = 26
    
    reg_x = int(chunk_x // VILLAGE_SPACING)
    reg_z = int(chunk_z // VILLAGE_SPACING)
    
    reg_seed = region_seed(world_seed, reg_x, reg_z, VILLAGE_SALT) & 0xffffffff
    mt = mt_init(reg_seed)
    idx = N
    
    # Village uses averaged two-draw algorithm
    r0, idx = mt_extract(mt, idx)
    r1, idx = mt_extract(mt, idx)
    local_x = ((r0 % VILLAGE_SEPARATION) + (r1 % VILLAGE_SEPARATION)) // 2
    
    r2, idx = mt_extract(mt, idx)
    r3, idx = mt_extract(mt, idx)
    local_z = ((r2 % VILLAGE_SEPARATION) + (r3 % VILLAGE_SEPARATION)) // 2
    
    village_chunk_x = reg_x * VILLAGE_SPACING + local_x
    village_chunk_z = reg_z * VILLAGE_SPACING + local_z
    
    return village_chunk_x == chunk_x and village_chunk_z == chunk_z


# ---------------------------------------------------------------------------
# Stronghold placement
# ---------------------------------------------------------------------------

def find_strongholds_in_radius(world_seed, target_x, target_z, radius, biome_gen=None):
    """
    Find all stronghold positions within a given radius of target position.
    Uses both quasi-random placement (via angle/radius with village checking) 
    and grid-based placement. Matches Bedrock 1.18+ algorithm.
    
    Args:
        world_seed: 32-bit world seed
        target_x, target_z: center block coordinates
        radius: search radius in blocks
        biome_gen: optional BiomeGenerator to filter by village-compatible biomes
    
    Returns: list of (x, z) block positions of strongholds
    """
    strongholds = []
    
    # Part 1: Quasi-random placement (initial ~3 strongholds via village finding)
    mt = mt_init(world_seed & 0xffffffff)
    idx = N
    
    angle_raw, idx = mt_extract(mt, idx)
    angle = (angle_raw & 0xffffffff) / (2**32) * math.pi * 2
    
    radius_raw, idx = mt_extract(mt, idx)
    r = (radius_raw % 16) + 40
    
    STRONGHOLD_COUNT = 3
    for i in range(STRONGHOLD_COUNT):
        # Calculate chunk coords from angle and radius
        cx = int(math.floor(r * math.cos(angle)))
        cz = int(math.floor(r * math.sin(angle)))
        
        found = False
        # Search 8x8 chunk area for villages
        for dx in range(-8, 8):
            if found:
                break
            for dz in range(-8, 8):
                search_chunk_x = cx + dx
                search_chunk_z = cz + dz
                
                # Check if village exists at this chunk
                if check_village_at_chunk(world_seed, search_chunk_x, search_chunk_z):
                    # Found village - stronghold is at this chunk's center
                    sh_x = search_chunk_x * 16 + 8
                    sh_z = search_chunk_z * 16 + 8
                    
                    dx_dist = sh_x - target_x
                    dz_dist = sh_z - target_z
                    dist = math.sqrt(dx_dist*dx_dist + dz_dist*dz_dist)
                    
                    if dist <= radius:
                        # Validate biome if generator provided
                        if biome_gen is None or _is_stronghold_valid_biome(biome_gen, sh_x, sh_z):
                            strongholds.append((sh_x, sh_z))
                    
                    found = True
                    break
        
        # Update angle and radius for next iteration
        if found:
            angle += 0.6 * math.pi
            r += 8
        else:
            angle += 0.25 * math.pi
            r += 4
    
    # Part 2: Grid-based placement (200x200 chunk grid)
    GRID_SIZE = 200
    target_grid_x = int(math.floor(target_x / (GRID_SIZE * 16)))
    target_grid_z = int(math.floor(target_z / (GRID_SIZE * 16)))
    grid_radius = int(math.ceil(radius / (GRID_SIZE * 16))) + 2
    
    for grid_x in range(target_grid_x - grid_radius, target_grid_x + grid_radius + 1):
        for grid_z in range(target_grid_z - grid_radius, target_grid_z + grid_radius + 1):
            grid_x100 = GRID_SIZE * grid_x + 100
            grid_z100 = GRID_SIZE * grid_z + 100
            
            # Compute grid cell seed using exact formula from maehy
            xMul = ((-1683231919 * grid_x100) & 0xffffffff)
            zMul = ((-1100435783 * grid_z100) & 0xffffffff)
            cell_seed = ((xMul + zMul + world_seed + 97858791) & 0xffffffff)
            
            mt_cell = mt_init(cell_seed)
            idx_cell = N
            
            spawn_prob, idx_cell = mt_extract(mt_cell, idx_cell)
            if (spawn_prob & 0xffffffff) / (2**32) < 0.25:
                # This grid cell spawns a stronghold
                min_x = GRID_SIZE * grid_x + GRID_SIZE - 150  # chunks
                max_x = GRID_SIZE * grid_x + 150
                min_z = GRID_SIZE * grid_z + GRID_SIZE - 150
                max_z = GRID_SIZE * grid_z + 150
                
                x_raw, idx_cell = mt_extract(mt_cell, idx_cell)
                z_raw, idx_cell = mt_extract(mt_cell, idx_cell)
                
                # Generate random position in range [min, max)
                x_chunk = min_x + (x_raw % (max_x - min_x))
                z_chunk = min_z + (z_raw % (max_z - min_z))
                
                sh_x = x_chunk * 16 + 8
                sh_z = z_chunk * 16 + 8
                
                dx_dist = sh_x - target_x
                dz_dist = sh_z - target_z
                dist = math.sqrt(dx_dist*dx_dist + dz_dist*dz_dist)
                
                if dist <= radius:
                    # Validate biome if generator provided
                    if biome_gen is None or _is_stronghold_valid_biome(biome_gen, sh_x, sh_z):
                        strongholds.append((sh_x, sh_z))
    
    return strongholds


def _is_stronghold_valid_biome(biome_gen, block_x, block_z):
    """
    Check if a stronghold position is in a village-compatible biome.
    Strongholds require villages to exist, so they only spawn in village biomes.
    """
    try:
        biome_id = biome_gen.biome_at_block(block_x, block_z)
        return is_village_biome(biome_id)
    except Exception:
        # If biome check fails, assume invalid to avoid false positives
        return False
