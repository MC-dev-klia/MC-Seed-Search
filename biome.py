"""
biome.py — Minecraft biome lookups via cubiomes (https://github.com/Cubitect/cubiomes)

Wraps the compiled cubiomes shared library to provide fast biome queries for
any seed, MC version, and dimension.  Includes the valid-biome tables for all
biome-gated Overworld structures and a single interactive prompt that accepts
either a structure preset or a custom biome list.
"""

import ctypes
import os

# ---------------------------------------------------------------------------
# Load the shared library
# ---------------------------------------------------------------------------
_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cubiomes", "libcubiomes.so")
_lib = ctypes.CDLL(_LIB_PATH)

# Opaque buffer size — must match sizeof(Generator) from cubiomes
GENERATOR_SIZE = 27592

# ---------------------------------------------------------------------------
# MC version constants  (from cubiomes/biomes.h enum MCVersion)
# ---------------------------------------------------------------------------
MC_1_7  = 8
MC_1_8  = 9
MC_1_9  = 10
MC_1_10 = 11
MC_1_11 = 12
MC_1_12 = 13
MC_1_13 = 16
MC_1_14 = 17
MC_1_15 = 18
MC_1_16 = 20
MC_1_17 = 21
MC_1_18 = 22
MC_1_19 = 24
MC_1_20 = 25
MC_1_21 = 28

MC_VERSIONS = {
    "1.7":  MC_1_7,  "1.8":  MC_1_8,  "1.9":  MC_1_9,
    "1.10": MC_1_10, "1.11": MC_1_11, "1.12": MC_1_12,
    "1.13": MC_1_13, "1.14": MC_1_14, "1.15": MC_1_15,
    "1.16": MC_1_16, "1.17": MC_1_17, "1.18": MC_1_18,
    "1.19": MC_1_19, "1.20": MC_1_20, "1.21": MC_1_21,
}

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------
DIM_OVERWORLD = 0
DIM_NETHER    = -1
DIM_END       = 1

DIMENSIONS = {
    "overworld": DIM_OVERWORLD,
    "nether":    DIM_NETHER,
    "end":       DIM_END,
}

# ---------------------------------------------------------------------------
# Biome ID <-> name maps  (cubiomes/biomes.h enum BiomeID)
# ---------------------------------------------------------------------------
BIOME_IDS: dict[str, int] = {
    # Classic overworld
    "ocean":                            0,
    "plains":                           1,
    "desert":                           2,
    "mountains":                        3,
    "forest":                           4,
    "taiga":                            5,
    "swamp":                            6,
    "river":                            7,
    "nether_wastes":                    8,
    "the_end":                          9,
    "frozen_ocean":                     10,
    "frozen_river":                     11,
    "snowy_tundra":                     12,
    "snowy_mountains":                  13,
    "mushroom_fields":                  14,
    "mushroom_field_shore":             15,
    "beach":                            16,
    "desert_hills":                     17,
    "wooded_hills":                     18,
    "taiga_hills":                      19,
    "mountain_edge":                    20,
    "jungle":                           21,
    "jungle_hills":                     22,
    "jungle_edge":                      23,
    "deep_ocean":                       24,
    "stone_shore":                      25,
    "snowy_beach":                      26,
    "birch_forest":                     27,
    "birch_forest_hills":               28,
    "dark_forest":                      29,
    "snowy_taiga":                      30,
    "snowy_taiga_hills":                31,
    "giant_tree_taiga":                 32,
    "giant_tree_taiga_hills":           33,
    "wooded_mountains":                 34,
    "savanna":                          35,
    "savanna_plateau":                  36,
    "badlands":                         37,
    "wooded_badlands_plateau":          38,
    "badlands_plateau":                 39,
    # 1.13+
    "small_end_islands":                40,
    "end_midlands":                     41,
    "end_highlands":                    42,
    "end_barrens":                      43,
    "warm_ocean":                       44,
    "lukewarm_ocean":                   45,
    "cold_ocean":                       46,
    "deep_warm_ocean":                  47,
    "deep_lukewarm_ocean":              48,
    "deep_cold_ocean":                  49,
    "deep_frozen_ocean":                50,
    # Mutated / rare variants (IDs 128+)
    "sunflower_plains":                 129,
    "desert_lakes":                     130,
    "gravelly_mountains":               131,
    "flower_forest":                    132,
    "taiga_mountains":                  133,
    "swamp_hills":                      134,
    "ice_spikes":                       140,
    "modified_jungle":                  149,
    "modified_jungle_edge":             151,
    "tall_birch_forest":                155,
    "tall_birch_hills":                 156,
    "dark_forest_hills":                157,
    "snowy_taiga_mountains":            158,
    "giant_spruce_taiga":               160,
    "giant_spruce_taiga_hills":         161,
    "modified_gravelly_mountains":      162,
    "shattered_savanna":                163,
    "shattered_savanna_plateau":        164,
    "eroded_badlands":                  165,
    "modified_wooded_badlands_plateau": 166,
    "modified_badlands_plateau":        167,
    # 1.16 Nether biomes
    "soul_sand_valley":                 170,
    "crimson_forest":                   171,
    "warped_forest":                    172,
    "basalt_deltas":                    173,
    # 1.18+ Overworld additions
    "meadow":                           177,
    "grove":                            178,
    "snowy_slopes":                     179,
    "jagged_peaks":                     180,
    "frozen_peaks":                     181,
    "stony_peaks":                      182,
    "deep_dark":                        183,
    "mangrove_swamp":                   184,
    "cherry_grove":                     185,
}

# Common alternate names users might type
_ALIASES: dict[str, str] = {
    "extreme_hills":                "mountains",
    "mega_taiga":                   "giant_tree_taiga",
    "mega_taiga_hills":             "giant_tree_taiga_hills",
    "roofed_forest":                "dark_forest",
    "cold_taiga":                   "snowy_taiga",
    "cold_taiga_hills":             "snowy_taiga_hills",
    "ice_plains":                   "snowy_tundra",
    "ice_mountains":                "snowy_mountains",
    "mushroom_island":              "mushroom_fields",
    "stone_beach":                  "stone_shore",
    "cold_beach":                   "snowy_beach",
    "mesa":                         "badlands",
    "mesa_plateau_f":               "wooded_badlands_plateau",
    "mesa_plateau":                 "badlands_plateau",
    "hell":                         "nether_wastes",
    "sky":                          "the_end",
    "nether":                       "nether_wastes",
    "deep_warm":                    "deep_warm_ocean",
    "deep_lukewarm":                "deep_lukewarm_ocean",
    "deep_cold":                    "deep_cold_ocean",
    "deep_frozen":                  "deep_frozen_ocean",
    "bamboo_jungle":                "modified_jungle",
}

BIOME_NAMES: dict[int, str] = {v: k for k, v in BIOME_IDS.items()}

# ---------------------------------------------------------------------------
# Structure → valid overworld biomes
#
# Each entry is a frozenset of biome IDs that allow the structure to spawn.
# None means the structure has no biome restriction (generates anywhere).
#
# NOTE: bastion_fortress uses the Nether (same salt 30084232 for both).
#       Biome noise does not gate them; they are excluded from this table.
# ---------------------------------------------------------------------------
STRUCTURE_VALID_BIOMES: dict[str, frozenset[int] | None] = {
    # Village — plains, desert, savanna, taiga, snowy variants
    "village": frozenset({
        1,   # plains
        2,   # desert
        5,   # taiga
        30,  # snowy_taiga
        35,  # savanna
        129, # sunflower_plains
        177, # meadow
    }),

    # Pillager Outpost — same base biomes as village (no snowy_taiga)
    "pillager_outpost": frozenset({
        1,   # plains
        2,   # desert
        5,   # taiga
        12,  # snowy_tundra
        30,  # snowy_taiga
        35,  # savanna
        129, # sunflower_plains
        177, # meadow (1.18+)
        178, # grove (1.18+)
        179, # snowy_slopes (1.18+)
        180, # jagged_peaks (1.18+)
        181, # frozen_peaks (1.18+)
        182, # stony_peaks (1.18+)
        185, # cherry_grove (1.19+)
    }),

    # Woodland Mansion — dark forest only
    "woodland_mansion": frozenset({
        29,  # dark_forest
    }),

    # Ocean Monument — deep ocean variants only
    "ocean_monument": frozenset({
        24,  # deep_ocean
        47,  # deep_warm_ocean
        48,  # deep_lukewarm_ocean
        49,  # deep_cold_ocean
        50,  # deep_frozen_ocean
    }),

    # Shipwreck — ocean and beach biomes
    "shipwreck": frozenset({
        0,   # ocean
        10,  # frozen_ocean
        16,  # beach
        24,  # deep_ocean
        26,  # snowy_beach
        44,  # warm_ocean
        45,  # lukewarm_ocean
        46,  # cold_ocean
        47,  # deep_warm_ocean
        48,  # deep_lukewarm_ocean
        49,  # deep_cold_ocean
        50,  # deep_frozen_ocean
    }),

    # Desert Pyramid — desert biomes
    "desert_pyramid": frozenset({
        2,   # desert
    }),

    # Jungle Temple — jungle biomes
    "jungle_temple": frozenset({
        21,  # jungle
    }),

    # Swamp Hut — swamp only
    "swamp_hut": frozenset({
        6,   # swamp
    }),

    # Igloo — snowy biomes
    "igloo": frozenset({
        12,  # snowy_tundra
        30,  # snowy_taiga
    }),

    # Ruined Portal — generates in any overworld biome, no restriction
    "ruined_portal": None,
}

# Aliases for structure preset lookups (maps alternate names to canonical names)
STRUCTURE_ALIASES: dict[str, str] = {
    # Bastion/Fortress aliases (no biome gate, not in STRUCTURE_VALID_BIOMES)
    "bastion_remnant": None,
    "nether_fortress": None,
    
    # Pillager Outpost aliases
    "pillager_post": "pillager_outpost",
    
    # Woodland Mansion aliases
    "dark_forest_mansion": "woodland_mansion",
    
    # Ocean Monument aliases
    "guardian_temple": "ocean_monument",
    
    # Shipwreck aliases
    "wreck": "shipwreck",
    "ship": "shipwreck",
    
    # Ruined Portal aliases
    "ruined": "ruined_portal",
    
    # Temple structure aliases (maps to their canonical names)
    "desert_temple": "desert_pyramid",
    "pyramid": "desert_pyramid",
    "jungle_pyramid": "jungle_temple",
    "witch_hut": "swamp_hut",
    "swamp": "swamp_hut",
    "ice_house": "igloo",
}

# Human-readable labels for the prompt
STRUCTURE_LABELS: dict[str, str] = {
    "village":          "Village               (salt 10387312,  spacing 34, sep 8,  linear 1)",
    "pillager_outpost": "Pillager Outpost       (salt 165745296, spacing 80, sep 24, linear 1)",
    "woodland_mansion": "Woodland Mansion       (salt 10387319,  spacing 80, sep 20, linear 1)",
    "ocean_monument":   "Ocean Monument         (salt 10387313,  spacing 32, sep 5,  linear 1)",
    "shipwreck":        "Shipwreck              (salt 165745295, spacing 24, sep 4,  linear 1)",
    "desert_pyramid":   "Desert Pyramid         (salt 14357617,  spacing 32, sep 8,  linear 0)",
    "jungle_temple":    "Jungle Temple          (salt 14357617,  spacing 32, sep 8,  linear 0)",
    "swamp_hut":        "Swamp Hut              (salt 14357617,  spacing 32, sep 8,  linear 0)",
    "igloo":            "Igloo                  (salt 14357617,  spacing 32, sep 8,  linear 0)",
    "ruined_portal":    "Ruined Portal          (salt 40552231,  spacing 40, sep 15, linear 0) — no biome gate",
}


def resolve_biome_name(name: str) -> int | None:
    """Return biome ID for a name string, or None if not found.
    Accepts canonical names, aliases, and is case/space-insensitive."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    return BIOME_IDS.get(key)


def list_biomes() -> str:
    """Return a sorted, human-readable list of all known biome names."""
    return "\n".join(sorted(BIOME_IDS.keys()))


# ---------------------------------------------------------------------------
# ctypes function signatures
# ---------------------------------------------------------------------------
_lib.setupGenerator.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
_lib.setupGenerator.restype  = None

_lib.applySeed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
_lib.applySeed.restype  = None

_lib.getBiomeAt.argtypes = [ctypes.c_void_p, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.getBiomeAt.restype  = ctypes.c_int


# ---------------------------------------------------------------------------
# BiomeGenerator — main public class
# ---------------------------------------------------------------------------
class BiomeGenerator:
    """
    Wraps a cubiomes Generator struct.

    Usage::

        gen = BiomeGenerator(mc_version=MC_1_21, dim=DIM_OVERWORLD)
        gen.apply_seed(12345)
        biome_id   = gen.get_biome(0, 64, 0)
        biome_name = gen.biome_name(biome_id)
    """

    def __init__(self, mc_version: int = MC_1_21, flags: int = 0,
                 dim: int = DIM_OVERWORLD):
        self._buf = (ctypes.c_uint8 * GENERATOR_SIZE)()
        self._ptr = ctypes.cast(self._buf, ctypes.c_void_p)
        self.mc_version = mc_version
        self.dim = dim
        _lib.setupGenerator(self._ptr, mc_version, flags)

    def apply_seed(self, seed: int) -> None:
        """Apply a world seed (up to 64-bit)."""
        _lib.applySeed(self._ptr, self.dim,
                       ctypes.c_uint64(seed & 0xFFFFFFFFFFFFFFFF))

    def get_biome(self, x: int, y: int, z: int, scale: int = 4) -> int:
        """
        Return the biome ID at position (x, y, z).

        scale=4  → biome/chunk coordinates (fast, ~4-block granularity) [default]
        scale=1  → exact block coordinates (slower)

        For pre-1.18 (layer-based), y is ignored.
        For 1.18+ (noise-based), y matters; use 64 for sea-level surface.
        When using scale=4 pass biome-coordinate values (block // 4).
        """
        return _lib.getBiomeAt(self._ptr, scale, x, y, z)

    def biome_name(self, biome_id: int) -> str:
        return BIOME_NAMES.get(biome_id, f"unknown({biome_id})")

    def biome_at_block(self, bx: int, bz: int, by: int = 64) -> int:
        """Get biome ID at block coordinates (converts to biome coords internally)."""
        return self.get_biome(bx >> 2, by >> 2, bz >> 2, scale=4)

    def check_structure_biome(self, bx: int, bz: int,
                               valid_biomes: frozenset[int] | None,
                               by: int = 64) -> tuple[bool, str]:
        """
        Check whether block position (bx, bz) is a valid biome for a structure.

        Returns (passes, biome_name).
        If valid_biomes is None the structure has no restriction → always passes.
        """
        bid = self.biome_at_block(bx, bz, by)
        name = self.biome_name(bid)
        if valid_biomes is None:
            return True, name
        return bid in valid_biomes, name

    def check_seed(self, seed: int,
                   requirements: list[tuple[int, int, int, set[int]]]
                   ) -> tuple[bool, list[tuple[int, int, str]]]:
        """
        Apply *seed* and verify every custom biome requirement.

        requirements — list of (x, z, y, allowed_biome_id_set)
        Returns (all_passed, [(x, z, biome_name), ...])
        """
        self.apply_seed(seed)
        results: list[tuple[int, int, str]] = []
        passed = True
        for x, z, y, allowed in requirements:
            bid = self.biome_at_block(x, z, y)
            name = self.biome_name(bid)
            if bid not in allowed:
                passed = False
            results.append((x, z, name))
        return passed, results


# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def prompt_biome_validation() -> frozenset[int] | None:
    """
    Ask the user for a biome filter applied at each structure candidate position.

    The user enters ONE of:
      - A structure name  → uses that structure's preset valid biome list
      - Biome name(s)     → comma-separated custom list
      - blank             → no biome filter

    Returns a frozenset of allowed biome IDs, or None if skipped.
    Always uses MC 1.21 noise (hardcoded).

    NOTE: Bastion Remnants and Nether Fortresses are Nether structures —
          leave this blank when searching for them.
    """
    structure_names = ", ".join(STRUCTURE_VALID_BIOMES.keys())
    print()
    print("Biome validation at structure candidate positions (optional).")
    print("  Enter a structure name to use its preset:  " + structure_names)
    print("  OR enter biome name(s) separated by commas for a custom filter.")
    print("  Type 'list' to see all biome names.  Leave blank to skip.")
    print()

    while True:
        raw = input("Biome filter: ").strip().lower()

        if not raw:
            return None

        if raw == "list":
            print(list_biomes())
            continue

        key = raw.replace(" ", "_").replace("-", "_")

        # Check if it is an alias, resolve to canonical name
        canonical_key = STRUCTURE_ALIASES.get(key, key)

        # Handle special case: None means no biome gate (bastion/fortress)
        if canonical_key is None:
            print(f"  {key} has no biome restriction — filter skipped.")
            return None

        # Check if it is a structure name
        if canonical_key in STRUCTURE_VALID_BIOMES:
            valid = STRUCTURE_VALID_BIOMES[canonical_key]
            if valid is None:
                print(f"  {canonical_key} has no biome restriction — filter skipped.")
                return None
            names = ", ".join(BIOME_NAMES.get(b, str(b)) for b in sorted(valid))
            print(f"  Using preset for {canonical_key}: [{names}]")
            return valid

        # Otherwise treat as comma-separated biome names
        biome_names_raw = [b.strip() for b in raw.split(",")]
        allowed: set[int] = set()
        bad: list[str] = []
        for bn in biome_names_raw:
            bid = resolve_biome_name(bn)
            if bid is None:
                bad.append(bn)
            else:
                allowed.add(bid)
        if bad:
            print(f"  Unknown: {', '.join(bad)}.  Type 'list' to see valid biome names.")
            continue
        labels = ", ".join(BIOME_NAMES.get(b, str(b)) for b in sorted(allowed))
        print(f"  Custom filter: [{labels}]")
        return frozenset(allowed)
