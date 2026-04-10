"""
main.py — Seed search loop with layered constraints, bounding-box bounds,
          biome point-checks, and 32-bit expansion mode.

Constraint types
----------------
  structure  — uses the Bedrock MT RNG to find candidate block positions in
               four adjacent regions and checks them against a bounding box.
               Optionally validates biome at each in-box position.

  biome      — checks the biome at a fixed world coordinate (x, y, z) against
               an allowed set.  Useful for filtering seeds where a particular
               spot must be a specific biome regardless of structure placement.

Ordering for performance
------------------------
  The FIRST structure constraint is processed by the fast numba JIT kernel.
  All subsequent constraints (structure or biome) are checked at Python level
  only for seeds that already passed the first constraint.  More selective
  constraints entered earlier = faster overall scan.
"""

import sys
import time
import re
import biome as bm
from structure import getpos, scan_batch
import structure_variants as sv

sys.stdout.reconfigure(line_buffering=True)

MASK32   = (1 << 32) - 1
SIGN_BIT = 1 << 63
TWO64    = 1 << 64

BANNER = """
Minecraft brute-force seed searcher designed for bedrock edition structures using cubiomes as refernce.
Supports many options such as multiple structure constraints, biome checks, and special options for speedup.

Usage: Run the scripts and follow the prompts for your desired constraints and options.

Inputs:
1 - Seed range: Specify the start and end of the seed range to search. Start is inclusive, end is exclusive.
2 - Output destination: Choose to print results to the console or save them to a file.
3 - Constraints: Define one or more structure constraints (with presets available) and biome point constraints.
    3a - Structure type: Specify structure type to fill (spacing, separation, salt, linear separation)
    3b - Bounds: Define search bounds for structure positions (radius, box, or closest preset), or positions for each quadrant.
    3c - Occurrence: Minimum number of quadrants that must contain the structure.
    3d - Offsets: Optional chunk offsets for structure position calculation accuracy.
    3e - Biome filters: Optionally specify allowed biomes for each quadrant of the structure
        3e.1 - Independent biome filters: Allow different biome sets for each quadrant.
        3e.2 - Corner check: If enabled, also check the biome at the 4 chunk corners around each structure position.
4 - Expansion mode: If biome checks are needed, optionally enable a mode that scans the first N upper 32-bit seed variants for biome matching.

Additional notes:
Biomes are taken from java biomes, mostly accurate but may differ in edge biomes.
Biome checks are not supported for nether or end structures, as the biome generator doesn't match.
This finder only checks the first four quadrants.
Some biome RNG constants are not availible as presets, so you will need to find them manually.
Use trusted refernce sources like https://www.chunkbase.com/apps/seed-map to verify for false positives.
Other additional structure generator checks are not implemented, such as checking y-level.
Certain order of constraints may be faster than others, so experiment with different arrangements if you need to optimize for speed.
Only checking certain quadrants may be faster than checking all quadrants.
Structure positions only occur every 16 blocks based on the offset, so if (0,0) is possible, then (1,0) is not possible, but (16,0) is possible.
Certain constrants may be impossible to satisfy, in certain cases the program will warn you.
"""
REGION_ORDER  = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
REGION_LABELS = {
    (0,  0):  "[+X +Z]",
    (-1, 0):  "[-X +Z]",
    (0, -1):  "[+X -Z]",
    (-1,-1):  "[-X -Z]",
}

PRESETS = {
    # Bastion Remnants / Nether Fortress
    "bastion":          (30,  4, 30084232,  False, "bastion"),
    "bastion remnant":  (30,  4, 30084232,  False, "bastion"),
    "fortress":         (30,  4, 30084232,  False, "fortress"),
    "nether fortress":  (30,  4, 30084232,  False, "fortress"),
    "bastion/fortress": (30,  4, 30084232,  False, "either"),
    
    # Village
    "village":          (34,  8, 10387312,  True, None),
    
    # Stronghold (special: no standard structure params)
    "stronghold":       (0, 0, 0, False, "stronghold"),
    
    # Pillager Outpost
    "pillager":         (80, 24, 165745296, True, None),
    "outpost":          (80, 24, 165745296, True, None),
    "pillager outpost": (80, 24, 165745296, True, None),
    "pillager post":    (80, 24, 165745296, True, None),
    
    # Woodland Mansion
    "mansion":          (80, 20, 10387319,  True, None),
    "woodland mansion": (80, 20, 10387319,  True, None),
    "dark forest mansion": (80, 20, 10387319,  True, None),
    
    # Ocean Monument
    "monument":         (32,  5, 10387313,  True, None),
    "ocean monument":   (32,  5, 10387313,  True, None),
    "guardian temple":  (32,  5, 10387313,  True, None),
    
    # Shipwreck
    "shipwreck":        (24,  4, 165745295, False, None),
    "wreck":            (24,  4, 165745295, False, None),
    "ship":             (24,  4, 165745295, False, None),
    
    # Ruined Portal
    "portal":           (40, 15, 40552231,  False, "portal"),
    "ruined portal":    (40, 15, 40552231,  False, "portal"),
    "ruined":           (40, 15, 40552231,  False, "portal"),
    
    # Temples (Desert Pyramid, Jungle Temple, Swamp Hut, Igloo)
    "temple":           (32,  8, 14357617,  False, None),
    "temples":          (32,  8, 14357617,  False, None),
    "desert pyramid":   (32,  8, 14357617,  False, None),
    "desert temple":    (32,  8, 14357617,  False, None),
    "desert":           (32,  8, 14357617,  False, None),
    "pyramid":          (32,  8, 14357617,  False, None),
    "jungle temple":    (32,  8, 14357617,  False, None),
    "jungle pyramid":   (32,  8, 14357617,  False, None),
    "jungle":           (32,  8, 14357617,  False, None),
    "swamp hut":        (32,  8, 14357617,  False, None),
    "witch hut":        (32,  8, 14357617,  False, None),
    "swamp":            (32,  8, 14357617,  False, None),
    "igloo":            (32,  8, 14357617,  False, None),
    "ice house":        (32,  8, 14357617,  False, None),
}

PRESET_NAMES = [
    "bastion/fortress", "bastion", "fortress", "stronghold", "village",
    "outpost", "mansion", "monument", "shipwreck", "portal", "temple",
]


# ---------------------------------------------------------------------------
# UI helpers — constraint input
# ---------------------------------------------------------------------------

def _prompt_rng():
    print()
    print("  Available presets:")
    print("    " + ", ".join(PRESET_NAMES))
    print("    (type a preset name, or press Enter to enter values manually)")
    raw = input("  Structure preset: ").strip().lower()
    if raw in PRESETS:
        preset_data = PRESETS[raw]
        sp, sep, sa, ls, struct_type = preset_data
        print(f"    Loaded '{raw}': spacing={sp}, separation={sep}, "
              f"salt={sa}, linear={int(ls)}")
        return sp, sep, sa, ls, raw, struct_type
    if raw:
        print(f"    Unknown preset '{raw}' — entering manually.")
    sp  = int(input("  Spacing: "))
    sep = int(input("  Separation: "))
    sa  = int(input("  Salt: "))
    ls  = bool(int(input("  Linear separation (0 or 1): ")))
    return sp, sep, sa, ls, f"{sp}/{sep}/{sa}", None


def _prompt_bounds(seperation):
    print("  Search bounds:")
    print("    (r)adius   — ±N blocks symmetric around origin")
    print("    (b)ox      — custom x1 z1 x2 z2")
    print(f"    (c)losest  — closest-possible preset  "
          f"[-16-16*seperation-error, error]  (seperation={seperation})")
    ch = input("    Mode [r]: ").strip().lower() or "r"

    if ch in ("r", "radius"):
        r = int(input("    Radius: "))
        return -r, -r, r, r

    if ch in ("c", "closest"):
        raw_e = input("    Error margin [0]: ").strip()
        e = int(raw_e) if raw_e else 0
        x1 = z1 = -16 - 16 * seperation - e
        x2 = z2 = e
        print(f"    Closest preset → ({x1},{z1}) to ({x2},{z2})")
        return x1, z1, x2, z2

    # Bounding box
    raw = input("    x1 z1 x2 z2 (space-separated): ").strip().split()
    x1, z1, x2, z2 = int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3])
    if x1 > x2: x1, x2 = x2, x1
    if z1 > z2: z1, z2 = z2, z1
    return x1, z1, x2, z2


def _prompt_structure_constraint(idx):
    print(f"\n=== Structure Constraint {idx} ===")
    sp, sep, sa, ls, label, struct_type = _prompt_rng()

    # Ask for variant type if applicable
    variant_filter = None
    if struct_type == "bastion":
        variant_names = ["bridge", "treasure", "hoglin", "housing"]
        print("  Bastion types: 0=any, 1=bridge, 2=treasure, 3=hoglin, 4=housing")
        var_input = input("  Bastion type [0]: ").strip() or "0"
        try:
            var_idx = int(var_input)
            if 1 <= var_idx <= 4:
                variant_filter = var_idx - 1  # Store as 0-3 for internal use
        except ValueError:
            pass
    elif struct_type == "portal" or struct_type == "ruined_portal":
        # Question 1: Underground vs Surface
        print("  Question 1 - Depth:")
        print("    0=any, 1=underground, 2=surface")
        depth_input = input("  Depth filter [0]: ").strip() or "0"
        try:
            depth_idx = int(depth_input)
            if depth_idx not in (0, 1, 2):
                depth_idx = 0
        except ValueError:
            depth_idx = 0
        
        # Question 2: Giant vs Normal
        print("  Question 2 - Portal type:")
        print("    0=any, 1=giant, 2=normal")
        type_input = input("  Portal type [0]: ").strip() or "0"
        try:
            type_idx = int(type_input)
            if type_idx not in (0, 1, 2):
                type_idx = 0
        except ValueError:
            type_idx = 0
        
        # Store as tuple (depth_filter, type_filter) if any filter is set
        if depth_idx > 0 or type_idx > 0:
            variant_filter = (depth_idx, type_idx)
        else:
            variant_filter = None
    elif struct_type == "fortress":
        # Fortress has no subtypes
        pass

    specific_quadrants = None
    specific_positions = None
    x1 = z1 = x2 = z2 = None

    occ_raw = input("  Min occurrence [1]: ").strip()
    occ = int(occ_raw) if occ_raw else 1

    # Skip quadrant-specific prompts for stronghold (doesn't use standard quadrant placement)
    if occ < 4 and struct_type != "stronghold":
        ans = input(
            "  Specify specific quadrants and positions? (y/n) [n]"
        ).strip().lower()
        if ans in ("y", "yes"):
            print("  Quadrants: (0,0)=[+X +Z], (-1,0)=[-X +Z], (0,-1)=[+X -Z], (-1,-1)=[-X -Z]")
            quad_input = input("  Quadrants to check (comma-separated, e.g. (0,0),(-1,0)) [all]: ").strip()
            if quad_input:
                try:
                    specific_quadrants = []
                    for match in re.finditer(r'\(\s*(\-?\d+)\s*,\s*(\-?\d+)\s*\)', quad_input):
                        rx = int(match.group(1))
                        rz = int(match.group(2))
                        specific_quadrants.append((rx, rz))
                    if not specific_quadrants:
                        raise ValueError("No valid quadrants found")
                except:
                    print("  Invalid quadrant format — using all quadrants.")
                    specific_quadrants = None
            else:
                specific_quadrants = None

            if specific_quadrants:
                specific_positions = {}
                for rx, rz in specific_quadrants:
                    pos_input = input(
                        f"  Positions for quadrant ({rx},{rz}) (range x1,z1-x2,z2 or 'from x1,z1 to x2,z2') [auto]: "
                    ).strip()
                    if pos_input:
                        try:
                            # "from X to Y" format
                            from_match = re.match(r"^\s*from\s+(\d+)\s*[ ,]\s*(\d+)\s+to\s+(\d+)\s*[ ,]\s*(\d+)\s*$", pos_input, re.IGNORECASE)
                            if from_match:
                                x1r, z1r, x2r, z2r = map(int, from_match.groups())
                                if x1r > x2r:
                                    x1r, x2r = x2r, x1r
                                if z1r > z2r:
                                    z1r, z2r = z2r, z1r
                                specific_positions[(rx, rz)] = (x1r, z1r, x2r, z2r)
                            # Range format: x1,z1-x2,z2
                            elif re.match(r"^\s*\(?\s*(-?\d+)\s*[ ,]\s*(-?\d+)\s*\)?\s*-\s*\(?\s*(-?\d+)\s*[ ,]\s*(-?\d+)\s*\)?\s*$", pos_input):
                                m = re.match(r"^\s*\(?\s*(-?\d+)\s*[ ,]\s*(-?\d+)\s*\)?\s*-\s*\(?\s*(-?\d+)\s*[ ,]\s*(-?\d+)\s*\)?\s*$", pos_input)
                                x1r, z1r, x2r, z2r = map(int, m.groups())
                                if x1r > x2r:
                                    x1r, x2r = x2r, x1r
                                if z1r > z2r:
                                    z1r, z2r = z2r, z1r
                                specific_positions[(rx, rz)] = (x1r, z1r, x2r, z2r)
                        except Exception:
                            print(f"  Invalid position format for ({rx},{rz}) — using auto positions.")
                            specific_positions[(rx, rz)] = None
                    else:
                        specific_positions[(rx, rz)] = None
            print("  Specific quadrants and positions configured." if specific_quadrants else "  Using all quadrants with auto positions.")

    # Ask for bounds unless specific positions are defined for all quadrants
    has_specific_positions = specific_positions and any(pos is not None for pos in specific_positions.values())
    
    if has_specific_positions:
        # When specific positions are defined, disable additional bounds filtering
        x1, z1, x2, z2 = -999999, -999999, 999999, 999999
        print("  Using specific position ranges — additional bounds filtering disabled.")
    else:
        # Ask for bounds for additional filtering
        x1, z1, x2, z2 = _prompt_bounds(sep)

    # Warning for impossible cases (only when bounds are actually used)
    if not has_specific_positions:
        radius = max(abs(x1), abs(x2), abs(z1), abs(z2))
        if radius < 16 * sep and occ >= 2:
            print(f"  WARNING: Search radius ({radius}) < 16 * separation ({16 * sep}) = {16 * sep}")
            print("           With min occurrence >= 2, this may be impossible as structures")
            print("           are spaced at least 16 * separation blocks apart.")
            ans = input("  Continue anyway? (y/n) [y]: ").strip().lower()
            if ans in ("n", "no"):
                return None, False  # Skip this constraint

    print()
    raw_offx = input("  Chunk offset X [8]: ").strip()
    raw_offy = input("  Chunk offset Z [8]: ").strip()
    offx = int(raw_offx) if raw_offx else 8
    offy = int(raw_offy) if raw_offy else 8
    if offx != 8 or offy != 8:
        print(f"  Using offset ({offx}, {offy}).")
    else:
        print("  Using default offset (8, 8).")

    quadrant_biomes = {}
    corner_check = False
    needs_biome_gen = False

    # Skip independent biome checks per quadrant for stronghold (doesn't use standard quadrant placement)
    if struct_type != "stronghold":
        ans = input(
            "  Use independent biome checks per quadrant? (y/n) [n]"
        ).strip().lower()
        if ans in ("y", "yes"):
            quadrants = specific_quadrants if specific_quadrants else [(0,0), (-1,0), (0,-1), (-1,-1)]
            for rx, rz in quadrants:
                print(f"  Biome filter for quadrant ({rx},{rz}) {REGION_LABELS[(rx,rz)]}:")
                biomes = bm.prompt_biome_validation()
                quadrant_biomes[(rx, rz)] = biomes
                if biomes is not None:
                    needs_biome_gen = True
        else:
            biomes = bm.prompt_biome_validation()
            if biomes is not None:
                needs_biome_gen = True
                for rx, rz in [(0,0), (-1,0), (0,-1), (-1,-1)]:
                    quadrant_biomes[(rx, rz)] = biomes
    else:
        # For stronghold, ask for global biome filter (not per-quadrant)
        biomes = bm.prompt_biome_validation()
        if biomes is not None:
            needs_biome_gen = True
            # Apply same biome filter to all quadrants for stronghold
            for rx, rz in [(0,0), (-1,0), (0,-1), (-1,-1)]:
                quadrant_biomes[(rx, rz)] = biomes

    if needs_biome_gen:
        ans = input(
            "  4-corner biome check? (y/n) [n]\n"
        ).strip().lower()
        corner_check = ans in ("y", "yes")
        print("    4-corner check ON." if corner_check else "    4-corner check OFF.")

    return {
        "type":        "structure",
        "label":       label,
        "struct_type": struct_type,
        "variant_filter": variant_filter,  # Filter by specific variant (bastion type, portal type)
        "spacing":     sp,
        "separation":  sep,
        "salt":        sa,
        "linear_sep":  ls,
        "occurence":   occ,
        "x1": x1, "z1": z1, "x2": x2, "z2": z2,
        "offx":        offx,
        "offy":        offy,
        "quadrant_biomes": quadrant_biomes,
        "corner_check": corner_check,
        "specific_quadrants": specific_quadrants,
        "specific_positions": specific_positions,
        "variants":    {},  # Will store (pos) -> variant_info mappings
    }, needs_biome_gen


def _prompt_biome_constraint(idx):
    """
    Prompt for a fixed-coordinate biome constraint.
    """
    print(f"\n=== Biome Point Constraint {idx} ===")
    print("  Check the biome at a fixed world coordinate.")
    raw = input("  Position — 'x z' or 'x y z' (y defaults to 64): ").strip()
    parts = raw.split()
    if len(parts) == 2:
        cx, cz, cy = int(parts[0]), int(parts[1]), 64
    elif len(parts) == 3:
        cx, cy, cz = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        print("  Expected 2 or 3 numbers — skipping this constraint.")
        return None

    biomes = bm.prompt_biome_validation()
    if biomes is None:
        print("  No biomes entered — constraint skipped.")
        return None

    return {
        "type":    "biome",
        "label":   f"biome@({cx},{cz})",
        "x":  cx,
        "z":  cz,
        "y":  cy,
        "allowed": biomes,
    }


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

def _classify_variant(seed32, struct_type, chunk_x, chunk_z, chunk_bx, chunk_bz, spacing, variant_filter=None):
    """Helper to classify structure variant based on type.
    
    Args:
        seed32: 32-bit world seed
        struct_type: structure type indicator from preset
        chunk_x, chunk_z: chunk coordinates (from block coords >> 4)
        chunk_bx, chunk_bz: block coordinates (for context)
        spacing: structure spacing (to compute region coords)
        variant_filter: optional variant preference to check against (None = accept all)
    
    Returns: (variant_label, matches_filter) or None if doesn't match filter
    """
    # Compute region coordinates from chunk coordinates
    region_x = chunk_x // spacing if spacing else 0
    region_z = chunk_z // spacing if spacing else 0
    
    if struct_type == "bastion":
        structure_name, bastion_type = sv.classify_bastion_or_fortress(seed32, region_x, region_z)
        if structure_name == "bastion":
            bastion_names = ["bridge", "treasure", "hoglin", "housing"]
            variant_label = f"bastion:{bastion_names[bastion_type]}"
            # Check if matches filter (variant_filter is 0-3 for bastion types)
            if variant_filter is not None and variant_filter >= 0 and bastion_type != variant_filter:
                return None  # Doesn't match filter
            return variant_label
        else:
            # It's a fortress, not bastion
            return None if (struct_type == "bastion" and variant_filter is not None) else "fortress"
    elif struct_type == "fortress":
        structure_name, _ = sv.classify_bastion_or_fortress(seed32, region_x, region_z)
        return "fortress" if structure_name == "fortress" else None
    elif struct_type == "either":
        structure_name, bastion_type = sv.classify_bastion_or_fortress(seed32, region_x, region_z)
        if structure_name == "bastion":
            bastion_names = ["bridge", "treasure", "hoglin", "housing"]
            variant_label = f"bastion:{bastion_names[bastion_type]}"
            if variant_filter is not None and variant_filter >= 0 and bastion_type != variant_filter:
                return None
            return variant_label
        else:
            return "fortress"
    elif struct_type == "portal" or struct_type == "ruined_portal":
        portal_info = sv.classify_portal_variant(seed32, chunk_x, chunk_z)
        variant_type = portal_info["variant_type"]
        # Check filter: tuple (depth_filter, type_filter) where 0=any, 1=first, 2=second
        if variant_filter is not None and isinstance(variant_filter, tuple):
            depth_filter, type_filter = variant_filter
            matches = True
            
            # Check depth filter
            if depth_filter == 1 and not portal_info["underground"]:
                matches = False  # Want underground but it's surface
            elif depth_filter == 2 and portal_info["underground"]:
                matches = False  # Want surface but it's underground
            
            # Check type filter
            if type_filter == 1 and not portal_info["giant"]:
                matches = False  # Want giant but it's normal
            elif type_filter == 2 and portal_info["giant"]:
                matches = False  # Want normal but it's giant
            
            if not matches:
                return None  # Doesn't match filter
        return variant_type
    elif struct_type == "stronghold":
        return "stronghold"
    return None

def _check_struct_positions(s32, c, biome_gen=None):
    # Special handling for stronghold (doesn't use standard region-based structure params)
    if c.get("struct_type") == "stronghold":
        # Stronghold: find strongholds in the bounding box region
        # Apply seed to biome generator if provided
        if biome_gen:
            biome_gen.apply_seed(s32)
        
        positions = []
        strongholds = sv.find_strongholds_in_radius(s32, 0, 0, 5000, biome_gen)  # Search within ~5000 blocks
        found = 0
        for sh_x, sh_z in strongholds:
            in_box = c["x1"] < sh_x < c["x2"] and c["z1"] < sh_z < c["z2"]
            if in_box:
                c["variants"][(sh_x, sh_z)] = "stronghold"
                positions.append(((0, 0), (sh_x, sh_z), True))  # Use dummy region (0,0)
                found += 1
        return positions, found
    
    # Standard structure handling
    quadrants = c.get("specific_quadrants") or [(0, 0), (-1, 0), (0, -1), (-1, -1)]
    positions = []
    found = 0
    for rx, rz in quadrants:
        pos_spec = None
        if c.get("specific_positions") and (rx, rz) in c["specific_positions"]:
            pos_spec = c["specific_positions"][(rx, rz)]

        if pos_spec is None:
            # Auto position for this quadrant
            pos = getpos(s32, rx, rz,
                         c["spacing"], c["separation"], c["salt"], c["linear_sep"],
                         c["offx"], c["offy"])
            bx, bz = pos
            in_box = c["x1"] < bx < c["x2"] and c["z1"] < bz < c["z2"]
            # Classify variant if applicable
            if in_box and c.get("struct_type"):
                try:
                    chunk_x, chunk_z = bx >> 4, bz >> 4
                    variant = _classify_variant(s32, c["struct_type"], chunk_x, chunk_z, bx, bz, c["spacing"], c.get("variant_filter"))
                    if variant:
                        c["variants"][pos] = variant
                except Exception:
                    pass  # Silently skip variant classification errors
            positions.append(((rx, rz), pos, in_box))
            if in_box:
                found += 1

        elif isinstance(pos_spec, tuple) and len(pos_spec) == 4:
            # Range: x1,z1,x2,z2 (inclusive)
            pos = getpos(s32, rx, rz,
                         c["spacing"], c["separation"], c["salt"], c["linear_sep"],
                         c["offx"], c["offy"])
            bx, bz = pos
            in_range = pos_spec[0] <= bx <= pos_spec[2] and pos_spec[1] <= bz <= pos_spec[3]
            in_box = in_range and (c["x1"] < bx < c["x2"] and c["z1"] < bz < c["z2"])
            # Classify variant if applicable
            if in_box and c.get("struct_type"):
                try:
                    chunk_x, chunk_z = bx >> 4, bz >> 4
                    variant = _classify_variant(s32, c["struct_type"], chunk_x, chunk_z, bx, bz, c["spacing"], c.get("variant_filter"))
                    if variant:
                        c["variants"][pos] = variant
                except Exception:
                    pass
            positions.append(((rx, rz), pos, in_box))
            if in_box:
                found += 1

        else:
            # Explicit point list - calculate actual position and check if it matches any specified point
            actual_pos = getpos(s32, rx, rz,
                         c["spacing"], c["separation"], c["salt"], c["linear_sep"],
                         c["offx"], c["offy"])
            bx, bz = actual_pos
            # Check if actual position matches any specified point
            matches_point = any(bx == px and bz == pz for px, pz in pos_spec)
            in_box = matches_point and (c["x1"] < bx < c["x2"] and c["z1"] < bz < c["z2"])
            # Classify variant if applicable
            if in_box and c.get("struct_type"):
                try:
                    chunk_x, chunk_z = bx >> 4, bz >> 4
                    variant = _classify_variant(s32, c["struct_type"], chunk_x, chunk_z, bx, bz, c["spacing"], c.get("variant_filter"))
                    if variant:
                        c["variants"][actual_pos] = variant
                except Exception:
                    pass
            positions.append(((rx, rz), actual_pos, in_box))
            if in_box:
                found += 1

    return positions, found


def _biome_passes(gen, pos, biomes, corner_check, offx, offy):
    bx, bz = pos
    bid  = gen.biome_at_block(bx, bz)
    name = gen.biome_name(bid)
    if bid not in biomes:
        return False, name                  # fast exit
    if corner_check:
        cx0 = bx - offx
        cz0 = bz - offy
        if not (
            gen.biome_at_block(cx0,      cz0)      in biomes
            and gen.biome_at_block(cx0 + 16, cz0)      in biomes
            and gen.biome_at_block(cx0,      cz0 + 16) in biomes
            and gen.biome_at_block(cx0 + 16, cz0 + 16) in biomes
        ):
            return False, name
    return True, name


def _check_biomes(gen, struct_constraints, all_positions, biome_constraints):
    per_struct = []
    for i, c in enumerate(struct_constraints):
        if not c.get("quadrant_biomes"):
            per_struct.append(None)
            continue

        pos_list  = all_positions[i]
        n_in_box  = sum(1 for _, _, ib in pos_list if ib)
        found     = 0
        pos_biome = {}
        seen      = 0

        for quad, pos, in_box in pos_list:
            if not in_box:
                continue
            biomes = c["quadrant_biomes"].get(quad)
            if biomes is None:
                # No biome check for this quadrant
                pos_biome[pos] = "no_check"
                found += 1
                seen += 1
                continue
            ok, name = _biome_passes(gen, pos, biomes, c["corner_check"],
                                     c["offx"], c["offy"])
            pos_biome[pos] = name
            if ok:
                found += 1
            # Early exit: remaining positions cannot make up the difference
            if found + (n_in_box - seen - 1) < c["occurence"]:
                return False, None, None
            seen += 1

        if found < c["occurence"]:
            return False, None, None
        per_struct.append(pos_biome)

    per_biome = []
    for bc in biome_constraints:
        bid  = gen.biome_at_block(bc["x"], bc["z"], bc["y"])
        name = gen.biome_name(bid)
        if bid not in bc["allowed"]:
            return False, None, None
        per_biome.append(name)

    return True, per_struct, per_biome


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_result(seed_out, struct_constraints, all_positions,
                   biome_constraints, per_struct_biome, per_biome_names):
    n_struct = len(struct_constraints)
    n_biome  = len(biome_constraints)

    if n_struct == 1 and n_biome == 0:
        # ---- compact single-constraint format --------------------------------
        pb = (per_struct_biome[0] or {}) if per_struct_biome else {}
        c = struct_constraints[0]
        parts = []
        for quad, pos, in_box in all_positions[0]:
            if in_box:
                label_parts = [str(pos)]
                if pos in pb:
                    label_parts.append(f"[{pb[pos]}]")
                if pos in c.get("variants", {}):
                    label_parts.append(f"({c['variants'][pos]})")
                parts.append(" ".join(label_parts))
            else:
                parts.append(f"{str(pos):<10}")
        return f"Seed {seed_out}: {' '.join(parts)}"

    # ---- multi-constraint format --------------------------------------------
    lines = [f"Seed {seed_out}:"]
    for i, c in enumerate(struct_constraints):
        pb = (per_struct_biome[i] or {}) if per_struct_biome else {}
        tokens = []
        for quad, pos, in_box in all_positions[i]:
            if not in_box:
                continue
            label_parts = [str(pos)]
            if pos in pb:
                label_parts.append(f"[{pb[pos]}]")
            if pos in c.get("variants", {}):
                label_parts.append(f"({c['variants'][pos]})")
            tokens.append(" ".join(label_parts))
        lines.append(f"  [{c['label']}] {' '.join(tokens)}")
    for i, bc in enumerate(biome_constraints):
        name = per_biome_names[i] if per_biome_names else "?"
        lines.append(f"  [{bc['label']}] {name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def seedsearch():
    print(BANNER)

    # ---- seed range ---------------------------------------------------------
    raw_start = input("SeedStart [0]: ").strip()
    seedstart = int(raw_start) if raw_start else 0
    raw_end = input(f"SeedEnd [{1 << 32}]: ").strip()
    seedend = int(raw_end) if raw_end else (1 << 32)

    # ---- output destination -------------------------------------------------
    print()
    out_raw = input("Output to (f)ile or (c)onsole? ").strip().lower()
    to_console = out_raw in ("c", "console")
    if to_console:
        output_file = None
        print("  Results will be printed to the console.")
    else:
        output_file = input("  File name: ").strip() or "seed_results.txt"
        print(f"  Results will be saved to '{output_file}'.")

    # ---- collect constraints ------------------------------------------------
    print()
    print("Enter search constraints.  The first constraint must be a structure.")
    print("After each, type '+' to add another, or press Enter to start the scan.")

    constraints    = []
    needs_biome_gen = False
    n_struct = 0
    n_biome_pt = 0

    # First constraint is always a structure
    c, nbg = _prompt_structure_constraint(1)
    if c is None:
        print("Constraint skipped. Exiting.")
        return
    constraints.append(c)
    n_struct += 1
    if nbg:
        needs_biome_gen = True

    # Additional constraints
    while True:
        print()
        raw = input("Type '+' to add another constraint, "
                    "or press Enter to start scan: ").strip()
        if raw != "+":
            break

        ctype_raw = input(
            "  Constraint type — (s)tructure or (b)iome point [s]: "
        ).strip().lower() or "s"

        if ctype_raw in ("s", "structure"):
            n_struct += 1
            c, nbg = _prompt_structure_constraint(n_struct)
            if c is not None:
                if nbg:
                    needs_biome_gen = True
                constraints.append(c)
        else:
            n_biome_pt += 1
            c = _prompt_biome_constraint(n_biome_pt)
            if c is not None:
                needs_biome_gen = True
                constraints.append(c)

    # Separate into structure / biome-point lists in entry order
    struct_constraints = [c for c in constraints if c["type"] == "structure"]
    biome_constraints  = [c for c in constraints if c["type"] == "biome"]

    # ---- expansion mode (asked once, after constraints) ---------------------
    expand_mode = False
    expand_top_count = 0
    expand_stop_on_matches = 1
    if needs_biome_gen:
        print()
        ans = input(
            "Enable 32-bit structure scan with 32-bit biome expansion?\n"
        ).strip().lower()
        expand_mode = ans in ("y", "yes")
        if expand_mode:
            raw_top = input("  Test first N upper 32-bit values [65535]: ").strip()
            expand_top_count = int(raw_top) if raw_top else 65535
            raw_stop = input("  Stop after N matched biome seeds [1]: ").strip()
            expand_stop_on_matches = int(raw_stop) if raw_stop else 1
            if expand_stop_on_matches < 0:
                expand_stop_on_matches = 0
        print("  Expansion mode ON." if expand_mode else "  Expansion mode OFF.")

    # ---- biome generator ---------------------------------------------------
    biome_gen = None
    if needs_biome_gen:
        biome_gen = bm.BiomeGenerator(mc_version=bm.MC_1_21, dim=bm.DIM_OVERWORLD)
        print("  Biome generator ready (MC 1.21, overworld).")

    print()

    # ---- build header -------------------------------------------------------
    mode_label = ("32-bit structure + 32-bit biome expansion"
                  if expand_mode else "standard scan")
    hdr_lines = [
        f"# Mode: {mode_label}",
        f"# Range [{seedstart}, {seedend})",
    ]
    for i, c in enumerate(struct_constraints):
        hdr_lines.append(
            f"# Structure {i+1} [{c['label']}]: "
            f"spacing={c['spacing']} separation={c['separation']} "
            f"salt={c['salt']} linear={int(c['linear_sep'])}"
        )
        hdr_lines.append(
            f"#   Bounds: ({c['x1']},{c['z1']}) to ({c['x2']},{c['z2']})"
            f"  occurence={c['occurence']}  offset=({c['offx']},{c['offy']})"
        )
        if c.get("specific_quadrants"):
            quads = ", ".join(f"({rx},{rz})" for rx, rz in c["specific_quadrants"])
            hdr_lines.append(f"#   Specific quadrants: {quads}")
        if c.get("specific_positions"):
            for quad, positions in c["specific_positions"].items():
                if positions is None:
                    continue
                if isinstance(positions, tuple) and len(positions) == 4:
                    x1r, z1r, x2r, z2r = positions
                    hdr_lines.append(f"#   Position range for {quad}: ({x1r},{z1r})-({x2r},{z2r})")
                else:
                    pos_str = ", ".join(f"({x},{z})" for x, z in positions)
                    hdr_lines.append(f"#   Positions for {quad}: {pos_str}")
        if c.get("quadrant_biomes"):
            for quad, biomes in c["quadrant_biomes"].items():
                if biomes is not None:
                    labels = ", ".join(
                        bm.BIOME_NAMES.get(b, str(b)) for b in sorted(biomes)
                    )
                    hdr_lines.append(
                        f"#   Biome filter for {quad}: [{labels}]"
                        f"  corner_check={c['corner_check']}"
                    )
    for bc in biome_constraints:
        labels = ", ".join(
            bm.BIOME_NAMES.get(b, str(b)) for b in sorted(bc["allowed"])
        )
        hdr_lines.append(
            f"# Biome point [{bc['label']}] "
            f"y={bc['y']}: [{labels}]"
        )
    header = "\n".join(hdr_lines)

    # ---- helpers ------------------------------------------------------------
    def emit(line, f=None):
        if to_console:
            print(line, flush=True)
        else:
            f.write(line + "\n")
            f.flush()

    # ---- primary constraint for JIT ----------------------------------------
    primary = struct_constraints[0]
    effective_radius = max(
        abs(primary["x1"]), abs(primary["x2"]),
        abs(primary["z1"]), abs(primary["z2"]),
    )

    # ---- main scan loop -----------------------------------------------------
    def run(f=None):
        emit(header, f)
        emit("", f)

        # Check if primary constraint is stronghold (no JIT support)
        if primary.get("struct_type") == "stronghold":
            print("Note: stronghold search is handled via Python (no JIT kernel).\n", flush=True)
            times = time.time()
            total_matched = 0
            s = seedstart
            
            while s < seedend:
                s32 = s & MASK32
                
                # Check structural constraints
                all_positions = []
                pass_all = True
                for i, c in enumerate(struct_constraints):
                    positions, found = _check_struct_positions(s32, c, biome_gen)
                    if found < c["occurence"]:
                        pass_all = False
                        break
                    all_positions.append(positions)
                
                if pass_all:
                    # Check biome constraints if generator exists
                    if biome_constraints and not biome_gen:
                        # Biome constraints specified but generator not initialized
                        s += 1
                        continue
                    if biome_constraints and biome_gen and not _check_biomes(biome_gen, struct_constraints, all_positions, biome_constraints):
                        s += 1
                        continue
                    
                    # Seed passed all checks
                    total_matched += 1
                    result_str = _format_result(s32, struct_constraints, all_positions,
                                               biome_constraints, None, None)
                    emit(result_str, f)
                
                s += 1
            
            elapsed = time.time() - times
            emit(f"\n# Finished.  Time: {elapsed:.2f}s", f)
            emit(f"# Seeds checked: {seedend - seedstart}", f)
            emit(f"# Matches found: {total_matched}", f)
            return

        # Standard JIT-based search for non-stronghold structures
        print("Compiling search kernel...", flush=True)
        scan_batch(0, 1,
                   primary["spacing"], primary["separation"], primary["salt"],
                   primary["linear_sep"], effective_radius, primary["occurence"])
        print("Ready — starting scan.\n", flush=True)

        times         = time.time()
        BATCH         = 10_000_000
        s             = seedstart
        total_jit     = 0   # seeds from JIT kernel (before Python box filter)
        total_struct   = 0   # seeds passing all structure constraint box checks
        total_matched  = 0   # seeds passing all constraints including biome
        batch_struct   = 0   # seeds in current batch passing structure checks

        while s < seedend:
            batch_end = min(s + BATCH, seedend)
             # --- progress report ---
            elapsed = time.time() - times
            if needs_biome_gen:
                prog = (
                    f"[Progress] scanned up to {s}"
                    f"  elapsed={elapsed:.1f}s"
                )
            elif batch_end >= BATCH:
                prog = (
                    f"[Progress] scanned up to {batch_end - BATCH}"
                    f"  elapsed={elapsed:.1f}s"
                    f"  hits={batch_struct}  (total={total_struct})"
                )
            else:
                prog = (
                    "Starting Search..."
                )
            print(prog, flush=True)
            if not to_console and f:
                f.write(prog + "\n")
                f.flush()
            # --- JIT scan (primary structure constraint, enlarged radius) ---
            jit_hits = scan_batch(
                s, batch_end,
                primary["spacing"], primary["separation"], primary["salt"],
                primary["linear_sep"], effective_radius, primary["occurence"],
            )
            total_jit   += len(jit_hits)
            batch_struct  = 0
            batch_matched = 0

            for seed_val_raw in jit_hits:
                s32 = int(seed_val_raw)

                # ---- check all structure constraints at Python level --------
                all_positions = []
                struct_ok = True

                for i, c in enumerate(struct_constraints):
                    positions, found = _check_struct_positions(s32, c, biome_gen)
                    all_positions.append(positions)
                    if found < c["occurence"]:
                        struct_ok = False
                        break

                if not struct_ok:
                    continue
                batch_struct += 1

                # ---- biome checks ------------------------------------------
                if biome_gen is None:
                    # No biome constraints: emit directly
                    batch_matched += 1
                    emit(_format_result(s32, struct_constraints, all_positions,
                                        biome_constraints, None, None), f)

                elif expand_mode:
                    # Try the first N upper 32-bit values
                    s32_masked = s32 & MASK32
                    matched_for_seed = 0
                    for top in range(expand_top_count):
                        full_seed = (top << 32) | s32_masked
                        if full_seed >= SIGN_BIT:
                            full_seed -= TWO64
                        biome_gen.apply_seed(full_seed)

                        ok, per_struct_biome, per_biome_names = _check_biomes(
                            biome_gen,
                            struct_constraints, all_positions,
                            biome_constraints,
                        )
                        if ok:
                            batch_matched += 1
                            matched_for_seed += 1
                            emit(_format_result(
                                full_seed, struct_constraints, all_positions,
                                biome_constraints, per_struct_biome, per_biome_names,
                            ), f)
                            if expand_stop_on_matches and matched_for_seed >= expand_stop_on_matches:
                                break

                else:
                    biome_gen.apply_seed(s32)
                    ok, per_struct_biome, per_biome_names = _check_biomes(
                        biome_gen,
                        struct_constraints, all_positions,
                        biome_constraints,
                    )
                    if ok:
                        batch_matched += 1
                        emit(_format_result(
                            s32, struct_constraints, all_positions,
                            biome_constraints, per_struct_biome, per_biome_names,
                        ), f)

            total_struct  += batch_struct
            total_matched += batch_matched

            s = batch_end

        elapsed = time.time() - times
        emit(f"\n# Finished.  Time: {elapsed:.2f}s", f)
        if not to_console:
            print(f"Done. Results saved to '{output_file}'.  Time: {elapsed:.2f}s",
                  flush=True)

    # ---- dispatch -----------------------------------------------------------
    if to_console:
        run()
    else:
        with open(output_file, "w") as f:
            run(f)

    # ---- wait for user before exiting (for console output) -----------------
    print("\nScan complete. Press Enter to exit...")
    input()


if __name__ == "__main__":
    seedsearch()
