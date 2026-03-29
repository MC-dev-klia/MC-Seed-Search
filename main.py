from typing import List, Tuple
import time
from math import floor
import numba as nb
import numpy as np
import biome as bm


MASK32 = 0xffffffff
N = 624
M = 397
MATRIX_A = 0x9908b0df
UPPER_MASK = 0x80000000
LOWER_MASK = 0x7fffffff

@nb.njit(cache=True)
def mt_init(seed32):
    mt = np.empty(N, dtype=np.uint32)
    mt[0] = seed32
    for i in range(1, N):
        mt[i] = (0x6c078965 * (mt[i-1] ^ (mt[i-1] >> 30)) + i) & MASK32
    return mt

@nb.njit(cache=True)
def mt_twist(mt):
    for i in range(N):
        y = (mt[i] & UPPER_MASK) | (mt[(i+1) % N] & LOWER_MASK)
        mt[i] = mt[(i + M) % N] ^ (y >> 1)
        if y & 1:
            mt[i] ^= MATRIX_A

@nb.njit(cache=True)
def mt_extract(mt, idx):
    if idx >= N:
        mt_twist(mt)
        idx = 0
    y = mt[idx]
    y ^= (y >> 11)
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= (y >> 18)
    return y & MASK32, idx+1

def getpos(world_seed, rx, rz, spacing, separation, salt, linear_separation):
    spawn_range = spacing - separation
    mixed = (world_seed + rx*341873128712 + rz*132897987541 + salt) & ((1<<64)-1)
    seed32 = mixed & 0xffffffff

    mt = mt_init(seed32)
    idx = N
    r0, idx = mt_extract(mt, idx)
    r1, idx = mt_extract(mt, idx)
    if linear_separation:
        r2, idx = mt_extract(mt, idx)
        r3, idx = mt_extract(mt, idx)
        off_x = ((r0 % spawn_range) + (r1 % spawn_range)) // 2
        off_z = ((r2 % spawn_range) + (r3 % spawn_range)) // 2
    else:
        off_x = r0 % spawn_range
        off_z = r1 % spawn_range

    chunk_x = rx * spacing + off_x
    chunk_z = rz * spacing + off_z
    return (chunk_x*16, chunk_z*16)


# -------------------------
# Seed scanning
# -------------------------
print(
"""
Minecraft Bedrock Edition — brute-force structural 48-bit seed searcher
with biome generation via cubiomes.

Bedrock and Java Edition share the same biome noise algorithm, so cubiomes
biome results apply directly to Bedrock seeds.

RNG constants  (Format: Spacing, Separation, Salt, Linear Separation)
  Bastion/Fortress:      30,  4, 30084232,  0
  Village:               34,  8, 10387312,  1
  Pillager Outpost:      80, 24, 165745296, 1
  Woodland Mansion:      80, 20, 10387319,  1
  Ocean Monument:        32,  5, 10387313,  1
  Shipwreck:             24,  4, 165745295, 1
  Ruined Portal:         40, 15, 40552231,  0
  Other Overworld:       32,  8, 14357617,  0

Search radius:   app accepts a structure hit if its position is within this
                 many blocks of the origin on both axes.
Min occurrence:  how many of the 4 checked regions must have a valid hit (max 4).
                 With structure biome validation ON a hit only counts if the
                 biome at the computed position allows the structure to spawn.
"""
)


def seedsearch():
    seedstart   = int(input("SeedStart: "))
    seedend     = int(input("SeedEnd: "))
    spacing     = int(input("Spacing: "))
    separation  = int(input("Separation: "))
    salt        = int(input("Salt: "))
    linear_sep  = bool(int(input("Linear separation: (0 or 1) ")))
    radius      = int(input("Search radius: "))
    occurence   = int(input("Min occurrence: "))
    output_file = input("Output file name (default: seed_results.txt): ").strip() \
                  or "seed_results.txt"

    # ---- structure biome validation ----------------------------------------
    struct_cfg = bm.prompt_structure_type()   # (key, valid_biomes|None) or None

    # ---- custom biome restriction at candidate positions -------------------
    biome_cfg = bm.prompt_biome_requirements()  # (mc_version, frozenset) or None

    # ---- compute the effective biome set to check per candidate position ---
    # A position counts as "found" only if its biome is in effective_biomes.
    # effective_biomes = intersection of structure valid biomes and custom biomes
    # (whichever are specified; None means no restriction from that source).
    struct_valid  = struct_cfg[1] if struct_cfg is not None else None
    custom_biomes = biome_cfg[1]  if biome_cfg  is not None else None

    if struct_valid is not None and custom_biomes is not None:
        effective_biomes = struct_valid & custom_biomes   # both must be satisfied
    elif struct_valid is not None:
        effective_biomes = struct_valid
    elif custom_biomes is not None:
        effective_biomes = custom_biomes
    else:
        effective_biomes = None   # no biome restriction

    # ---- set up generator --------------------------------------------------
    mc_version = biome_cfg[0] if biome_cfg is not None else bm.MC_1_21
    biome_gen  = None
    if effective_biomes is not None or struct_cfg is not None:
        biome_gen = bm.BiomeGenerator(mc_version=mc_version, dim=bm.DIM_OVERWORLD)
        ver_label = [k for k, v in bm.MC_VERSIONS.items() if v == mc_version][0]
        print(f"\nBiome generator ready  (MC {ver_label}, overworld)")

    print()
    times = time.time()
    seeds = range(seedstart, seedend)

    with open(output_file, 'w') as f:
        f.write(f"# Seed search results — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Range [{seedstart}, {seedend})  spacing={spacing} "
                f"separation={separation} salt={salt} linear={int(linear_sep)}\n")
        f.write(f"# Radius={radius}  min_occurrence={occurence}\n")
        if struct_cfg is not None:
            f.write(f"# Structure biome check: {struct_cfg[0]}\n")
        if custom_biomes is not None:
            labels = ", ".join(bm.BIOME_NAMES.get(b, str(b)) for b in sorted(custom_biomes))
            f.write(f"# Custom biome requirement at candidate positions: [{labels}]\n")
        f.write("\n")

        for seed in seeds:

            # --- compute the 4 candidate positions --------------------------
            i = getpos(seed,  0,  0, spacing, separation, salt, linear_sep)
            j = getpos(seed, -1,  0, spacing, separation, salt, linear_sep)
            k = getpos(seed,  0, -1, spacing, separation, salt, linear_sep)
            l = getpos(seed, -1, -1, spacing, separation, salt, linear_sep)

            # --- radius filter (fast, no biome lookup yet) ------------------
            i_in = -radius < i[0] < radius and -radius < i[1] < radius
            j_in = -radius < j[0] < radius and -radius < j[1] < radius
            k_in = -radius < k[0] < radius and -radius < k[1] < radius
            l_in = -radius < l[0] < radius and -radius < l[1] < radius

            if i_in + j_in + k_in + l_in < occurence:
                continue

            # --- biome check per candidate position -------------------------
            # A position counts as "found" only when it is in radius AND its
            # biome is in effective_biomes (structure valid ∩ custom, if set).
            pos_biome_names: dict[tuple, str] = {}

            if biome_gen is not None and effective_biomes is not None:
                biome_gen.apply_seed(seed)
                found = 0
                for pos, in_rad in [(i, i_in), (j, j_in), (k, k_in), (l, l_in)]:
                    if not in_rad:
                        continue
                    bid   = biome_gen.biome_at_block(pos[0], pos[1])
                    bname = biome_gen.biome_name(bid)
                    pos_biome_names[pos] = bname
                    if bid in effective_biomes:
                        found += 1
            else:
                found = i_in + j_in + k_in + l_in

            if found < occurence:
                continue

            # --- build output line with biome annotations -------------------
            pos_parts = []
            for pos, in_rad in [(i, i_in), (j, j_in), (k, k_in), (l, l_in)]:
                if in_rad and pos in pos_biome_names:
                    pos_parts.append(f"{pos}[{pos_biome_names[pos]}]")
                else:
                    pos_parts.append(str(pos))
            pos_str = " ".join(pos_parts)

            line = f"Seed {seed}: {pos_str}"
            f.write(line + "\n")

            if seed % 1_000_000 == 0 and seed != seedstart:
                elapsed = time.time() - times
                prog = f"[Progress] scanned up to {seed}  elapsed={elapsed:.1f}s"
                print(prog)
                f.write(prog + "\n")

        elapsed = time.time() - times
        f.write(f"\n# Finished scanning.  Time: {elapsed:.2f}s\n")

    print(f"Done. Results saved to '{output_file}'.  Time: {elapsed:.2f}s")


seedsearch()
