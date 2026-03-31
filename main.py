"""
main.py — Seed search loop and all user-facing output.
"""

import sys
import time
import biome as bm
from structure import getpos

# Force line-buffered stdout so progress lines appear immediately in the
# workflow console (Python defaults to block-buffered when not a real TTY).
sys.stdout.reconfigure(line_buffering=True)


BANNER = """
Minecraft brute-force seed searcher designed for bedrock edition structures.

RNG constants  (Format: Spacing, Separation, Salt, Linear Separation)
  Bastion/Fortress:      30,  4, 30084232,  0
  Village:               34,  8, 10387312,  1
  Pillager Outpost:      80, 24, 165745296, 1
  Woodland Mansion:      80, 20, 10387319,  1
  Ocean Monument:        32,  5, 10387313,  1
  Shipwreck:             24,  4, 165745295, 0
  Ruined Portal:         40, 15, 40552231,  0
  Temples (dependant on biome):       32,  8, 14357617,  0

Search radius:   accepts a hit if the position is within this many blocks of
                 the origin on both axes.
Min occurrence:  how many of the 4 checked regions must have a valid hit (max 4).
                 With biome validation ON a hit only counts if the biome at the
                 computed position is in the allowed set.
"""

MASK48 = (1 << 48) - 1


def seedsearch():
    print(BANNER)

    # ---- search parameters -------------------------------------------------
    seedstart  = int(input("SeedStart: "))
    seedend    = int(input("SeedEnd: "))
    spacing    = int(input("Spacing: "))
    separation = int(input("Separation: "))
    salt       = int(input("Salt: "))
    linear_sep = bool(int(input("Linear separation: (0 or 1) ")))
    radius     = int(input("Search radius: "))
    occurence  = int(input("Min occurrence: "))

    # ---- output destination ------------------------------------------------
    print()
    out_raw = input("Output to (f)ile or (c)onsole? ").strip().lower()
    to_console = out_raw in ("c", "console")
    if to_console:
        output_file = None
        print("  Results will be printed to the console.")
    else:
        output_file = input("  File name: ").strip() or "seed_results.txt"
        print(f"  Results will be saved to '{output_file}'.")

    # ---- biome validation --------------------------------------------------
    effective_biomes = bm.prompt_biome_validation()  # frozenset[int] | None

    biome_gen = None
    if effective_biomes is not None:
        biome_gen = bm.BiomeGenerator(mc_version=bm.MC_1_21, dim=bm.DIM_OVERWORLD)
        print("  Biome generator ready (MC 1.21, overworld).")

    # ---- 48-bit structure + 16-bit biome expansion mode --------------------
    # Structure positions only depend on the lower 48 bits of the seed.
    # For each 48-bit structural match, iterate all 65536 top-bit values and
    # check biome with the full 64-bit seed. Seeds that fail the structure
    # check are skipped immediately without any biome lookups.
    expand_16 = False
    if effective_biomes is not None:
        print()
        ans = input(
            "Enable 48-bit structure scan with 16-bit biome expansion?\n"
            "  (Scans lower-48-bit seeds for structure, then tries all 65536\n"
            "   top-bit variants for biome — much faster for full-space searches)\n"
            "  (y/n) [n]: "
        ).strip().lower()
        expand_16 = ans in ("y", "yes")
        if expand_16:
            print("  Expansion mode ON — SeedStart/SeedEnd treated as 48-bit values.")
        else:
            print("  Expansion mode OFF — seeds scanned as-is.")
    print()

    # ---- header ------------------------------------------------------------
    mode_label = "48-bit structure + 16-bit biome expansion" if expand_16 else "standard scan"
    header = (
        f"# Mode: {mode_label}\n"
        f"# Range [{seedstart}, {seedend})  spacing={spacing} "
        f"separation={separation} salt={salt} linear={int(linear_sep)}\n"
        f"# Radius={radius}  min_occurrence={occurence}"
    )
    if effective_biomes is not None:
        labels = ", ".join(bm.BIOME_NAMES.get(b, str(b)) for b in sorted(effective_biomes))
        header += f"\n# Biome filter: [{labels}]"

    # ---- helpers -----------------------------------------------------------
    def emit(line, f=None):
        """Write a line to the file, or print it if in console mode."""
        if to_console:
            print(line, flush=True)
        else:
            f.write(line + "\n")
            f.flush()

    def format_result(seed_out, positions_in_radius, pos_biome):
        """Build the result line for one seed."""
        parts = []
        for pos, in_rad in positions_in_radius:
            if in_rad and pos in pos_biome:
                pos_str   = f"{pos}".ljust(10)
                biome_str = f"[{pos_biome[pos]}]".ljust(18)
                parts.append(f"{pos_str}  {biome_str}")
            else:
                parts.append(f"{str(pos)}".ljust(10))
        return f"Seed {seed_out}: {' '.join(parts)}"

    # ---- main scan loop ----------------------------------------------------
    def run(f=None):
        emit(header, f)
        emit("", f)

        times = time.time()

        for s48 in range(seedstart, seedend):

            # periodic progress — must be before any continue so it always fires
            if s48 % 1_000_000 == 0 and s48 != seedstart:
                elapsed = time.time() - times
                prog = f"[Progress] scanned up to {s48}  elapsed={elapsed:.1f}s"
                print(prog, flush=True)
                if not to_console and f:
                    f.write(prog + "\n")
                    f.flush()

            # ------------------------------------------------------------------
            # Structure check — uses the 48-bit seed (top bits don't affect it)
            # ------------------------------------------------------------------
            i = getpos(s48,  0,  0, spacing, separation, salt, linear_sep)
            j = getpos(s48, -1,  0, spacing, separation, salt, linear_sep)
            k = getpos(s48,  0, -1, spacing, separation, salt, linear_sep)
            l = getpos(s48, -1, -1, spacing, separation, salt, linear_sep)

            i_in = -radius < i[0] < radius and -radius < i[1] < radius
            j_in = -radius < j[0] < radius and -radius < j[1] < radius
            k_in = -radius < k[0] < radius and -radius < k[1] < radius
            l_in = -radius < l[0] < radius and -radius < l[1] < radius

            radius_count = i_in + j_in + k_in + l_in
            if radius_count < occurence:
                continue  # structure doesn't match — skip all top-bit variants

            positions_in_radius = [(i, i_in), (j, j_in), (k, k_in), (l, l_in)]

            if expand_16 and biome_gen is not None:
                for top in range(0x10000):
                    full_seed = (top << 48) | (s48 & MASK48)
                    # convert unsigned 64-bit to signed (Minecraft seeds are signed int64)
                    if full_seed >= (1 << 63):
                        full_seed -= (1 << 64)
                    biome_gen.apply_seed(full_seed)

                    found = 0
                    pos_biome: dict[tuple, str] = {}
                    for pos, in_rad in positions_in_radius:
                        if not in_rad:
                            continue
                        bid  = biome_gen.biome_at_block(pos[0], pos[1])
                        name = biome_gen.biome_name(bid)
                        pos_biome[pos] = name
                        if bid in effective_biomes:
                            found += 1

                    if found >= occurence:
                        emit(format_result(full_seed, positions_in_radius, pos_biome), f)

            else:
                pos_biome: dict[tuple, str] = {}

                if biome_gen is not None:
                    biome_gen.apply_seed(s48)
                    found = 0
                    for pos, in_rad in positions_in_radius:
                        if not in_rad:
                            continue
                        bid  = biome_gen.biome_at_block(pos[0], pos[1])
                        name = biome_gen.biome_name(bid)
                        pos_biome[pos] = name
                        if bid in effective_biomes:
                            found += 1
                else:
                    found = radius_count

                if found >= occurence:
                    emit(format_result(s48, positions_in_radius, pos_biome), f)

        elapsed = time.time() - times
        emit(f"\n# Finished scanning.  Time: {elapsed:.2f}s", f)
        if not to_console:
            print(f"Done. Results saved to '{output_file}'.  Time: {elapsed:.2f}s")

    # ---- dispatch ----------------------------------------------------------
    if to_console:
        run()
    else:
        with open(output_file, "w") as f:
            run(f)


seedsearch()
