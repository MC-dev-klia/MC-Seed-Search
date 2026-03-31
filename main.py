"""
main.py — Seed search loop and all user-facing output.
"""

import sys
import time
import biome as bm
from structure import getpos, scan_batch

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

        # Warm up the numba JIT so compilation time is not counted in scan time
        print("Compiling search kernel...", flush=True)
        scan_batch(0, 1, spacing, separation, salt, linear_sep, radius, occurence)
        print("Ready — starting scan.\n", flush=True)

        times = time.time()
        BATCH = 25_000_000
        s = seedstart

        while s < seedend:
            batch_end = min(s + BATCH, seedend)

            # --- fast numba structure scan — returns only seeds that pass ---
            hits = scan_batch(s, batch_end, spacing, separation, salt,
                              linear_sep, radius, occurence)

            # --- for each structural hit, compute positions + biome check ---
            for seed_val_raw in hits:
                s48 = int(seed_val_raw)

                pi = getpos(s48,  0,  0, spacing, separation, salt, linear_sep)
                pj = getpos(s48, -1,  0, spacing, separation, salt, linear_sep)
                pk = getpos(s48,  0, -1, spacing, separation, salt, linear_sep)
                pl = getpos(s48, -1, -1, spacing, separation, salt, linear_sep)

                i_in = -radius < pi[0] < radius and -radius < pi[1] < radius
                j_in = -radius < pj[0] < radius and -radius < pj[1] < radius
                k_in = -radius < pk[0] < radius and -radius < pk[1] < radius
                l_in = -radius < pl[0] < radius and -radius < pl[1] < radius
                positions_in_radius = [(pi, i_in), (pj, j_in), (pk, k_in), (pl, l_in)]

                if expand_16 and biome_gen is not None:
                    for top in range(0x10000):
                        full_seed = (top << 48) | (s48 & MASK48)
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
                        found = i_in + j_in + k_in + l_in

                    if found >= occurence:
                        emit(format_result(s48, positions_in_radius, pos_biome), f)

            # --- progress after each batch ---
            elapsed = time.time() - times
            prog = f"[Progress] scanned up to {batch_end}  elapsed={elapsed:.1f}s  hits={len(hits)}"
            print(prog, flush=True)
            if not to_console and f:
                f.write(prog + "\n")
                f.flush()

            s = batch_end

        elapsed = time.time() - times
        emit(f"\n# Finished scanning.  Time: {elapsed:.2f}s", f)
        if not to_console:
            print(f"Done. Results saved to '{output_file}'.  Time: {elapsed:.2f}s", flush=True)

    # ---- dispatch ----------------------------------------------------------
    if to_console:
        run()
    else:
        with open(output_file, "w") as f:
            run(f)


seedsearch()
