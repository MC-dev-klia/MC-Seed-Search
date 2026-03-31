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

# ---------------------------------------------------------------------------
# Structure presets  (spacing, separation, salt, linear_sep)
# Multiple aliases map to the same values.
# ---------------------------------------------------------------------------
PRESETS = {
    "bastion":          (30,  4, 30084232,  False),
    "fortress":         (30,  4, 30084232,  False),
    "bastion/fortress": (30,  4, 30084232,  False),
    "village":          (34,  8, 10387312,  True),
    "pillager":         (80, 24, 165745296, True),
    "outpost":          (80, 24, 165745296, True),
    "pillager outpost": (80, 24, 165745296, True),
    "mansion":          (80, 20, 10387319,  True),
    "woodland mansion": (80, 20, 10387319,  True),
    "monument":         (32,  5, 10387313,  True),
    "ocean monument":   (32,  5, 10387313,  True),
    "shipwreck":        (24,  4, 165745295, False),
    "portal":           (40, 15, 40552231,  False),
    "ruined portal":    (40, 15, 40552231,  False),
    "temple":           (32,  8, 14357617,  False),
    "temples":          (32,  8, 14357617,  False),
}

PRESET_NAMES = [
    "bastion/fortress", "village", "outpost",
    "mansion", "monument", "shipwreck", "portal", "temple",
]


def _prompt_structure_constants():
    """
    Ask for structure RNG constants via a named preset or manual entry.
    Returns (spacing, separation, salt, linear_sep).
    """
    print("\nAvailable presets:")
    print("  " + ", ".join(PRESET_NAMES))
    print("  (type a preset name, or press Enter to enter values manually)\n")

    raw = input("Structure preset (or Enter to skip): ").strip().lower()

    if raw in PRESETS:
        sp, sep, salt, ls = PRESETS[raw]
        print(f"  Loaded '{raw}': spacing={sp}, separation={sep}, "
              f"salt={salt}, linear_sep={int(ls)}")
        return sp, sep, salt, ls

    if raw:
        print(f"  Unknown preset '{raw}' — entering values manually.")

    spacing    = int(input("Spacing: "))
    separation = int(input("Separation: "))
    salt       = int(input("Salt: "))
    linear_sep = bool(int(input("Linear separation: (0 or 1) ")))
    return spacing, separation, salt, linear_sep


def seedsearch():
    print(BANNER)

    # ---- search parameters -------------------------------------------------
    seedstart  = int(input("SeedStart: "))
    seedend    = int(input("SeedEnd: "))
    spacing, separation, salt, linear_sep = _prompt_structure_constants()
    radius     = int(input("Search radius: "))
    occurence  = int(input("Min occurrence: "))

    # ---- chunk offset -------------------------------------------------------
    print()
    print("Structure chunk offset (default 8, 8 — chunk centre).")
    raw_offx = input("  Offset X [8]: ").strip()
    raw_offy = input("  Offset Z [8]: ").strip()
    offx = int(raw_offx) if raw_offx else 8
    offy = int(raw_offy) if raw_offy else 8
    if offx != 8 or offy != 8:
        print(f"  Using offset ({offx}, {offy}).")
    else:
        print(f"  Using default offset (8, 8).")

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

    # ---- 4-corner biome check -----------------------------------------------
    corner_check = False
    if effective_biomes is not None:
        print()
        ans_c = input(
            "Enable 4-corner biome check? (y/n) [n]\n"
            "  Checks 4 chunk corners + structure position (5 points total).\n"
            "  All must be valid — slower but avoids false positives near biome edges.\n"
            "  (y/n) [n]: "
        ).strip().lower()
        corner_check = ans_c in ("y", "yes")
        print("  4-corner biome check ON." if corner_check else "  4-corner biome check OFF.")

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
        f"  offset=({offx},{offy})"
    )
    if effective_biomes is not None:
        labels = ", ".join(bm.BIOME_NAMES.get(b, str(b)) for b in sorted(effective_biomes))
        header += f"\n# Biome filter: [{labels}]"
        header += f"  corner_check={corner_check}"

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

    # ---- biome helper (captures corner_check / offx / offy from outer scope) -
    def biome_passes(gen, pos):
        """
        Check biome validity at a structure position.

        The structure position itself is always checked first — it is the
        cheapest single-point query and the most likely to fail for rare
        biomes.  When it fails we return immediately without paying for any
        corner calls.  When corner_check=True and the structure point passes,
        we then check the four chunk-corner points.

        Returns (passes: bool, biome_name: str).
        """
        bx, bz = pos

        # --- Structure position (fast first gate) ---
        struct_bid = gen.biome_at_block(bx, bz)
        name = gen.biome_name(struct_bid)
        if struct_bid not in effective_biomes:
            return False, name          # early exit — skip corner work

        # --- Optional 4-corner check (only reached if structure point passed) ---
        if corner_check:
            cx0 = bx - offx            # chunk-origin X (block coords)
            cz0 = bz - offy            # chunk-origin Z
            if not (
                gen.biome_at_block(cx0,      cz0)      in effective_biomes
                and gen.biome_at_block(cx0 + 16, cz0)      in effective_biomes
                and gen.biome_at_block(cx0,      cz0 + 16) in effective_biomes
                and gen.biome_at_block(cx0 + 16, cz0 + 16) in effective_biomes
            ):
                return False, name

        return True, name

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
        total_struct  = 0  # seeds passing the fast structure scan
        total_matched = 0  # seeds that also passed the biome filter

        while s < seedend:
            batch_end = min(s + BATCH, seedend)

            # --- fast numba structure scan — returns only seeds that pass ---
            hits = scan_batch(s, batch_end, spacing, separation, salt,
                              linear_sep, radius, occurence)
            total_struct += len(hits)
            batch_matched = 0

            # --- for each structural hit, compute positions + biome check ---
            for seed_val_raw in hits:
                s48 = int(seed_val_raw)

                pi = getpos(s48,  0,  0, spacing, separation, salt, linear_sep, offx, offy)
                pj = getpos(s48, -1,  0, spacing, separation, salt, linear_sep, offx, offy)
                pk = getpos(s48,  0, -1, spacing, separation, salt, linear_sep, offx, offy)
                pl = getpos(s48, -1, -1, spacing, separation, salt, linear_sep, offx, offy)

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
                            ok, name = biome_passes(biome_gen, pos)
                            pos_biome[pos] = name
                            if ok:
                                found += 1

                        if found >= occurence:
                            emit(format_result(full_seed, positions_in_radius, pos_biome), f)
                            batch_matched += 1

                else:
                    pos_biome: dict[tuple, str] = {}

                    if biome_gen is not None:
                        biome_gen.apply_seed(s48)
                        found = 0
                        for pos, in_rad in positions_in_radius:
                            if not in_rad:
                                continue
                            ok, name = biome_passes(biome_gen, pos)
                            pos_biome[pos] = name
                            if ok:
                                found += 1
                    else:
                        found = i_in + j_in + k_in + l_in

                    if found >= occurence:
                        emit(format_result(s48, positions_in_radius, pos_biome), f)
                        batch_matched += 1

            total_matched += batch_matched

            # --- progress after each batch ---
            elapsed = time.time() - times
            if effective_biomes is not None:
                prog = (
                    f"[Progress] scanned up to {batch_end}  elapsed={elapsed:.1f}s"
                    f"  struct_hits={len(hits)}  validated={batch_matched}"
                    f"  (total struct={total_struct}, total validated={total_matched})"
                )
            else:
                prog = (
                    f"[Progress] scanned up to {batch_end}  elapsed={elapsed:.1f}s"
                    f"  hits={len(hits)}  (total={total_struct})"
                )
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
