"""
main.py — Seed search loop with layered constraints, bounding-box bounds,
          biome point-checks, and 48-bit expansion mode.

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
import biome as bm
from structure import getpos, scan_batch

sys.stdout.reconfigure(line_buffering=True)

MASK48   = (1 << 48) - 1
SIGN_BIT = 1 << 63
TWO64    = 1 << 64

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
"""

REGION_ORDER  = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
REGION_LABELS = {
    (0,  0):  "( 0, 0) [+X +Z]",
    (-1, 0):  "(-1, 0) [-X +Z]",
    (0, -1):  "( 0,-1) [+X -Z]",
    (-1,-1):  "(-1,-1) [-X -Z]",
}

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
        sp, sep, sa, ls = PRESETS[raw]
        print(f"    Loaded '{raw}': spacing={sp}, separation={sep}, "
              f"salt={sa}, linear={int(ls)}")
        return sp, sep, sa, ls, raw
    if raw:
        print(f"    Unknown preset '{raw}' — entering manually.")
    sp  = int(input("  Spacing: "))
    sep = int(input("  Separation: "))
    sa  = int(input("  Salt: "))
    ls  = bool(int(input("  Linear separation (0 or 1): ")))
    return sp, sep, sa, ls, f"{sp}/{sep}/{sa}"


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
    sp, sep, sa, ls, label = _prompt_rng()
    x1, z1, x2, z2 = _prompt_bounds(sep)

    occ_raw = input("  Min occurrence [1]: ").strip()
    occ = int(occ_raw) if occ_raw else 1

    print()
    raw_offx = input("  Chunk offset X [8]: ").strip()
    raw_offy = input("  Chunk offset Z [8]: ").strip()
    offx = int(raw_offx) if raw_offx else 8
    offy = int(raw_offy) if raw_offy else 8
    if offx != 8 or offy != 8:
        print(f"  Using offset ({offx}, {offy}).")
    else:
        print("  Using default offset (8, 8).")

    biomes = bm.prompt_biome_validation()
    corner_check = False
    needs_biome_gen = biomes is not None
    if biomes is not None:
        ans = input(
            "  4-corner biome check? (y/n) [n]\n"
            "    Checks structure position + 4 chunk corners (5 points).\n"
            "    All must be in the allowed biome set.\n"
            "    (y/n) [n]: "
        ).strip().lower()
        corner_check = ans in ("y", "yes")
        print("    4-corner check ON." if corner_check else "    4-corner check OFF.")

    return {
        "type":        "structure",
        "label":       label,
        "spacing":     sp,
        "separation":  sep,
        "salt":        sa,
        "linear_sep":  ls,
        "occurence":   occ,
        "x1": x1, "z1": z1, "x2": x2, "z2": z2,
        "offx":        offx,
        "offy":        offy,
        "biomes":      biomes,
        "corner_check": corner_check,
    }, needs_biome_gen


def _prompt_biome_constraint(idx):
    """
    Prompt for a fixed-coordinate biome constraint.

    The user enters a world position and a set of allowed biomes.
    Returns a constraint dict, or None if the user left the biome list blank.
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

def _check_struct_positions(s48, c):
    positions = []
    found = 0
    for rx, rz in ((0, 0), (-1, 0), (0, -1), (-1, -1)):
        pos = getpos(s48, rx, rz,
                     c["spacing"], c["separation"], c["salt"], c["linear_sep"],
                     c["offx"], c["offy"])
        bx, bz = pos
        in_box = c["x1"] < bx < c["x2"] and c["z1"] < bz < c["z2"]
        positions.append((pos, in_box))
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
        if c["biomes"] is None:
            per_struct.append(None)
            continue

        pos_list  = all_positions[i]
        n_in_box  = sum(1 for _, ib in pos_list if ib)
        found     = 0
        pos_biome = {}
        seen      = 0

        for pos, in_box in pos_list:
            if not in_box:
                continue
            ok, name = _biome_passes(gen, pos,
                                     c["biomes"], c["corner_check"],
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
        parts = []
        for pos, in_box in all_positions[0]:
            if in_box and pos in pb:
                parts.append(f"{str(pos):<10}  [{pb[pos]}]")
            elif in_box:
                parts.append(f"{str(pos):<10}")
            else:
                parts.append(f"{str(pos):<10}")
        return f"Seed {seed_out}: {' '.join(parts)}"

    # ---- multi-constraint format --------------------------------------------
    lines = [f"Seed {seed_out}:"]
    for i, c in enumerate(struct_constraints):
        pb = (per_struct_biome[i] or {}) if per_struct_biome else {}
        tokens = []
        for pos, in_box in all_positions[i]:
            if not in_box:
                continue
            if pos in pb:
                tokens.append(f"{pos}[{pb[pos]}]")
            else:
                tokens.append(str(pos))
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
    seedstart = int(input("SeedStart: "))
    seedend   = int(input("SeedEnd: "))

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

    # ---- expand-16 mode (asked once, after constraints) --------------------
    expand_16 = False
    if needs_biome_gen:
        print()
        ans = input(
            "Enable 48-bit structure scan with 16-bit biome expansion?\n"
            "  (Scan 48-bit seeds for structure, then test all 65536 top-bit\n"
            "   variants for biome — much faster for full-space searches.)\n"
            "  (y/n) [n]: "
        ).strip().lower()
        expand_16 = ans in ("y", "yes")
        print("  Expansion mode ON." if expand_16 else "  Expansion mode OFF.")

    # ---- biome generator ---------------------------------------------------
    biome_gen = None
    if needs_biome_gen:
        biome_gen = bm.BiomeGenerator(mc_version=bm.MC_1_21, dim=bm.DIM_OVERWORLD)
        print("  Biome generator ready (MC 1.21, overworld).")

    print()

    # ---- build header -------------------------------------------------------
    mode_label = ("48-bit structure + 16-bit biome expansion"
                  if expand_16 else "standard scan")
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
        if c["biomes"] is not None:
            labels = ", ".join(
                bm.BIOME_NAMES.get(b, str(b)) for b in sorted(c["biomes"])
            )
            hdr_lines.append(
                f"#   Biome filter: [{labels}]"
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

        print("Compiling search kernel...", flush=True)
        scan_batch(0, 1,
                   primary["spacing"], primary["separation"], primary["salt"],
                   primary["linear_sep"], effective_radius, primary["occurence"])
        print("Ready — starting scan.\n", flush=True)

        times         = time.time()
        BATCH         = 25_000_000
        s             = seedstart
        total_jit     = 0   # seeds from JIT kernel (before Python box filter)
        total_struct   = 0   # seeds passing all structure constraint box checks
        total_matched  = 0   # seeds passing all constraints including biome

        while s < seedend:
            batch_end = min(s + BATCH, seedend)

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
                s48 = int(seed_val_raw)

                # ---- check all structure constraints at Python level --------
                all_positions = []
                struct_ok = True

                for i, c in enumerate(struct_constraints):
                    positions, found = _check_struct_positions(s48, c)
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
                    emit(_format_result(s48, struct_constraints, all_positions,
                                        biome_constraints, None, None), f)

                elif expand_16:
                    # Try all 65536 top-bit values
                    s48_masked = s48 & MASK48
                    for top in range(0x10000):
                        full_seed = (top << 48) | s48_masked
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
                            emit(_format_result(
                                full_seed, struct_constraints, all_positions,
                                biome_constraints, per_struct_biome, per_biome_names,
                            ), f)

                else:
                    biome_gen.apply_seed(s48)
                    ok, per_struct_biome, per_biome_names = _check_biomes(
                        biome_gen,
                        struct_constraints, all_positions,
                        biome_constraints,
                    )
                    if ok:
                        batch_matched += 1
                        emit(_format_result(
                            s48, struct_constraints, all_positions,
                            biome_constraints, per_struct_biome, per_biome_names,
                        ), f)

            total_struct  += batch_struct
            total_matched += batch_matched

            # --- progress report ---
            elapsed = time.time() - times
            if needs_biome_gen:
                prog = (
                    f"[Progress] scanned up to {batch_end}"
                    f"  elapsed={elapsed:.1f}s"
                    f"  struct_hits={batch_struct}  validated={batch_matched}"
                    f"  (total struct={total_struct}, total validated={total_matched})"
                )
            else:
                prog = (
                    f"[Progress] scanned up to {batch_end}"
                    f"  elapsed={elapsed:.1f}s"
                    f"  hits={batch_struct}  (total={total_struct})"
                )
            print(prog, flush=True)
            if not to_console and f:
                f.write(prog + "\n")
                f.flush()

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


if __name__ == "__main__":
    seedsearch()
