"""
structure.py — Bedrock Edition structural position calculations.

Implements the Mersenne Twister RNG used by Minecraft Bedrock to determine
where structures are placed within each region, the getpos() function that
combines the world seed, region coordinates, and RNG constants into a
block-level structure position, and JIT-compiled variant classifiers
(bastion vs fortress, ruined portal variants, village placement, and
stronghold placement) ported from MCBE-1.18-Seed-Finder.
"""

import math

import numba as nb
import numpy as np


# ---------------------------------------------------------------------------
# Mersenne Twister constants
# ---------------------------------------------------------------------------
MASK32     = 0xffffffff
N          = 624
M          = 397
MATRIX_A   = 0x9908b0df
UPPER_MASK = 0x80000000
LOWER_MASK = 0x7fffffff


# ---------------------------------------------------------------------------
# Region / structure salt constants
# ---------------------------------------------------------------------------
R_X_CONST = 341873128712
R_Z_CONST = 132897987541

BASTION_SALT       = 30084232
VILLAGE_SALT       = 10387312
VILLAGE_SPACING    = 34
VILLAGE_SEPARATION = 26

STRONGHOLD_GRID_SIZE = 200          # chunks
STRONGHOLD_XMUL      = -1683231919
STRONGHOLD_ZMUL      = -1100435783
STRONGHOLD_SALT      = 97858791

# Village-compatible biomes (used for stronghold biome filtering)
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
    """Check if a biome ID is compatible with village (and stronghold) spawning."""
    return biome_id in VILLAGE_BIOME_IDS


# ---------------------------------------------------------------------------
# Mersenne Twister primitives
# ---------------------------------------------------------------------------
@nb.njit(cache=True)
def mt_init(seed32):
    """Initialise a 624-element MT state from a 32-bit seed."""
    mt   = np.empty(N, dtype=np.uint32)
    mt[0] = seed32
    MULT  = np.int64(0x6c078965)
    for i in range(1, N):
        p     = mt[i - 1]
        mt[i] = np.uint32(
            (MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(i))
            & np.int64(0xFFFFFFFF)
        )
    return mt


@nb.njit(cache=True)
def mt_twist(mt):
    """Apply one full twist to the MT state array."""
    MATRIX_A_  = np.uint32(0x9908b0df)
    UPPER_MASK_ = np.uint32(0x80000000)
    LOWER_MASK_ = np.uint32(0x7fffffff)
    for i in range(N):
        y = (mt[i] & UPPER_MASK_) | (mt[(i + 1) % N] & LOWER_MASK_)
        mt[i] = mt[(i + M) % N] ^ (y >> np.uint32(1))
        if y & np.uint32(1):
            mt[i] ^= MATRIX_A_


@nb.njit(cache=True)
def mt_extract(mt, idx):
    """Extract and temper one value from the MT state, twisting when needed."""
    if idx >= N:
        mt_twist(mt)
        idx = 0
    y = mt[idx]
    y ^= (y >> np.uint32(11))
    y ^= (y << np.uint32(7))  & np.uint32(0x9D2C5680)
    y ^= (y << np.uint32(15)) & np.uint32(0xEFC60000)
    y ^= (y >> np.uint32(18))
    return y, idx + 1


# ---------------------------------------------------------------------------
# Standard / linear-separation batch scanners (primary structure constraint)
# ---------------------------------------------------------------------------
@nb.njit(cache=True, parallel=True, boundscheck=False)
def _scan_batch_standard(seeds_start, seeds_end, spacing, separation, salt,
                         radius, occurence):
    """Specialised scanner for linear_sep=False (standard two-draw placement)."""
    spawn_range = spacing - separation
    n           = seeds_end - seeds_start
    R_X         = np.int64(R_X_CONST)
    R_Z         = np.int64(R_Z_CONST)
    MULT        = np.int64(0x6c078965)
    MATRIX_A_   = np.uint32(0x9908b0df)
    UPPER_MASK_ = np.uint32(0x80000000)
    LOWER_MASK_ = np.uint32(0x7fffffff)
    T1          = np.uint32(0x9D2C5680)
    T2          = np.uint32(0xEFC60000)
    sr          = np.int64(spawn_range)
    sp          = np.int64(spacing)
    rad         = np.int64(radius)
    occ         = np.int32(occurence)

    sa = np.int64(salt)
    reg_s0 = sa
    reg_s1 = sa - R_X
    reg_s2 = sa - R_Z
    reg_s3 = sa - R_X - R_Z

    neg_sp16 = -sp * np.int64(16)
    bx_base0 = np.int64(0)
    bx_base1 = neg_sp16
    bx_base2 = np.int64(0)
    bx_base3 = neg_sp16
    bz_base0 = np.int64(0)
    bz_base1 = np.int64(0)
    bz_base2 = neg_sp16
    bz_base3 = neg_sp16

    is_hit = np.zeros(n, dtype=np.bool_)

    for ii in nb.prange(n):
        world_seed = np.int64(seeds_start) + np.int64(ii)
        found      = np.int32(0)

        mt = np.empty(M + 2, dtype=np.uint32)

        for region in range(4):
            if region == 0:
                s_off  = reg_s0
                bx_b   = bx_base0
                bz_b   = bz_base0
            elif region == 1:
                s_off  = reg_s1
                bx_b   = bx_base1
                bz_b   = bz_base1
            elif region == 2:
                s_off  = reg_s2
                bx_b   = bx_base2
                bz_b   = bz_base2
            else:
                s_off  = reg_s3
                bx_b   = bx_base3
                bz_b   = bz_base3

            s32   = np.uint32(world_seed + s_off)
            mt[0] = s32
            for k in range(1, M + 2):
                p     = mt[k - 1]
                mt[k] = np.uint32(
                    (MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(k))
                    & np.int64(0xFFFFFFFF)
                )

            y  = (mt[0] & UPPER_MASK_) | (mt[1] & LOWER_MASK_)
            v0 = mt[M] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v0 ^= MATRIX_A_
            v0 ^= v0 >> np.uint32(11)
            v0 ^= (v0 << np.uint32(7))  & T1
            v0 ^= (v0 << np.uint32(15)) & T2
            v0 ^= v0 >> np.uint32(18)

            y  = (mt[1] & UPPER_MASK_) | (mt[2] & LOWER_MASK_)
            v1 = mt[M + 1] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v1 ^= MATRIX_A_
            v1 ^= v1 >> np.uint32(11)
            v1 ^= (v1 << np.uint32(7))  & T1
            v1 ^= (v1 << np.uint32(15)) & T2
            v1 ^= v1 >> np.uint32(18)

            off_x = np.int64(v0) % sr
            off_z = np.int64(v1) % sr

            bx = bx_b + off_x * np.int64(16) + np.int64(8)
            bz = bz_b + off_z * np.int64(16) + np.int64(8)

            if -rad < bx < rad and -rad < bz < rad:
                found += np.int32(1)

            if found + np.int32(3 - region) < occ:
                break

        if found >= occ:
            is_hit[ii] = True

    count = np.int64(0)
    for ii in range(n):
        if is_hit[ii]:
            count += np.int64(1)
    result = np.empty(count, dtype=np.int64)
    ci = np.int64(0)
    for ii in range(n):
        if is_hit[ii]:
            result[ci] = np.int64(seeds_start) + np.int64(ii)
            ci += np.int64(1)
    return result


@nb.njit(cache=True, parallel=True, boundscheck=False)
def _scan_batch_linear(seeds_start, seeds_end, spacing, separation, salt,
                       radius, occurence):
    """Specialised scanner for linear_sep=True (averaged four-draw placement)."""
    spawn_range = spacing - separation
    n           = seeds_end - seeds_start
    R_X         = np.int64(R_X_CONST)
    R_Z         = np.int64(R_Z_CONST)
    MULT        = np.int64(0x6c078965)
    MATRIX_A_   = np.uint32(0x9908b0df)
    UPPER_MASK_ = np.uint32(0x80000000)
    LOWER_MASK_ = np.uint32(0x7fffffff)
    T1          = np.uint32(0x9D2C5680)
    T2          = np.uint32(0xEFC60000)
    sr          = np.int64(spawn_range)
    sp          = np.int64(spacing)
    rad         = np.int64(radius)
    occ         = np.int32(occurence)

    sa = np.int64(salt)
    reg_s0 = sa
    reg_s1 = sa - R_X
    reg_s2 = sa - R_Z
    reg_s3 = sa - R_X - R_Z

    neg_sp16 = -sp * np.int64(16)
    bx_base0 = np.int64(0)
    bx_base1 = neg_sp16
    bx_base2 = np.int64(0)
    bx_base3 = neg_sp16
    bz_base0 = np.int64(0)
    bz_base1 = np.int64(0)
    bz_base2 = neg_sp16
    bz_base3 = neg_sp16

    is_hit = np.zeros(n, dtype=np.bool_)

    for ii in nb.prange(n):
        world_seed = np.int64(seeds_start) + np.int64(ii)
        found      = np.int32(0)

        mt = np.empty(M + 4, dtype=np.uint32)

        for region in range(4):
            if region == 0:
                s_off = reg_s0
                bx_b  = bx_base0
                bz_b  = bz_base0
            elif region == 1:
                s_off = reg_s1
                bx_b  = bx_base1
                bz_b  = bz_base1
            elif region == 2:
                s_off = reg_s2
                bx_b  = bx_base2
                bz_b  = bz_base2
            else:
                s_off = reg_s3
                bx_b  = bx_base3
                bz_b  = bz_base3

            s32   = np.uint32(world_seed + s_off)
            mt[0] = s32
            for k in range(1, M + 4):
                p     = mt[k - 1]
                mt[k] = np.uint32(
                    (MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(k))
                    & np.int64(0xFFFFFFFF)
                )

            y  = (mt[0] & UPPER_MASK_) | (mt[1] & LOWER_MASK_)
            v0 = mt[M] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v0 ^= MATRIX_A_
            v0 ^= v0 >> np.uint32(11)
            v0 ^= (v0 << np.uint32(7))  & T1
            v0 ^= (v0 << np.uint32(15)) & T2
            v0 ^= v0 >> np.uint32(18)

            y  = (mt[1] & UPPER_MASK_) | (mt[2] & LOWER_MASK_)
            v1 = mt[M + 1] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v1 ^= MATRIX_A_
            v1 ^= v1 >> np.uint32(11)
            v1 ^= (v1 << np.uint32(7))  & T1
            v1 ^= (v1 << np.uint32(15)) & T2
            v1 ^= v1 >> np.uint32(18)

            y  = (mt[2] & UPPER_MASK_) | (mt[3] & LOWER_MASK_)
            v2 = mt[M + 2] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v2 ^= MATRIX_A_
            v2 ^= v2 >> np.uint32(11)
            v2 ^= (v2 << np.uint32(7))  & T1
            v2 ^= (v2 << np.uint32(15)) & T2
            v2 ^= v2 >> np.uint32(18)

            y  = (mt[3] & UPPER_MASK_) | (mt[4] & LOWER_MASK_)
            v3 = mt[M + 3] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v3 ^= MATRIX_A_
            v3 ^= v3 >> np.uint32(11)
            v3 ^= (v3 << np.uint32(7))  & T1
            v3 ^= (v3 << np.uint32(15)) & T2
            v3 ^= v3 >> np.uint32(18)

            off_x = (np.int64(v0) % sr + np.int64(v1) % sr) // np.int64(2)
            off_z = (np.int64(v2) % sr + np.int64(v3) % sr) // np.int64(2)

            bx = bx_b + off_x * np.int64(16) + np.int64(8)
            bz = bz_b + off_z * np.int64(16) + np.int64(8)

            if -rad < bx < rad and -rad < bz < rad:
                found += np.int32(1)

            if found + np.int32(3 - region) < occ:
                break

        if found >= occ:
            is_hit[ii] = True

    count = np.int64(0)
    for ii in range(n):
        if is_hit[ii]:
            count += np.int64(1)
    result = np.empty(count, dtype=np.int64)
    ci = np.int64(0)
    for ii in range(n):
        if is_hit[ii]:
            result[ci] = np.int64(seeds_start) + np.int64(ii)
            ci += np.int64(1)
    return result


def scan_batch(seeds_start, seeds_end, spacing, separation, salt,
               linear_sep, radius, occurence):
    """Python wrapper — dispatches to the correct specialised JIT kernel."""
    args = (int(seeds_start), int(seeds_end),
            int(spacing), int(separation), int(salt),
            int(radius), int(occurence))
    if linear_sep:
        return _scan_batch_linear(*args)
    return _scan_batch_standard(*args)


# (Stronghold batch prefilter `_scan_batch_stronghold` and its Python wrapper
# `scan_batch_stronghold` are defined at the bottom of this file, after the
# helper kernels they depend on.)


def getpos(world_seed, rx, rz, spacing, separation, salt, linear_separation,
           offx=8, offy=8):
    """Block-level (x, z) position of a structure candidate in region (rx, rz)."""
    spawn_range = spacing - separation
    mixed  = (world_seed + rx * R_X_CONST + rz * R_Z_CONST + salt) & ((1 << 64) - 1)
    seed32 = mixed & 0xffffffff

    mt  = mt_init(np.uint32(seed32))
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
    return (chunk_x * 16 + offx, chunk_z * 16 + offy)


# ---------------------------------------------------------------------------
# Variant classification kernels (JIT)
# ---------------------------------------------------------------------------
@nb.njit(cache=True)
def _classify_bastion_jit(world_seed_low, region_x, region_z):
    """Classify a structure at (region_x, region_z) as bastion or fortress.

    Returns (is_bastion: bool, bastion_type: int).  bastion_type is -1 if
    the structure is a fortress, otherwise 0..3 (bridge/treasure/hoglin/housing).
    """
    R_X = np.int64(R_X_CONST)
    R_Z = np.int64(R_Z_CONST)
    salt = np.int64(BASTION_SALT)
    reg_seed = np.uint32(
        (np.int64(world_seed_low) + np.int64(region_x) * R_X
         + np.int64(region_z) * R_Z + salt) & np.int64(0xffffffff)
    )

    mt = mt_init(reg_seed)
    idx = N

    _x1, idx = mt_extract(mt, idx)
    _y1, idx = mt_extract(mt, idx)
    check_val, idx = mt_extract(mt, idx)
    is_bastion = (np.int64(check_val) % np.int64(6)) >= np.int64(2)

    bastion_type = np.int64(-1)
    if is_bastion:
        _rotation, idx = mt_extract(mt, idx)
        bastion_type_raw, idx = mt_extract(mt, idx)
        bastion_type = np.int64(bastion_type_raw) % np.int64(4)

    return is_bastion, bastion_type


@nb.njit(cache=True)
def _classify_portal_jit(world_seed_low, chunk_x, chunk_z):
    """Classify a ruined portal's properties.

    Returns (underground, airpocket, rotation, mirror, giant, variant_num).
    """
    # ----- chunk seed -----
    mt0 = mt_init(np.uint32(world_seed_low))
    idx0 = N
    xMul_raw, idx0 = mt_extract(mt0, idx0)
    zMul_raw, idx0 = mt_extract(mt0, idx0)
    xMul = np.int64(xMul_raw >> np.uint32(1)) | np.int64(1)
    zMul = np.int64(zMul_raw >> np.uint32(1)) | np.int64(1)

    cs = (np.int64(chunk_x) * xMul + np.int64(chunk_z) * zMul) ^ np.int64(world_seed_low)
    chunk_seed_val = np.uint32(cs & np.int64(0xffffffff))

    mt = mt_init(chunk_seed_val)
    idx = N

    inv_2_32 = np.float64(1.0) / np.float64(4294967296.0)

    underground_raw, idx = mt_extract(mt, idx)
    underground = (np.float64(underground_raw) * inv_2_32) < 0.5

    airpocket_raw, idx = mt_extract(mt, idx)
    airpocket = (np.float64(airpocket_raw) * inv_2_32) < 0.5

    rotation_raw, idx = mt_extract(mt, idx)
    rotation = np.int64(rotation_raw) % np.int64(4)

    mirror_raw, idx = mt_extract(mt, idx)
    mirror = (np.float64(mirror_raw) * inv_2_32) >= 0.5

    giant_raw, idx = mt_extract(mt, idx)
    giant = (np.float64(giant_raw) * inv_2_32) < 0.05

    if giant:
        var_raw, idx = mt_extract(mt, idx)
        variant_num = np.int64(var_raw) % np.int64(3) + np.int64(1)
    else:
        var_raw, idx = mt_extract(mt, idx)
        variant_num = np.int64(var_raw) % np.int64(10) + np.int64(1)

    return underground, airpocket, rotation, mirror, giant, variant_num


@nb.njit(cache=True)
def _check_village_at_chunk_jit(world_seed_low, chunk_x, chunk_z):
    """Return True if a village generates exactly at the given chunk."""
    R_X = np.int64(R_X_CONST)
    R_Z = np.int64(R_Z_CONST)
    salt = np.int64(VILLAGE_SALT)
    spacing = np.int64(VILLAGE_SPACING)
    separation = np.int64(VILLAGE_SEPARATION)

    cx = np.int64(chunk_x)
    cz = np.int64(chunk_z)

    reg_x = cx // spacing
    reg_z = cz // spacing

    reg_seed = np.uint32(
        (np.int64(world_seed_low) + reg_x * R_X + reg_z * R_Z + salt)
        & np.int64(0xffffffff)
    )

    mt = mt_init(reg_seed)
    idx = N

    r0, idx = mt_extract(mt, idx)
    r1, idx = mt_extract(mt, idx)
    local_x = ((np.int64(r0) % separation) + (np.int64(r1) % separation)) // np.int64(2)

    r2, idx = mt_extract(mt, idx)
    r3, idx = mt_extract(mt, idx)
    local_z = ((np.int64(r2) % separation) + (np.int64(r3) % separation)) // np.int64(2)

    village_chunk_x = reg_x * spacing + local_x
    village_chunk_z = reg_z * spacing + local_z

    return village_chunk_x == cx and village_chunk_z == cz


# ---------------------------------------------------------------------------
# Stronghold placement kernels (JIT)
# ---------------------------------------------------------------------------
@nb.njit(cache=True, inline='always')
def _village_chunk_in_region(world_seed_low, reg_x, reg_z):
    """Return the (chunk_x, chunk_z) of the (single) village in a village
    region. One MT init, four draws — identical math to
    `_check_village_at_chunk_jit` but returns the position instead of a
    bool comparison."""
    R_X = np.int64(R_X_CONST)
    R_Z = np.int64(R_Z_CONST)
    salt = np.int64(VILLAGE_SALT)
    spacing = np.int64(VILLAGE_SPACING)
    separation = np.int64(VILLAGE_SEPARATION)

    reg_seed = np.uint32(
        (np.int64(world_seed_low) + reg_x * R_X + reg_z * R_Z + salt)
        & np.int64(0xffffffff)
    )

    mt = mt_init(reg_seed)
    idx = N
    r0, idx = mt_extract(mt, idx)
    r1, idx = mt_extract(mt, idx)
    local_x = ((np.int64(r0) % separation) + (np.int64(r1) % separation)) // np.int64(2)
    r2, idx = mt_extract(mt, idx)
    r3, idx = mt_extract(mt, idx)
    local_z = ((np.int64(r2) % separation) + (np.int64(r3) % separation)) // np.int64(2)

    return reg_x * spacing + local_x, reg_z * spacing + local_z


@nb.njit(cache=True)
def _quasi_strongholds_jit(world_seed_low):
    """Run the initial quasi-random stronghold placement (up to 3 villages).

    Returns (out, count) where out[i] = (chunk_x, chunk_z) for the i-th
    found village/stronghold (0 <= i < count, count <= 3).

    Optimisation: instead of probing 256 chunks per attempt with a full MT
    init each, look up the (<=4) village regions touching the
    16x16 search window and check whether each region's single village
    falls inside the window. Reduces MT inits per attempt from up to 256
    down to at most 4.
    """
    out = np.zeros((3, 2), dtype=np.int64)
    out_count = 0

    spacing = np.int64(VILLAGE_SPACING)

    mt = mt_init(np.uint32(world_seed_low))
    idx = N

    inv_2_32 = np.float64(1.0) / np.float64(4294967296.0)

    angle_raw, idx = mt_extract(mt, idx)
    angle = np.float64(angle_raw) * inv_2_32 * (2.0 * math.pi)

    radius_raw, idx = mt_extract(mt, idx)
    r = np.float64((np.int64(radius_raw) % np.int64(16)) + np.int64(40))

    for _i in range(3):
        cx = np.int64(math.floor(r * math.cos(angle)))
        cz = np.int64(math.floor(r * math.sin(angle)))

        # Search window: dx,dz in [-8, 8) -> chunks in [cx-8, cx+8) x [cz-8, cz+8).
        wx_lo = cx - np.int64(8)
        wx_hi = cx + np.int64(8)   # exclusive
        wz_lo = cz - np.int64(8)
        wz_hi = cz + np.int64(8)   # exclusive

        # Up to 2 regions in each axis (16-chunk window < 34-chunk spacing).
        rx_lo = wx_lo // spacing
        rx_hi = (wx_hi - np.int64(1)) // spacing
        rz_lo = wz_lo // spacing
        rz_hi = (wz_hi - np.int64(1)) // spacing

        found = False
        best_sx = np.int64(0)
        best_sz = np.int64(0)

        for rx in range(rx_lo, rx_hi + np.int64(1)):
            for rz in range(rz_lo, rz_hi + np.int64(1)):
                sx, sz = _village_chunk_in_region(world_seed_low, rx, rz)
                # Inside the C# window [cx-8, cx+8) x [cz-8, cz+8) ?
                if wx_lo <= sx < wx_hi and wz_lo <= sz < wz_hi:
                    # C# iterates dx outer ascending, dz inner ascending and
                    # accepts the FIRST match in that scan order. Equivalent
                    # to picking min (sx, sz) lexicographically.
                    if (not found) or (sx < best_sx) or (sx == best_sx and sz < best_sz):
                        best_sx = sx
                        best_sz = sz
                        found = True

        if found:
            out[out_count, 0] = best_sx
            out[out_count, 1] = best_sz
            out_count += 1
            angle += 0.6 * math.pi
            r += 8.0
        else:
            angle += 0.25 * math.pi
            r += 4.0

    return out, out_count


@nb.njit(cache=True)
def _grid_strongholds_jit(world_seed_low, gx_min, gx_max, gz_min, gz_max):
    """Compute grid-placed stronghold chunk coordinates inside a grid range.

    Iterates only the grid cells inside [gx_min..gx_max] x [gz_min..gz_max].
    Returns (out, count) where out[i] = (chunk_x, chunk_z) for each spawned
    stronghold.
    """
    nx = gx_max - gx_min + 1
    nz = gz_max - gz_min + 1
    cap = nx * nz
    if cap < 1:
        cap = 1
    out = np.zeros((cap, 2), dtype=np.int64)
    out_count = 0

    GRID_SIZE = np.int64(STRONGHOLD_GRID_SIZE)
    XMUL_C    = np.int64(STRONGHOLD_XMUL)
    ZMUL_C    = np.int64(STRONGHOLD_ZMUL)
    SALT_C    = np.int64(STRONGHOLD_SALT)
    inv_2_32  = np.float64(1.0) / np.float64(4294967296.0)

    for grid_x in range(gx_min, gx_max + 1):
        for grid_z in range(gz_min, gz_max + 1):
            grid_x100 = GRID_SIZE * np.int64(grid_x) + np.int64(100)
            grid_z100 = GRID_SIZE * np.int64(grid_z) + np.int64(100)

            xMul = (XMUL_C * grid_x100) & np.int64(0xffffffff)
            zMul = (ZMUL_C * grid_z100) & np.int64(0xffffffff)
            cell_seed = np.uint32(
                (xMul + zMul + np.int64(world_seed_low) + SALT_C) & np.int64(0xffffffff)
            )

            mt = mt_init(cell_seed)
            idx = N

            spawn_prob, idx = mt_extract(mt, idx)
            if (np.float64(spawn_prob) * inv_2_32) < 0.25:
                min_x = GRID_SIZE * np.int64(grid_x) + GRID_SIZE - np.int64(150)
                max_x = GRID_SIZE * np.int64(grid_x) + np.int64(150)
                min_z = GRID_SIZE * np.int64(grid_z) + GRID_SIZE - np.int64(150)
                max_z = GRID_SIZE * np.int64(grid_z) + np.int64(150)

                x_raw, idx = mt_extract(mt, idx)
                z_raw, idx = mt_extract(mt, idx)

                x_chunk = min_x + np.int64(x_raw) % (max_x - min_x)
                z_chunk = min_z + np.int64(z_raw) % (max_z - min_z)

                out[out_count, 0] = x_chunk
                out[out_count, 1] = z_chunk
                out_count += 1

    return out, out_count


# ---------------------------------------------------------------------------
# Python wrappers around the JIT classifiers
# ---------------------------------------------------------------------------
def classify_bastion_or_fortress(world_seed, region_x, region_z):
    """Classify a structure at (region_x, region_z) as bastion or fortress.

    Returns (structure_type, bastion_subtype) where structure_type is
    "bastion" or "fortress" and bastion_subtype is 0..3 for bastions or
    None for fortresses.
    """
    is_bastion, bastion_type = _classify_bastion_jit(
        np.int64(world_seed & 0xffffffff),
        np.int64(region_x), np.int64(region_z),
    )
    if is_bastion:
        return "bastion", int(bastion_type)
    return "fortress", None


def classify_portal_variant(world_seed, chunk_x, chunk_z):
    """Classify a ruined portal at (chunk_x, chunk_z) and return its variant info dict."""
    underground, airpocket, rotation, mirror, giant, variant_num = _classify_portal_jit(
        np.int64(world_seed & 0xffffffff),
        np.int64(chunk_x), np.int64(chunk_z),
    )

    if giant:
        variant = f"giant_portal_{int(variant_num)}"
        variant_short = "giant"
    else:
        variant = f"portal_{int(variant_num)}"
        variant_short = "normal"

    variant_type = (
        f"{variant_short}:underground" if underground
        else f"{variant_short}:surface"
    )

    return {
        "underground": bool(underground),
        "airpocket":   bool(airpocket),
        "rotation":    int(rotation),
        "mirror":      bool(mirror),
        "giant":       bool(giant),
        "variant":     variant,
        "variant_short": variant_short,
        "variant_type":  variant_type,
    }


def check_village_at_chunk(world_seed, chunk_x, chunk_z):
    """Return True if a village generates exactly at the given chunk."""
    return bool(_check_village_at_chunk_jit(
        np.int64(world_seed & 0xffffffff),
        np.int64(chunk_x), np.int64(chunk_z),
    ))


def _is_stronghold_valid_biome(biome_gen, block_x, block_z):
    """Check a stronghold position lies in a village-compatible biome."""
    try:
        biome_id = biome_gen.biome_at_block(block_x, block_z)
        return is_village_biome(biome_id)
    except Exception:
        return False


def find_strongholds_in_box(world_seed, x1, z1, x2, z2,
                            biome_gen=None, skip_quasi=False):
    """Find all stronghold positions whose block (x, z) falls inside the
    bounding box (x1, z1) -> (x2, z2).

    Only grid cells whose stronghold-eligible region overlaps the bounding
    box are scanned, which makes the search proportional to the box area
    rather than to a fixed search radius.

    Args:
        world_seed: world seed (only the low 32 bits are used).
        x1, z1, x2, z2: block-coordinate bounding box (exclusive bounds —
            same convention used elsewhere in the search).
        biome_gen: optional BiomeGenerator for village-biome filtering.
        skip_quasi: if True, skip the initial quasi-random placement
            (the first ~3 strongholds derived from village finding) and
            only scan the deterministic 200x200-chunk grid.  This is
            significantly faster when only the grid placements matter.

    Returns:
        list of (block_x, block_z) tuples for matching strongholds.
    """
    s32 = int(world_seed) & 0xffffffff
    strongholds = []

    # Normalise box ordering defensively.
    if x1 > x2:
        x1, x2 = x2, x1
    if z1 > z2:
        z1, z2 = z2, z1

    # ----- Quasi-random placement (up to 3 strongholds via village finding)
    if not skip_quasi:
        chunks, count = _quasi_strongholds_jit(np.int64(s32))
        for i in range(int(count)):
            cx = int(chunks[i, 0])
            cz = int(chunks[i, 1])
            sh_x = cx * 16 + 8
            sh_z = cz * 16 + 8
            if x1 < sh_x < x2 and z1 < sh_z < z2:
                if biome_gen is None or _is_stronghold_valid_biome(biome_gen, sh_x, sh_z):
                    strongholds.append((sh_x, sh_z))

    # ----- Grid placement, restricted to the bounding box ------------------
    SCALE = STRONGHOLD_GRID_SIZE * 16  # = 3200 blocks per grid cell
    # ±1 cell of padding to cover strongholds near grid borders.
    gx_min = (x1 // SCALE) - 1
    gx_max = (x2 // SCALE) + 1
    gz_min = (z1 // SCALE) - 1
    gz_max = (z2 // SCALE) + 1

    chunks, count = _grid_strongholds_jit(
        np.int64(s32),
        np.int64(gx_min), np.int64(gx_max),
        np.int64(gz_min), np.int64(gz_max),
    )
    for i in range(int(count)):
        sh_x = int(chunks[i, 0]) * 16 + 8
        sh_z = int(chunks[i, 1]) * 16 + 8
        if x1 < sh_x < x2 and z1 < sh_z < z2:
            if biome_gen is None or _is_stronghold_valid_biome(biome_gen, sh_x, sh_z):
                strongholds.append((sh_x, sh_z))

    return strongholds


# ---------------------------------------------------------------------------
# Stronghold batch prefilter (parallel JIT)
# ---------------------------------------------------------------------------
@nb.njit(cache=True, parallel=True, boundscheck=False)
def _scan_batch_stronghold(seeds_start, seeds_end, x1, z1, x2, z2,
                           occurence, skip_quasi):
    """Parallel JIT prefilter: returns seeds with >=occurence strongholds
    inside the (x1, z1)-(x2, z2) bounding box.  Biome-blind — biome filtering
    is reapplied at Python level on the much smaller candidate list.

    Reuses the same `_quasi_strongholds_jit` and `_grid_strongholds_jit`
    helpers used by the per-seed Python wrappers, so MT19937 semantics are
    bit-for-bit identical.  The batch parallelism comes from numba's
    `prange` distributing seeds across threads.
    """
    n = seeds_end - seeds_start
    is_hit = np.zeros(n, dtype=np.bool_)

    GRID  = np.int64(STRONGHOLD_GRID_SIZE)
    SCALE = GRID * np.int64(16)
    MASK  = np.int64(0xffffffff)
    occ   = np.int32(occurence)

    bx1 = np.int64(x1); bx2 = np.int64(x2)
    bz1 = np.int64(z1); bz2 = np.int64(z2)

    gx_min = bx1 // SCALE - np.int64(1)
    gx_max = bx2 // SCALE + np.int64(1)
    gz_min = bz1 // SCALE - np.int64(1)
    gz_max = bz2 // SCALE + np.int64(1)

    for ii in nb.prange(n):
        s32 = (np.int64(seeds_start) + np.int64(ii)) & MASK
        found = np.int32(0)

        if not skip_quasi:
            q_chunks, q_count = _quasi_strongholds_jit(s32)
            for qi in range(q_count):
                sh_x = q_chunks[qi, 0] * np.int64(16) + np.int64(8)
                sh_z = q_chunks[qi, 1] * np.int64(16) + np.int64(8)
                if bx1 < sh_x < bx2 and bz1 < sh_z < bz2:
                    found += np.int32(1)
                    if found >= occ:
                        break

        if found < occ:
            g_chunks, g_count = _grid_strongholds_jit(
                s32, gx_min, gx_max, gz_min, gz_max
            )
            for gi in range(g_count):
                sh_x = g_chunks[gi, 0] * np.int64(16) + np.int64(8)
                sh_z = g_chunks[gi, 1] * np.int64(16) + np.int64(8)
                if bx1 < sh_x < bx2 and bz1 < sh_z < bz2:
                    found += np.int32(1)
                    if found >= occ:
                        break

        if found >= occ:
            is_hit[ii] = True

    count = np.int64(0)
    for ii in range(n):
        if is_hit[ii]:
            count += np.int64(1)
    result = np.empty(count, dtype=np.int64)
    ci = np.int64(0)
    for ii in range(n):
        if is_hit[ii]:
            result[ci] = np.int64(seeds_start) + np.int64(ii)
            ci += np.int64(1)
    return result


def scan_batch_stronghold(seeds_start, seeds_end, x1, z1, x2, z2,
                          occurence, skip_quasi):
    """Python wrapper around the parallel stronghold batch prefilter."""
    return _scan_batch_stronghold(
        int(seeds_start), int(seeds_end),
        int(x1), int(z1), int(x2), int(z2),
        int(occurence), bool(skip_quasi),
    )
