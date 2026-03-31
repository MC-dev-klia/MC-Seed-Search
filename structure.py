"""
structure.py — Bedrock Edition structural position calculations.

Implements the Mersenne Twister RNG used by Minecraft Bedrock to determine
where structures are placed within each region, and the getpos() function
that combines the world seed, region coordinates, and RNG constants into a
block-level structure position.
"""

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


@nb.njit(cache=True)
def mt_init(seed32):
    """Initialise a 624-element MT state from a 32-bit seed."""
    mt = np.empty(N, dtype=np.uint32)
    mt[0] = seed32
    for i in range(1, N):
        mt[i] = (0x6c078965 * (mt[i-1] ^ (mt[i-1] >> 30)) + i) & MASK32
    return mt


@nb.njit(cache=True)
def mt_twist(mt):
    """Apply one full twist to the MT state array."""
    for i in range(N):
        y = (mt[i] & UPPER_MASK) | (mt[(i+1) % N] & LOWER_MASK)
        mt[i] = mt[(i + M) % N] ^ (y >> 1)
        if y & 1:
            mt[i] ^= MATRIX_A


@nb.njit(cache=True)
def mt_extract(mt, idx):
    """Extract and temper one value from the MT state, twisting when needed."""
    if idx >= N:
        mt_twist(mt)
        idx = 0
    y = mt[idx]
    y ^= (y >> 11)
    y ^= (y << 7)  & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= (y >> 18)
    return y & MASK32, idx + 1


@nb.njit(cache=True, parallel=True)
def _scan_batch(seeds_start, seeds_end, spacing, separation, salt,
                linear_sep, radius, occurence):
    """
    Optimised batch scanner.

    Speedups over the naive version:
    1. Parallel execution via nb.prange (one thread per CPU core).
    2. Scalar rolling MT — instead of allocating a partial array of M+4
       uint32s and indexing into it, we unroll the init recurrence with a
       single rolling scalar variable and capture only the 4-9 values
       actually needed for the twist+temper.  This eliminates the 1604-byte
       per-seed stack array and lets the compiler keep all intermediate
       values in registers.
    3. Adaptive loop length — when linear_sep is False we only need indices
       0,1,2 and M,M+1, so the rolling loop stops at M+1 instead of M+3.
    4. Dual early-exit — break as soon as (a) found >= occ (already won) or
       (b) found + remaining_regions < occ (can't possibly win).
    5. Two-phase gather — parallel mark pass then sequential collect,
       avoiding any shared-counter race conditions.
    """
    spawn_range  = spacing - separation
    n            = seeds_end - seeds_start
    R_X          = np.int64(341873128712)
    R_Z          = np.int64(132897987541)
    MULT         = np.int64(0x6c078965)
    MATRIX_A_    = np.uint32(0x9908b0df)
    UPPER_MASK_  = np.uint32(0x80000000)
    LOWER_MASK_  = np.uint32(0x7fffffff)
    T1           = np.uint32(0x9D2C5680)
    T2           = np.uint32(0xEFC60000)
    sr           = np.int64(spawn_range)
    sp           = np.int64(spacing)
    rad          = np.int64(radius)
    occ          = np.int32(occurence)

    # Phase 1 — parallel mark pass
    is_hit = np.zeros(n, dtype=np.bool_)

    for ii in nb.prange(n):
        world_seed = np.int64(seeds_start) + np.int64(ii)
        found      = np.int32(0)

        for region in range(4):
            rx = np.int64(-(region & 1))
            rz = np.int64(-((region >> 1) & 1))

            # 32-bit seed for this region
            s32 = np.uint32(world_seed + rx * R_X + rz * R_Z + np.int64(salt))

            # ------------------------------------------------------------------
            # Scalar rolling MT initialisation.
            #
            # Instead of filling an array mt[0..M+3] and then reading back
            # just 4-9 slots, we keep a single rolling variable p and capture
            # only the indices actually needed for the twist+temper:
            #   a0 = mt[0], a1 = mt[1], a2 = mt[2]        (always)
            #   a3 = mt[3], a4 = mt[4]                     (linear_sep only)
            #   b0 = mt[M], b1 = mt[M+1]                   (always)
            #   b2 = mt[M+2], b3 = mt[M+3]                 (linear_sep only)
            # ------------------------------------------------------------------
            p  = s32
            a0 = p
            p  = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(1)) & np.int64(0xFFFFFFFF))
            a1 = p
            p  = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(2)) & np.int64(0xFFFFFFFF))
            a2 = p

            if linear_sep:
                p  = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(3)) & np.int64(0xFFFFFFFF))
                a3 = p
                p  = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(4)) & np.int64(0xFFFFFFFF))
                a4 = p

                # Roll indices 5 .. M-1 (discarding intermediate values)
                for k in range(np.int64(5), np.int64(M)):
                    p = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + k) & np.int64(0xFFFFFFFF))

                # Capture b0..b3
                p  = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(M))     & np.int64(0xFFFFFFFF))
                b0 = p
                p  = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(M + 1)) & np.int64(0xFFFFFFFF))
                b1 = p
                p  = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(M + 2)) & np.int64(0xFFFFFFFF))
                b2 = p
                p  = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(M + 3)) & np.int64(0xFFFFFFFF))
                b3 = p

                # --- Twist+temper index 0 ---
                y  = (a0 & UPPER_MASK_) | (a1 & LOWER_MASK_)
                v0 = b0 ^ (y >> np.uint32(1))
                if y & np.uint32(1):
                    v0 ^= MATRIX_A_
                v0 ^= v0 >> np.uint32(11)
                v0 ^= (v0 << np.uint32(7))  & T1
                v0 ^= (v0 << np.uint32(15)) & T2
                v0 ^= v0 >> np.uint32(18)

                # --- Twist+temper index 1 ---
                y  = (a1 & UPPER_MASK_) | (a2 & LOWER_MASK_)
                v1 = b1 ^ (y >> np.uint32(1))
                if y & np.uint32(1):
                    v1 ^= MATRIX_A_
                v1 ^= v1 >> np.uint32(11)
                v1 ^= (v1 << np.uint32(7))  & T1
                v1 ^= (v1 << np.uint32(15)) & T2
                v1 ^= v1 >> np.uint32(18)

                # --- Twist+temper index 2 ---
                y  = (a2 & UPPER_MASK_) | (a3 & LOWER_MASK_)
                v2 = b2 ^ (y >> np.uint32(1))
                if y & np.uint32(1):
                    v2 ^= MATRIX_A_
                v2 ^= v2 >> np.uint32(11)
                v2 ^= (v2 << np.uint32(7))  & T1
                v2 ^= (v2 << np.uint32(15)) & T2
                v2 ^= v2 >> np.uint32(18)

                # --- Twist+temper index 3 ---
                y  = (a3 & UPPER_MASK_) | (a4 & LOWER_MASK_)
                v3 = b3 ^ (y >> np.uint32(1))
                if y & np.uint32(1):
                    v3 ^= MATRIX_A_
                v3 ^= v3 >> np.uint32(11)
                v3 ^= (v3 << np.uint32(7))  & T1
                v3 ^= (v3 << np.uint32(15)) & T2
                v3 ^= v3 >> np.uint32(18)

                off_x = (np.int64(v0) % sr + np.int64(v1) % sr) // np.int64(2)
                off_z = (np.int64(v2) % sr + np.int64(v3) % sr) // np.int64(2)

            else:
                # Roll indices 3 .. M-1 (two fewer iterations than linear_sep)
                for k in range(np.int64(3), np.int64(M)):
                    p = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + k) & np.int64(0xFFFFFFFF))

                # Capture b0..b1 only
                p  = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(M))     & np.int64(0xFFFFFFFF))
                b0 = p
                p  = np.uint32((MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(M + 1)) & np.int64(0xFFFFFFFF))
                b1 = p

                # --- Twist+temper index 0 ---
                y  = (a0 & UPPER_MASK_) | (a1 & LOWER_MASK_)
                v0 = b0 ^ (y >> np.uint32(1))
                if y & np.uint32(1):
                    v0 ^= MATRIX_A_
                v0 ^= v0 >> np.uint32(11)
                v0 ^= (v0 << np.uint32(7))  & T1
                v0 ^= (v0 << np.uint32(15)) & T2
                v0 ^= v0 >> np.uint32(18)

                # --- Twist+temper index 1 ---
                y  = (a1 & UPPER_MASK_) | (a2 & LOWER_MASK_)
                v1 = b1 ^ (y >> np.uint32(1))
                if y & np.uint32(1):
                    v1 ^= MATRIX_A_
                v1 ^= v1 >> np.uint32(11)
                v1 ^= (v1 << np.uint32(7))  & T1
                v1 ^= (v1 << np.uint32(15)) & T2
                v1 ^= v1 >> np.uint32(18)

                off_x = np.int64(v0) % sr
                off_z = np.int64(v1) % sr

            bx = (rx * sp + off_x) * np.int64(16) + np.int64(8)
            bz = (rz * sp + off_z) * np.int64(16) + np.int64(8)

            if -rad < bx < rad and -rad < bz < rad:
                found += np.int32(1)

            # Early exit — already satisfied, or can't reach occ any more
            if found >= occ or found + np.int32(3 - region) < occ:
                break

        if found >= occ:
            is_hit[ii] = True

    # Phase 2 — sequential gather (no contention)
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
    """Python wrapper — triggers numba JIT on first call, cached afterwards."""
    return _scan_batch(int(seeds_start), int(seeds_end),
                       int(spacing), int(separation), int(salt),
                       bool(linear_sep), int(radius), int(occurence))


def getpos(world_seed, rx, rz, spacing, separation, salt, linear_separation):
    """
    Return the block-level (x, z) position of a structure candidate in
    region (rx, rz) for the given world seed and structure RNG constants.

    Parameters
    ----------
    world_seed       : 48-bit world seed
    rx, rz           : region coordinates (integers)
    spacing          : region size in chunks
    separation       : minimum separation in chunks
    salt             : structure-specific RNG salt
    linear_separation: if True uses the averaged two-draw algorithm
    """
    spawn_range = spacing - separation
    mixed = (world_seed + rx * 341873128712 + rz * 132897987541 + salt) & ((1 << 64) - 1)
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
    return (chunk_x * 16 + 8, chunk_z * 16 + 8)
