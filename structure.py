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
# Inline temper macro — applied after each twist computation
# ---------------------------------------------------------------------------
# Used identically in both scan kernels; kept as a comment for clarity.
#
#   v ^= v >> 11
#   v ^= (v << 7)  & 0x9D2C5680
#   v ^= (v << 15) & 0xEFC60000
#   v ^= v >> 18
# ---------------------------------------------------------------------------


@nb.njit(cache=True, parallel=True)
def _scan_batch_standard(seeds_start, seeds_end, spacing, separation, salt,
                         radius, occurence):
    """
    Specialised scanner for linear_sep=False (standard two-draw placement).

    Optimisations vs the old unified kernel:
      - No runtime branch on linear_sep — LLVM sees a clean loop body and
        can optimise register allocation / loop unrolling more aggressively.
      - Partial-MT buffer shrunk to M+2 (399 elements) since only v0 and v1
        are needed; saves 2 MT-init iterations per region (8 per seed).
    """
    spawn_range = spacing - separation
    n           = seeds_end - seeds_start
    R_X         = np.int64(341873128712)
    R_Z         = np.int64(132897987541)
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

    # Phase 1 — parallel mark pass
    is_hit = np.zeros(n, dtype=np.bool_)

    for ii in nb.prange(n):
        world_seed = np.int64(seeds_start) + np.int64(ii)
        found      = np.int32(0)

        # MT buffer: only indices 0..M+1 (=398) are needed for v0 and v1.
        # v0 twist reads mt[0], mt[1], mt[M];  v1 reads mt[1], mt[2], mt[M+1].
        mt = np.empty(M + 2, dtype=np.uint32)

        for region in range(4):
            rx = np.int64(-(region & 1))
            rz = np.int64(-((region >> 1) & 1))

            s32   = np.uint32(world_seed + rx * R_X + rz * R_Z + np.int64(salt))
            mt[0] = s32
            for k in range(1, M + 2):
                p     = mt[k - 1]
                mt[k] = np.uint32(
                    (MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(k))
                    & np.int64(0xFFFFFFFF)
                )

            # --- Inline twist+temper for index 0 ---
            y  = (mt[0] & UPPER_MASK_) | (mt[1] & LOWER_MASK_)
            v0 = mt[M] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v0 ^= MATRIX_A_
            v0 ^= v0 >> np.uint32(11)
            v0 ^= (v0 << np.uint32(7))  & T1
            v0 ^= (v0 << np.uint32(15)) & T2
            v0 ^= v0 >> np.uint32(18)

            # --- Inline twist+temper for index 1 ---
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

            bx = (rx * sp + off_x) * np.int64(16) + np.int64(8)
            bz = (rz * sp + off_z) * np.int64(16) + np.int64(8)

            if -rad < bx < rad and -rad < bz < rad:
                found += np.int32(1)

            # Early exit: remaining regions cannot reach occurence threshold
            if found + np.int32(3 - region) < occ:
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


@nb.njit(cache=True, parallel=True)
def _scan_batch_linear(seeds_start, seeds_end, spacing, separation, salt,
                       radius, occurence):
    """
    Specialised scanner for linear_sep=True (averaged four-draw placement).

    Optimisations vs the old unified kernel:
      - No runtime branch on linear_sep — LLVM sees a clean loop body and
        can optimise register allocation / loop unrolling more aggressively.
      - MT buffer remains M+4 (401 elements) since v2/v3 need mt[M+2]/mt[M+3].
    """
    spawn_range = spacing - separation
    n           = seeds_end - seeds_start
    R_X         = np.int64(341873128712)
    R_Z         = np.int64(132897987541)
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

    # Phase 1 — parallel mark pass
    is_hit = np.zeros(n, dtype=np.bool_)

    for ii in nb.prange(n):
        world_seed = np.int64(seeds_start) + np.int64(ii)
        found      = np.int32(0)

        mt = np.empty(M + 4, dtype=np.uint32)

        for region in range(4):
            rx = np.int64(-(region & 1))
            rz = np.int64(-((region >> 1) & 1))

            s32   = np.uint32(world_seed + rx * R_X + rz * R_Z + np.int64(salt))
            mt[0] = s32
            for k in range(1, M + 4):
                p     = mt[k - 1]
                mt[k] = np.uint32(
                    (MULT * np.int64(p ^ (p >> np.uint32(30))) + np.int64(k))
                    & np.int64(0xFFFFFFFF)
                )

            # --- Inline twist+temper for index 0 ---
            y  = (mt[0] & UPPER_MASK_) | (mt[1] & LOWER_MASK_)
            v0 = mt[M] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v0 ^= MATRIX_A_
            v0 ^= v0 >> np.uint32(11)
            v0 ^= (v0 << np.uint32(7))  & T1
            v0 ^= (v0 << np.uint32(15)) & T2
            v0 ^= v0 >> np.uint32(18)

            # --- Inline twist+temper for index 1 ---
            y  = (mt[1] & UPPER_MASK_) | (mt[2] & LOWER_MASK_)
            v1 = mt[M + 1] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v1 ^= MATRIX_A_
            v1 ^= v1 >> np.uint32(11)
            v1 ^= (v1 << np.uint32(7))  & T1
            v1 ^= (v1 << np.uint32(15)) & T2
            v1 ^= v1 >> np.uint32(18)

            # --- Inline twist+temper for index 2 ---
            y  = (mt[2] & UPPER_MASK_) | (mt[3] & LOWER_MASK_)
            v2 = mt[M + 2] ^ (y >> np.uint32(1))
            if y & np.uint32(1):
                v2 ^= MATRIX_A_
            v2 ^= v2 >> np.uint32(11)
            v2 ^= (v2 << np.uint32(7))  & T1
            v2 ^= (v2 << np.uint32(15)) & T2
            v2 ^= v2 >> np.uint32(18)

            # --- Inline twist+temper for index 3 ---
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

            bx = (rx * sp + off_x) * np.int64(16) + np.int64(8)
            bz = (rz * sp + off_z) * np.int64(16) + np.int64(8)

            if -rad < bx < rad and -rad < bz < rad:
                found += np.int32(1)

            # Early exit: remaining regions cannot reach occurence threshold
            if found + np.int32(3 - region) < occ:
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
    """
    Python wrapper — dispatches to the correct specialised JIT kernel.

    Using two separate compiled functions (instead of a single function with
    a runtime boolean flag) lets LLVM optimise each kernel in isolation:
    no dead-code paths, no unused register pressure, and different partial-MT
    buffer sizes for the two cases.
    """
    args = (int(seeds_start), int(seeds_end),
            int(spacing), int(separation), int(salt),
            int(radius), int(occurence))
    if linear_sep:
        return _scan_batch_linear(*args)
    return _scan_batch_standard(*args)


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
    mixed  = (world_seed + rx * 341873128712 + rz * 132897987541 + salt) & ((1 << 64) - 1)
    seed32 = mixed & 0xffffffff

    mt  = mt_init(seed32)
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
