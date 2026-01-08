from typing import List, Tuple
import time
from math import floor
import numba as nb
import numpy as np


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
    if idx >= N:  # needs twist
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
This is Minecraft Bedrock Edition brute force seed searching app scanning for possibile structural 48-bit seeds.
These seeds do not nessasary match because of biome restrictions, but altering the first 16-bits using a biome seedfinder may resolve the problem.
Additionally, this app does not seperation structure variants, such as seperating bastions from fortresses.
This app also only search regions [-1,-1] to [0,0], the first 4 quadrant around the origin.

Input:
SeedStart: The first seed the app simulates, and it adds one to the seed for the next seed.
SeedEnd: Once we hit this seed, the app no longer scans, and stops. This is exclusive.

The next four inputs are RNG constants. You will need to find them, then input it.
Spacing: The number of chunks each region is in size.
Seperation: The number of chunks that seperate each region.
Salt: The number that the RNG uses to differ structure randomness from each other.
Linear Seperation: Describes what algorithm it uses for seperation.

List of RNG constants: (Format: {Spacing, Seperation, Salt, Linear Seperation})
Bastion and Fortress: {30, 4, 30084232, 0}
Village: {34, 8, 10387312, 1}
Pillage Outpost: {80, 24, 165745296, 1}
Woodland Mansion: {80, 20, 10387319, 1}
Monument: {32, 5, 10387313, 1} 
Shipwreck: {24, 4, 165745295, 1}
Ruined Portal: {40, 15, 40552231, 0}
Many other Overworld Structures: {32, 8, 14357617, 0}

Search radius: In this radius, the app will accept the match and add 1 to the counter.
Min occurence: How many of the structures do you want in the radius before it gets outputed. (must be <= 4)

"""
)
def seedsearch():
    seedstart = int(input("SeedStart: "))
    seedend = int(input("SeedEnd: "))
    spacing = int(input("Spacing: "))
    separation = int(input("Separation: "))
    salt = int(input("Salt: "))
    linear_seperation = bool(int(input("Linear seperation: (0 or 1) ")))
    radius = int(input("Search radius: "))
    occurence = int(input("Min occurence: "))
    output_file = input("Output file name (default: seed_results.txt): ") or "seed_results.txt"
    times = time.time()

    seeds = range(seedstart, seedend)

    with open(output_file, 'w') as f:
        for seed in seeds:
            found = 0
            i = (getpos(seed, 0, 0, spacing, separation, salt, linear_seperation))
            if -radius < i[0] < radius and -radius < i[1] < radius:
                found+=1
            if found<1 and occurence>=4:
                continue

            j = getpos(seed, -1, 0, spacing, separation, salt, linear_seperation)
            if -radius < j[0] < radius and -radius < j[1] < radius:
                found+=1


            k = getpos(seed, 0, -1, spacing, separation, salt, linear_seperation)
            if -radius < k[0] < radius and -radius < k[1] < radius:
                found+=1


            l = getpos(seed, -1, -1, spacing, separation, salt, linear_seperation)
            if -radius < l[0] < radius and -radius < l[1] < radius:
                found+=1
            if(found >= occurence):
                f.write(f"Seed {seed}: {i,j,k,l}\n")
            if seed % 1000000 == 0 and seed != seedstart:
                f.write(f"Scanned up to {seed}. Time taken: {time.time()-times}\n")
        f.write("Finished Scanning\n")
        f.write(f"Time: {time.time()-times}\n")
    
    print(f"Results saved to {output_file}")

seedsearch()    
