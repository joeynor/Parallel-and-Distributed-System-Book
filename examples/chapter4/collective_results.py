"""Compare root-only and all-rank collective results on uneven blocks."""

import sys

import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    try:
        if len(sys.argv) > 2:
            raise ValueError
        n = int(sys.argv[1]) if len(sys.argv) == 2 else 7
        if n < 1:
            raise ValueError
    except ValueError:
        n = None
else:
    n = None

n = comm.bcast(n, root=0)
if n is None:
    if rank == 0:
        print("Usage: collective_results.py [N], where N >= 1")
    raise SystemExit(1)

q, remainder = divmod(n, size)
counts = np.array([q + (r < remainder) for r in range(size)],
                  dtype=np.int32)
displacements = np.zeros(size, dtype=np.int32)
displacements[1:] = np.cumsum(counts[:-1])

start = int(displacements[rank])
count = int(counts[rank])
local = np.arange(start + 1, start + count + 1, dtype=np.float64)

# Object gather returns a list on the root; other ranks receive None.
local_counts = comm.gather(count, root=0)

# Typed Gatherv writes the numerical blocks into preallocated root storage.
reconstructed = np.empty(n, dtype=np.float64) if rank == 0 else None
comm.Gatherv(local, [reconstructed, counts, displacements, MPI.DOUBLE]
             if rank == 0 else None, root=0)

# A reduction has a result only at the root.
total = comm.reduce(float(local.sum()), op=MPI.SUM, root=0)

# An all-reduction makes the same maximum available on every rank.
local_max = float(local.max()) if count else float("-inf")
global_max = comm.allreduce(local_max, op=MPI.MAX)

if rank == 0:
    expected = np.arange(1, n + 1, dtype=np.float64)
    passed = (local_counts == counts.tolist()
              and np.array_equal(reconstructed, expected)
              and total == float(expected.sum())
              and global_max == float(n))
    print(f"N={n} P={size} counts={local_counts} "
          f"sum={total:.0f} max={global_max:.0f} "
          f"{'PASS' if passed else 'FAIL'}")
