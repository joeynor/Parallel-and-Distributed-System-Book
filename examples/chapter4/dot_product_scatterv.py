import sys
from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n = None
if rank == 0:
    try:
        if len(sys.argv) > 2:
            raise ValueError
        n = int(sys.argv[1]) if len(sys.argv) == 2 else 17
        if n < 0:
            raise ValueError
    except ValueError:
        n = None
n = comm.bcast(n, root=0)
if n is None:
    if rank == 0:
        print("Usage: dot_product_scatterv.py [N], where N >= 0")
    raise SystemExit(1)

q, remainder = divmod(n, size)
counts = np.array([q + (r < remainder) for r in range(size)],
                  dtype=np.int32)
displacements = np.zeros(size, dtype=np.int32)
displacements[1:] = np.cumsum(counts[:-1])

local_n = int(counts[rank])
local_x = np.empty(local_n, dtype=np.float64)
local_y = np.empty(local_n, dtype=np.float64)

if rank == 0:
    x = np.arange(1, n + 1, dtype=np.float64)
    y = np.full(n, 2.0, dtype=np.float64)
else:
    x = y = None

comm.Scatterv([x, counts, displacements, MPI.DOUBLE],
              local_x, root=0)
comm.Scatterv([y, counts, displacements, MPI.DOUBLE],
              local_y, root=0)

local_dot = float(np.dot(local_x, local_y))
global_dot = comm.reduce(local_dot, op=MPI.SUM, root=0)

if rank == 0:
    expected = float(np.dot(x, y))
    print(f"N={n} P={size} dot={global_dot:.1f} "
          f"expected={expected:.1f} "
          f"{'PASS' if np.isclose(global_dot, expected) else 'FAIL'}")
