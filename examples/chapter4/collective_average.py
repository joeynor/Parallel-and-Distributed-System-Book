import sys
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    try:
        if len(sys.argv) > 2:
            raise ValueError
        n = int(sys.argv[1]) if len(sys.argv) == 2 else 17
        if n < 1:
            raise ValueError
    except ValueError:
        n = None
else:
    n = None

n = comm.bcast(n, root=0)
if n is None:
    if rank == 0:
        print("Usage: collective_average.py [N], where N >= 1")
    raise SystemExit(1)

if rank == 0:
# now working in creating chunks
    data = list(range(1, n + 1))
    q, remainder = divmod(n, size)
    counts = [q + (r < remainder) for r in range(size)]
    starts = [r * q + min(r, remainder) for r in range(size)]
    chunks = [data[s:s + c] for s, c in zip(starts, counts)]
else:
    data = None
    chunks = None

local = comm.scatter(chunks, root=0)
local_sum = sum(local)
local_count = len(local)

global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
global_count = comm.reduce(local_count, op=MPI.SUM, root=0)

if rank == 0:
    average = global_sum / global_count
    expected = sum(data) / len(data)
    passed = global_sum == sum(data) and average == expected
    print(f"N={n} P={size} sum={global_sum} "
          f"average={average:.2f} {'PASS' if passed else 'FAIL'}")
