from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

left = rank - 1 if rank > 0 else MPI.PROC_NULL
right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

# Indices 1..4 are owned; indices 0 and 5 are ghost cells.
field = np.full(6, -1.0, dtype=np.float64)
field[1:5] = rank

requests = [
    comm.Irecv(field[0:1], source=left, tag=20),
    comm.Irecv(field[5:6], source=right, tag=21),
    comm.Isend(field[1:2], dest=left, tag=21),
    comm.Isend(field[4:5], dest=right, tag=20),
]

# This work does not need ghost cells.
interior_total = float(np.sum(field[2:4]))

MPI.Request.Waitall(requests)

for r in range(size):
    comm.Barrier()
    if rank == r:
        print(f"rank={rank} left={left} right={right} "
              f"field={field.tolist()} interior={interior_total}",
              flush=True)
