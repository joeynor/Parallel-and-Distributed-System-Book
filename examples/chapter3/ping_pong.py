from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if comm.Get_size() != 2:
    if rank == 0:
        print("Run with exactly 2 ranks", flush=True)
    raise SystemExit(1)

warmup, repetitions = 10000, 1000000
# One byte; use buffer communication to avoid object pickling.
token = np.zeros(1, dtype=np.uint8)
start = 0.0
for i in range(-warmup, repetitions):
    if rank == 0:
        if i == 0:
            start = MPI.Wtime()
        comm.Send([token, MPI.BYTE], dest=1, tag=20)
        comm.Recv([token, MPI.BYTE], source=1, tag=21)
    else:
        comm.Recv([token, MPI.BYTE], source=0, tag=20)
        comm.Send([token, MPI.BYTE], dest=0, tag=21)

if rank == 0:
    elapsed = MPI.Wtime() - start
    mean_us = 1e6 * elapsed / repetitions
    print(f"Mean round trip: {mean_us:.3f} us", flush=True)
