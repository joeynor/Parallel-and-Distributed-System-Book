from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start = 0.0 

if size == 1:
    print("One rank: token remains 0", flush=True)
elif rank == 0:
    start = MPI.Wtime()
    comm.send(0, dest=1, tag=30)
    token = comm.recv(source=size - 1, tag=30)
    expected = size * (size - 1) // 2
    print(f"Returned token: {token}; expected: {expected}",
          flush=True)
else:
    token = comm.recv(source=rank - 1, tag=30)
    token += rank
    comm.send(token, dest=(rank + 1) % size, tag=30)

if rank == 0:
    elapsed = MPI.Wtime() - start
    mean_us = 1e6 * elapsed 
    print(f"Total Time: {mean_us:.3f} us", flush=True)
