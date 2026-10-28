# Chapter 4 mpi4py examples

Run from this directory with the Chapter 3 MPI environment active:

```sh
mpiexec -n 4 python -m mpi4py collective_average.py 17
mpiexec -n 4 python -m mpi4py collective_results.py 7
mpiexec -n 4 python -m mpi4py dot_product_scatterv.py 17
mpiexec -n 4 python -m mpi4py nonblocking_halo.py
```

The average, collective-results, and dot-product examples accept process
counts larger than the number of elements. `collective_results.py` contrasts
object `gather`, typed `Gatherv`, root-only `reduce`, and all-rank `allreduce`.
The halo example uses four owned values per rank and two ghost cells.
Boundary ranks communicate with `MPI.PROC_NULL` where a neighbor does not
exist.
