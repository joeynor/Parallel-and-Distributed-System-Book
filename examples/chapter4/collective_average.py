import sys
from mpi4py import MPI


# Initialize MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()      # Process rank (0 to size-1)
size = comm.Get_size()      # Total number of processes

# Rank 0 handles command-line argument parsing
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
    # Non-root processes start with n = None
    n = None

# Broadcast the value of N from rank 0 to all processes
n = comm.bcast(n, root=0)

# Check if N was provided correctly
if n is None:
    if rank == 0:
        print("Usage: collective_average.py [N], where N >= 1")
    raise SystemExit(1)

# Rank 0 prepares the data and partitions it into chunks for distribution
if rank == 0:
    # Create list of integers from 1 to N
    data = list(range(1, n + 1))
    
    # Calculate how many elements each process gets (chunk size)
    # q = quotient (base number of elements per process)
    # remainder = leftover elements to distribute among first 'remainder' processes
    q, remainder = divmod(n, size)
    
    # Determine count of elements each process receives
    counts = [q + (r < remainder) for r in range(size)]
    
    # Calculate starting index for each process's data slice
    starts = [r * q + min(r, remainder) for r in range(size)]
    
    # Create list of slices (chunks), one per process
    chunks = [data[s:s + c] for s, c in zip(starts, counts)]
else:
    # Non-root processes receive None and will get data from scatter
    data = None
    chunks = None

# Scatter the chunked data to all processes (rank 0 sends, others receive)
local = comm.scatter(chunks, root=0)
local_sum = sum(local)       # Sum of elements in this process's data
local_count = len(local)     # Number of elements in this process's data

# Reduce operation: sum up local values across all processes
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
global_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# Rank 0 computes and reports the result
if rank == 0:
    average = global_sum / global_count
    
    # Calculate expected values for verification (computed only on rank 0)
    expected = sum(data) / len(data)
    
    # Check if both sum and average match expected values
    passed = global_sum == sum(data) and average == expected
    
    print(f"N={n} P={size} sum={global_sum} "
          f"average={average:.2f} {'PASS' if passed else 'FAIL'}")
