Last name of Student 1: Jain
First name of Student 1: Rohil
Email of Student 1: rohiljain@ucsb.edu
Last name of Student 2: Gehlot
First name of Student 2: Sharanya
Email of Student 2: sgehlot@ucsb.edu



Report for Question 2.a
Performance:

Test Case: Regular matrix where A[i][i]=0 and A[i][j]=-1/n for i≠j
           Vector d with all elements = (2n-1)/n
           Vector x initially all zeros

Results with 2 cores (2 MPI processes):
  Parallel Time: 10.238485 seconds
  GFLOPS: 3.3559
  
Results with 4 cores (4 MPI processes):
  Parallel Time: 6.054778 seconds
  GFLOPS: 5.6748

Speedup and Efficiency:
  Speedup (4 cores vs 2 cores): 10.238485 / 6.054778 = 1.69x
  Efficiency (4 cores): (1.69 / 2) × 100% = 84.5%

Analysis:
The implementation achieves good parallel speedup of 1.69x when doubling from 
2 to 4 cores, with an efficiency of 84.5%. This indicates effective 
parallelization with reasonable communication overhead. The GFLOPS nearly 
doubled from 3.36 to 5.67, demonstrating that the block row mapping strategy 
scales well for this problem size.

The 84.5% efficiency means that 15.5% of computational resources are lost to 
overhead, primarily from the MPI_Allgather operation performed in each of the 
1024 iterations. This communication pattern requires all processes to exchange 
their local portions of vector x, which becomes more costly as the number of 
processes increases.


Report for Question 2.b
Performance (n=4096, t=1024):

Test Case: Upper triangular matrix where A[i][i]=0, A[i][j]=-1/n for i<j, 
           and A[i][j]=0 for i>j
           Vector d with all elements = (2n-1)/n
           Vector x initially all zeros

Results with 2 cores (2 MPI processes):
  Parallel Time: 7.899278 seconds
  GFLOPS: 2.1754

Results with 4 cores (4 MPI processes):
  Parallel Time: 4.931044 seconds
  GFLOPS: 3.4849

Speedup and Efficiency:
  Speedup (4 cores vs 2 cores): 7.899278 / 4.931044 = 1.60x
  Efficiency (4 cores): (1.60 / 2) × 100% = 80.1%

Comparison with Problem 2.a:
  Problem 2.a efficiency (4 cores): 84.5%
  Problem 2.b efficiency (4 cores): 80.1%
  Efficiency difference: 4.4% lower for upper triangular

Explanation of Lower Efficiency in Problem 2.b:
The upper triangular matrix implementation has LOWER efficiency (80.1%) 
compared to the regular matrix implementation (84.5%) due to load imbalance:

1. UNEQUAL WORK DISTRIBUTION: With upper triangular matrices, processes that 
   own earlier rows (lower rank processes) have MORE work than processes 
   owning later rows. For example:
   - Process 0 owns rows 0 to (n/p-1): Each row has ~n elements to compute
   - Process p-1 owns rows n-n/p to n-1: Each row has fewer elements to compute
   
2. IDLE TIME: Higher-ranked processes finish their computation earlier and must 
   wait at synchronization points (MPI_Allgather), wasting computational 
   resources.

3. CONSTANT COMMUNICATION: Despite doing less computation, all processes still 
   participate equally in MPI_Allgather, so the communication-to-computation 
   ratio is WORSE for upper triangular matrices.

Strategy to Improve Efficiency:
Use CYCLIC or BLOCK-CYCLIC mapping instead of block row mapping:

BLOCK-CYCLIC MAPPING: Assign rows to processes in a round-robin fashion with 
block size b:
  - Rows 0 to b-1 → Process 0
  - Rows b to 2b-1 → Process 1
  - Rows pb to (p+1)b-1 → Process 0 (wrap around)
  
This distributes the workload more evenly because each process gets a mix of 
rows from the top (more work) and bottom (less work) of the matrix.

For upper triangular matrices, the total work for process rank r would be more 
balanced:
  - Each process gets approximately n²/(2p) operations
  - Reduces idle time at synchronization points
  - Improves overall efficiency closer to the regular matrix case

