/*
 * File:     treesum_mpi.c
 *
 * Purpose:  Use tree-structured communication to find the global sum
 *           of a random collection of ints.  This version doesn't
 *           require that comm_sz be a power of 2.
 *
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/*-------------------------------------------------------------------
 * Function:
 *  global_sum
 *
 * Purpose:
 *  Implement a global sum using tree-structured communication
 *
 * Notes:
 *  1.  The return value for global sum is only valid on process 0
 */
int global_sum(int my_int /* in */, int my_rank /* in */, int comm_sz /* in */,
               MPI_Comm comm /* in */) {
  int my_sum = my_int; //I can start with my own value being the starting point for the sum
  int partner;
  int step = 1;
 
  //At each iteration, processes are grouped into blocks of size 2 * step
  //Ranks divisible by 2*step RECEIVE
  //Otherwise they SEND and then drop out since they are no longer needed
  while(step < comm_sz) //I need to do this until I have exceeded the number of processors
  {
    //If I have a divisible rank, it needs to receive from its neighbor
    if(my_rank % (2 * step) == 0) 
    {
      //My partner would just be step above me
      partner = my_rank + step;
      //Only receive if the partner rank exists so that non-power-of-two works
      if(partner < comm_sz)
      {
        int recv_val;
        MPI_Recv(&recv_val, 1, MPI_INT, partner, 0, comm, MPI_STATUS_IGNORE);
        my_sum += recv_val;
      }
    }
    else
    {
      //Otherwise I have to send. The target is step below me. I can terminate once I am done
      partner = my_rank - step;
      MPI_Send(&my_sum, 1, MPI_INT, partner, 0, comm);
      break; //Terminate
    }
    //The next step to move to is step * 2
    step *= 2;
  }
  return my_sum;
} /* Global_sum */
