
/*
 * File:     treesum_test_mpi.c
 *
 * Purpose:  test tree-structured communication to find the global sum
 *           of a random collection of ints.  This version doesn't
 *           require that no of processes  be a power of 2.
 *
 * Compile:  mpicc -Wall -o treesum_test_mpi  treesum_test_mpi.c treesum_mpi.c
 * minunit.c Run:      ibrun  -v ./treesum_test_mpi
 *
 * IPP:      Programming Assignment Text Book Problem 3.3 Page 148
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "minunit.h"

int my_rank, no_proc;
MPI_Comm comm;

int global_sum(int my_int, int my_rank, int no_proc, MPI_Comm comm);

/*-------------------------------------------------------------------
 * Test global_sum()
 * If successful, return NULL
 */
char *treesum_test1() {
  /* Your solution */
  //Start by initializing the communicator, rank, and size
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &no_proc);

  //All 1 test
  int result = global_sum(1, my_rank, no_proc, comm);
  if(my_rank == 0) //Only want to test the return of the head process
  {
    //If the sum of all 1s isn't the number of processes, it failed
    mu_assert("global_sum failed for all-ones test", result == no_proc);
  }

  //All given own rank value test
  result = global_sum(my_rank, my_rank, no_proc, comm); 
  int expected = no_proc * (no_proc - 1) / 2; //The arithmetic sum of 1 to no_proc
  if(my_rank == 0)
  { 
    mu_assert("global_sum failed for rank-sum test", result == expected);
  }
  return NULL;
}

/*-------------------------------------------------------------------
 * Run all tests.  Ignore returned messages.
 */
void run_all_tests(void) { mu_run_test(treesum_test1); }

/*-------------------------------------------------------------------
 * The main entrance to run all tests.
 * Only Proc 0 prints the test summary
 */
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &no_proc);
  MPI_Comm_rank(comm, &my_rank);

  run_all_tests();

  if (my_rank == 0) {
    mu_print_test_summary("Summary:");
  }
  MPI_Finalize();
  return 0;
}
