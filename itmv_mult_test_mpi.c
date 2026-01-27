#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "minunit.h"

int my_rank, no_proc;
MPI_Comm comm;

#define MAX_TEST_MATRIX_SIZE 2048

#define FAIL 0
#define SUCC 1

#define procmap(i, r) ((int)floor((double)i / r))
#define local(i, r) (i % r)

#define TEST_CORRECTNESS 1
#define UPPER_TRIANGULAR 1

int itmv_mult(double local_A[], double local_x[],
              double local_d[], double local_y[],
              double global_x[], int matrix_type,
              int n, int t, int blocksize,
              int my_rank, int no_proc,
              MPI_Comm comm);

int itmv_mult_seq(double A[], double x[], double d[], double y[],
                  int matrix_type, int n, int t);

void print_error(char *msgheader, char *msg) {
  if (my_rank == 0) {
    printf("%s Proc 0 error: %s\n", msgheader, msg);
  }
}

void print_itmv_sample(char *msgheader, double A[], double x[], double d[],
                       double y[], int matrix_type, int n, int t) {
  printf("%s Test matrix type %d, size n=%d, t=%d\n", msgheader, matrix_type, n, t);
  if (n < 4 || A == NULL || x == NULL || d == NULL || y == NULL) return;
  printf("%s check x[0-3] %f, %f, %f, %f\n", msgheader, x[0], x[1], x[2], x[3]);
  printf("%s check d[0-3] are %f, %f, %f, %f\n", msgheader, d[0], d[1], d[2], d[3]);
  printf("%s check A[0][0-3] are %f, %f, %f, %f\n", msgheader, A[0], A[1], A[2], A[3]);
  printf("%s check A[1][0-3] are %f, %f, %f, %f\n", msgheader, A[n], A[n + 1], A[n + 2], A[n + 3]);
  printf("%s check A[2][0-3] are %f, %f, %f, %f\n", msgheader, A[2 * n], A[2 * n + 1], A[2 * n + 2], A[2 * n + 3]);
  printf("%s check A[3][0-3] are %f, %f, %f, %f\n", msgheader, A[3 * n], A[3 * n + 1], A[3 * n + 2], A[3 * n + 3]);
}

void print_itmv_sample_distributed(char *msgheader, double local_A[],
                                   double local_x[], double local_d[],
                                   double local_y[], int matrix_type, int n,
                                   int t, int blocksize) {
  int i, local_i;

  printf("%s Distributed blocksize=%d matrix type %d, size n=%d, t=%d\n",
         msgheader, blocksize, matrix_type, n, t);
  if (n < 4) return;
  for (i = 0; i < n; i++) {
    if (procmap(i, blocksize) == my_rank) {
      local_i = local(i, blocksize);
      printf("%s Proc %d x[%d] locally x[%d] = %f; d[%d] locally d[%d] = %f\n",
             msgheader, my_rank, i, local_i, local_x[local_i], i, local_i,
             local_d[local_i]);
      printf(
          "%s Proc %d Row A[%d] locally A[%d] last 4 elements = %f, %f, %f, %f\n",
          msgheader, my_rank, i, local_i, local_A[local_i * n + n - 4],
          local_A[local_i * n + n - 3], local_A[local_i * n + n - 2],
          local_A[local_i * n + n - 1]);
    }
  }
}

double *compute_expected(char *testmsg, int n, int t, int matrix_type) {
  int i, j, start;
  double *A, *x, *d, *y;

  A = malloc(n * n * sizeof(double));
  x = malloc(n * sizeof(double));
  d = malloc(n * sizeof(double));
  y = malloc(n * sizeof(double));
  
  for (i = 0; i < n; i++) {
    x[i] = 0;
    d[i] = (2.0 * n - 1.0) / n;
  }
  for (i = 0; i < n; i++) {
    A[i * n + i] = 0.0;
    if (matrix_type == UPPER_TRIANGULAR)
      start = i + 1;
    else
      start = 0;
    for (j = start; j < n; j++) {
      if (i != j) A[i * n + j] = -1.0 / n;
    }
  }
#ifdef DEBUG1
  print_itmv_sample(testmsg, A, x, d, y, matrix_type, n, t);
#endif
  itmv_mult_seq(A, x, d, y, matrix_type, n, t);

  free(A);
  free(x);
  free(d);
  return y;
}

char *validate_vect(char *msgheader, double global_x[], int n, int t,
                    int matrix_type) {
  int i;
  double *expected;

  if (n <= 0) return "Failed: 0 or negative size";
  if (n > MAX_TEST_MATRIX_SIZE) return "Failed: Too big to validate";

  expected = compute_expected(msgheader, n, t, matrix_type);
  for (i = 0; i < n; i++) {
#ifdef DEBUG1
    printf("%s Proc 0: i=%d  Expected %f Actual %f\n", msgheader, i,
           expected[i], global_x[i]);
#endif
    mu_assert("One mismatch in iterative mat-vect multiplication",
              global_x[i] == expected[i]);
  }
  free(expected);
  return NULL;
}

int allocate_space(double **local_A, double **local_x, double **local_d,
                   double **local_y, double **global_x, int blocksize, int n) {
  int succ = 1, all_succ = 1;

  *local_A = malloc(blocksize * n * sizeof(double));
  *local_x = malloc(blocksize * sizeof(double));
  *local_d = malloc(blocksize * sizeof(double));
  *local_y = malloc(blocksize * sizeof(double));
  *global_x = malloc(n * sizeof(double));
  
  if (*local_A == NULL || *local_x == NULL || *local_d == NULL ||
      *local_y == NULL || *global_x == NULL) {
    if (*local_A != NULL) free(*local_A);
    if (*local_x != NULL) free(*local_x);
    if (*local_d != NULL) free(*local_d);
    if (*local_y != NULL) free(*local_y);
    if (*global_x != NULL) free(*global_x);
    succ = 0;
  }
  
  MPI_Allreduce(&succ, &all_succ, 1, MPI_INT, MPI_PROD, comm);
  return all_succ;
}

int init_matrix(double *local_A, double *local_x, double *local_d,
                double *local_y, int blocksize, int n, int matrix_type,
                int my_rank) {
  int local_i, j, global_i;
  
  if (local_A == NULL || local_x == NULL || local_d == NULL ||
      local_y == NULL || blocksize <= 0)
    return FAIL;

  for (local_i = 0; local_i < blocksize; local_i++) {
    local_x[local_i] = 0.0;
    local_d[local_i] = (2.0 * n - 1.0) / n;
    local_y[local_i] = 0.0;
  }

  for (local_i = 0; local_i < blocksize; local_i++) {
    global_i = my_rank * blocksize + local_i;
    
    for (j = 0; j < n; j++) {
      if (global_i == j) {
        local_A[local_i * n + j] = 0.0;
      } else {
        if (matrix_type == UPPER_TRIANGULAR) {
          if (j > global_i) {
            local_A[local_i * n + j] = -1.0 / n;
          } else {
            local_A[local_i * n + j] = 0.0;
          }
        } else {
          local_A[local_i * n + j] = -1.0 / n;
        }
      }
    }
  }

  return SUCC;
}

char *itmv_test(char *testmsg, int test_correctness, int n, int matrix_type,
                int t) {
  double startwtime = 0, endwtime = 0;
  double *local_A, *local_x, *local_d, *local_y, *global_x;
  int succ, blocksize;
  char *msg;

  blocksize = n / no_proc;
  succ = allocate_space(&local_A, &local_x, &local_d, &local_y, &global_x,
                        blocksize, n);
  if (succ == 0) {
    msg = "Failed space allocation";
    print_error(testmsg, msg);
    return msg;
  }
  succ = init_matrix(local_A, local_x, local_d, local_y, blocksize, n,
                     matrix_type, my_rank);

#ifdef DEBUG1
  print_itmv_sample_distributed(testmsg, local_A, local_x, local_d, local_y,
                                matrix_type, n, t, blocksize);
#endif

  if (my_rank == 0) startwtime = MPI_Wtime();

  succ = itmv_mult(local_A, local_x, local_d, local_y, global_x, matrix_type, n,
                   t, blocksize, my_rank, no_proc, comm);
  if (succ == 0) {
    msg = "Failed matrix multiplication";
    print_error(testmsg, msg);
    return msg;
  }
  if (my_rank == 0) {
    endwtime = MPI_Wtime();
    double latency = endwtime - startwtime;
    double gflops = (double) 2 * n * n * t / 1e9;
    if (matrix_type == UPPER_TRIANGULAR)
      gflops = (double) n * (n + 1) * t / 1e9;
    gflops = gflops / latency;
    printf("%s: Latency = %f sec at Proc 0 of %d processes. %.4f GFLOPS. Matrix dimension %d \n", 
           testmsg, latency, no_proc, gflops, n);
  }
  msg = NULL;
  if (test_correctness == TEST_CORRECTNESS) {
    if (my_rank == 0) {
      msg = validate_vect(testmsg, global_x, n, t, matrix_type);
      if (msg != NULL) print_error(testmsg, msg);
    }
  }
  free(local_A);
  free(local_x);
  free(local_y);
  free(local_d);
  free(global_x);
  return msg;
}

char *itmv_test1() {
  return itmv_test("Test 1", TEST_CORRECTNESS, 4, !UPPER_TRIANGULAR, 1);
}

char *itmv_test2() {
  return itmv_test("Test 2", TEST_CORRECTNESS, 4, !UPPER_TRIANGULAR, 2);
}

char *itmv_test3() {
  return itmv_test("Test 3", TEST_CORRECTNESS, 8, !UPPER_TRIANGULAR, 1);
}

char *itmv_test4() {
  return itmv_test("Test 4", TEST_CORRECTNESS, 8, !UPPER_TRIANGULAR, 2);
}

char *itmv_test5() {
  return itmv_test("Test 5", TEST_CORRECTNESS, 4, UPPER_TRIANGULAR, 1);
}

char *itmv_test6() {
  return itmv_test("Test 6", TEST_CORRECTNESS, 4, UPPER_TRIANGULAR, 2);
}

char *itmv_test7() {
  return itmv_test("Test 7", TEST_CORRECTNESS, 8, UPPER_TRIANGULAR, 1);
}

char *itmv_test8() {
  return itmv_test("Test 8", TEST_CORRECTNESS, 8, UPPER_TRIANGULAR, 2);
}

char *itmv_test9() {
  return itmv_test("Test 9: n=2K t=1K", !TEST_CORRECTNESS, 2048,
                   !UPPER_TRIANGULAR, 1024);
}

char *itmv_test10() {
  return itmv_test("Test 10: n=2K t=1K upper", !TEST_CORRECTNESS, 2048,
                   UPPER_TRIANGULAR, 1024);
}

char *itmv_test11() {
  return itmv_test("Test 11: n=4K t=1K", !TEST_CORRECTNESS, 4096,
                   !UPPER_TRIANGULAR, 1024);
}

char *itmv_test12() {
  return itmv_test("Test 12: n=4K t=1K upper", !TEST_CORRECTNESS, 4096,
                   UPPER_TRIANGULAR, 1024);
}

char *itmv_test13() {
  return itmv_test("Test 13: n=8K t=1K", !TEST_CORRECTNESS, 4096 * 2,
                   !UPPER_TRIANGULAR, 1024);
}

char *itmv_test14() {
  return itmv_test("Test 14: n=8K t=1K upper", !TEST_CORRECTNESS, 4096 * 2,
                   UPPER_TRIANGULAR, 1024);
}

void run_all_tests(void) {
  mu_run_test(itmv_test1);
  mu_run_test(itmv_test2);
  mu_run_test(itmv_test3);
  mu_run_test(itmv_test4);
  mu_run_test(itmv_test5);
  mu_run_test(itmv_test6);
  mu_run_test(itmv_test7);
  mu_run_test(itmv_test8);
  mu_run_test(itmv_test11);
  mu_run_test(itmv_test12);
}

void testmain() {
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &no_proc);
  MPI_Comm_rank(comm, &my_rank);

  run_all_tests();

  if (my_rank == 0) {
    mu_print_test_summary("Summary:");
  }
}
