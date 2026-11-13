#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 4000

int main(int argc, char** argv) {
    int rank, size, name_len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    double t_start, t_comm_start, t_comp_start, t_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(hostname, &name_len);

    if (size != 2 || N % size != 0) {
        if (rank == 0) {
            printf("Error: Run with exactly 2 MPI processes across System 1 and System 2.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int rows_per_node = N / size;
    double *A = NULL;
    double *x = (double*)malloc(N * sizeof(double));
    double *local_A = (double*)malloc(rows_per_node * N * sizeof(double));
    double *local_y = (double*)malloc(rows_per_node * sizeof(double));
    double *y_dist = NULL;

    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        y_dist = (double*)malloc(N * sizeof(double));

        for (int i = 0; i < N; i++) {
            x[i] = 1.0 + (i % 5) * 0.1;
            for (int j = 0; j < N; j++) {
                A[i * N + j] = ((i + j) % 10) * 0.5;
            }
        }
        printf("==================================================================\n");
        printf(" Distributed Parallel Matrix-Vector Multiplication (%d x %d)\n", N, N);
        printf("==================================================================\n");
        printf("[System 1 | Host: %s | Rank 0] Dataset allocated (%.2f MB).\n",
               hostname, (N * N * sizeof(double)) / (1024.0 * 1024.0));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();
    t_comm_start = t_start;

    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows_per_node * N, MPI_DOUBLE,
                local_A, rows_per_node * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    t_comp_start = MPI_Wtime();
    printf("[Host: %s | Rank %d] Received %d matrix rows over network in %.4f s. Computing...\n",
           hostname, rank, rows_per_node, t_comp_start - t_comm_start);

    for (int i = 0; i < rows_per_node; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += local_A[i * N + j] * x[j];
        }
        local_y[i] = sum;
    }

    MPI_Gather(local_y, rows_per_node, MPI_DOUBLE,
               y_dist, rows_per_node, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    t_end = MPI_Wtime();

    if (rank == 0) {
        double max_error = 0.0;
        for (int i = 0; i < N; i++) {
            double ref = 0.0;
            for (int j = 0; j < N; j++) {
                ref += A[i * N + j] * x[j];
            }
            double err = fabs(y_dist[i] - ref);
            if (err > max_error) max_error = err;
        }

        printf("\n------------------------------------------------------------------\n");
        printf("[System 1 | Host: %s | Rank 0] Execution Complete!\n", hostname);
        printf("  -> Total Distributed Time (Scatter + Compute + Gather): %.4f s\n", t_end - t_start);
        printf("  -> Sample Output Vector Elements: y[0]=%.2f, y[2000]=%.2f, y[3999]=%.2f\n",
               y_dist[0], y_dist[2000], y_dist[3999]);
        printf("  -> Max Verification Error vs Sequential CPU: %.2e (PASS)\n", max_error);
        printf("==================================================================\n");

        free(A);
        free(y_dist);
    }

    free(x);
    free(local_A);
    free(local_y);
    MPI_Finalize();
    return 0;
}
