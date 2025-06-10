#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int N = 100000;
    int opt;
    while ((opt = getopt(argc, argv, "n:")) != -1) {
        if (opt == 'n') {
            N = atoi(optarg);
            if (N <= 0) {
                fprintf(stderr, "Error: N must be a positive integer.\n");
                return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr, "Usage: %s [-n array_size]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    double* array = malloc(N * sizeof(double));
    if (!array) {
        perror("malloc");
        return EXIT_FAILURE;
    }
    srand((unsigned)time(NULL) ^ getpid());
    for (int i = 0; i < N; ++i) {
        array[i] = (double)rand() / RAND_MAX;
    }

    // -----------------------
    // 1. Последовательный расчёт
    // -----------------------
    double start_seq = MPI_Wtime();
    double sum_seq = 0.0;
    for (int i = 0; i < N; ++i) {
        sum_seq += array[i];
    }
    double end_seq = MPI_Wtime();
    printf("Sequential time: %.10f seconds\n", end_seq - start_seq);

    // -----------------------
    // 2. MPI-параллельный расчёт
    // -----------------------
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int base = N / world_size;
    int rem  = N % world_size;
    int local_n = (world_rank < rem ? base + 1 : base);

    int *counts = malloc(world_size * sizeof(int));
    int *displs = malloc(world_size * sizeof(int));
    if (!counts || !displs) {
        perror("malloc");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (world_rank == 0) {
        int offset = 0;
        for (int p = 0; p < world_size; ++p) {
            counts[p] = (p < rem ? base + 1 : base);
            displs[p] = offset;
            offset += counts[p];
        }
    }
    MPI_Bcast(counts, world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, world_size, MPI_INT, 0, MPI_COMM_WORLD);

    double *local = malloc(local_n * sizeof(double));
    if (!local) {
        perror("malloc");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Scatterv(array, counts, displs, MPI_DOUBLE,
                 local,  local_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_par = MPI_Wtime();

    double sum_local = 0.0;
    for (int i = 0; i < local_n; ++i) {
        sum_local += local[i];
    }
    double sum_global = 0.0;
    MPI_Reduce(&sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_par = MPI_Wtime();
    if (world_rank == 0) {
        printf("Parallel time:   %.10f seconds\n", end_par - start_par);
    }

    free(array);
    free(local);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}
