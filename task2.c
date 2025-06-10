#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

//Пузырьковая
void bubble_sort_sequential(double *arr, int n) {
    for (int i = 0; i < n - 1; ++i) {
        int swapped = 0;
        for (int j = 0; j < n - 1 - i; ++j) {
            if (arr[j] > arr[j+1]) {
                double tmp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp;
                swapped = 1;
            }
        }
        if (!swapped) break;
    }
}

int main(int argc, char *argv[]) {
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

    double *arr_seq = malloc(N * sizeof(double));
    double *arr_par = malloc(N * sizeof(double));
    if (!arr_seq || !arr_par) {
        perror("malloc");
        return EXIT_FAILURE;
    }
    srand(time(NULL) ^ getpid());
    for (int i = 0; i < N; ++i) {
        double v = (double)rand() / RAND_MAX;
        arr_seq[i] = arr_par[i] = v;
    }

    // Последовательная часть
    double start_seq = MPI_Wtime();
    bubble_sort_sequential(arr_seq, N);
    double end_seq   = MPI_Wtime();
    printf("Sequential time: %.10f seconds\n", end_seq - start_seq);

    // MPI
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int base = N / world_size;
    int rem  = N % world_size;
    int local_n = (world_rank < rem ? base + 1 : base);

    int *counts = malloc(world_size * sizeof(int));
    int *displs = malloc(world_size * sizeof(int));
    if (world_rank == 0) {
        int offset = 0;
        for (int p = 0; p < world_size; ++p) {
            counts[p] = (p < rem ? base + 1 : base);
            displs[p] = offset;
            offset += counts[p];
        }
    }

    double *local = malloc(local_n * sizeof(double));
    MPI_Scatterv(arr_par, counts, displs, MPI_DOUBLE,
                 local,    local_n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_par = MPI_Wtime();

    bubble_sort_sequential(local, local_n);

    MPI_Gatherv(local, local_n, MPI_DOUBLE,
                arr_par, counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        double *merged = malloc(N * sizeof(double));
        int *idx = calloc(world_size, sizeof(int));
        for (int i = 0; i < N; ++i) {
            int best_p = -1;
            double best_v = 0;
            for (int p = 0; p < world_size; ++p) {
                int start = displs[p];
                int cnt   = counts[p];
                if (idx[p] < cnt) {
                    double v = arr_par[start + idx[p]];
                    if (best_p < 0 || v < best_v) {
                        best_p = p;
                        best_v = v;
                    }
                }
            }
            merged[i] = best_v;
            idx[best_p]++;
        }
        double end_par = MPI_Wtime();
        printf("Parallel time:   %.10f seconds\n", end_par - start_par);
        free(merged);
        free(idx);
    }

    free(arr_seq);
    free(arr_par);
    free(local);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}
