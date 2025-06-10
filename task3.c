#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>

void array_operations_sequential(double* a, double* b, double* sum, double* diff, double* prod, double* quot, int N) {
    for (int i = 0; i < N; i++) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        if (b[i] != 0.0) {
            quot[i] = a[i] / b[i];
        } else {
            quot[i] = 0.0;
        }
    }
}

void array_operations_parallel(double* a, double* b, double* sum, double* diff, double* prod, double* quot, 
                             int N, int rank, int size) {

    /* 1. Распределение работы между процессами */
    int chunk_size = N / size;
    int remainder = N % size;
    int start = rank * chunk_size + (rank < remainder ? rank : remainder);
    int end = start + chunk_size + (rank < remainder ? 1 : 0);
    
    /* 2. Параллельное выполнение операций */
    for (int i = start; i < end; i++) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        if (b[i] != 0.0) {
            quot[i] = a[i] / b[i];
        } else {
            quot[i] = 0.0;
        }
    }
}

int main(int argc, char* argv[]) {
    int N = 10000000;
    int opt;
    
    /* 1. Парсинг аргументов командной строки */
    while ((opt = getopt(argc, argv, "n:")) != -1) {
        switch (opt) {
            case 'n':
                N = atoi(optarg);
                if (N <= 0) {
                    fprintf(stderr, "Error: N must be a positive integer.\n");
                    exit(EXIT_FAILURE);
                }
                break;
            default:
                fprintf(stderr, "Usage: %s [-n array_size]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }
    
    /* 2. Инициализация MPI */
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    double *a = NULL, *b = NULL;
    double *sum_seq = NULL, *diff_seq = NULL, *prod_seq = NULL, *quot_seq = NULL;
    double *sum_par = NULL, *diff_par = NULL, *prod_par = NULL, *quot_par = NULL;
    
    if (rank == 0) {
        /* 3. Выделение памяти и инициализация массивов (только в root процессе) */
        a = malloc(N * sizeof(double));
        b = malloc(N * sizeof(double));
        sum_seq = malloc(N * sizeof(double));
        diff_seq = malloc(N * sizeof(double));
        prod_seq = malloc(N * sizeof(double));
        quot_seq = malloc(N * sizeof(double));
        sum_par = malloc(N * sizeof(double));
        diff_par = malloc(N * sizeof(double));
        prod_par = malloc(N * sizeof(double));
        quot_par = malloc(N * sizeof(double));

        if (!a || !b || !sum_seq || !diff_seq || !prod_seq || !quot_seq ||
            !sum_par || !diff_par || !prod_par || !quot_par) {
            perror("Failed to allocate memory for arrays");
            free(a); free(b); free(sum_seq); free(diff_seq); free(prod_seq); free(quot_seq);
            free(sum_par); free(diff_par); free(prod_par); free(quot_par);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* 4. Заполнение массивов случайными значениями */
        srand(time(NULL) ^ getpid());
        for (int i = 0; i < N; i++) {
            a[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
            b[i] = (double)rand() / RAND_MAX * 100.0 + 1.0;
        }
    }
    
    /* 5. Последовательный расчёт (только в root процессе) */
    double start_seq = 0, end_seq = 0;
    if (rank == 0) {
        start_seq = MPI_Wtime();
        array_operations_sequential(a, b, sum_seq, diff_seq, prod_seq, quot_seq, N);
        end_seq = MPI_Wtime();
    }
    
    /*6. Параллельный расчёт */ 
    double start_par = MPI_Wtime();
    
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int chunk_size = N / size;
    int remainder = N % size;
    int local_size = chunk_size + (rank < remainder ? 1 : 0);
    
    /* Выделение памяти под локальные данные */
    double *local_a = malloc(local_size * sizeof(double));
    double *local_b = malloc(local_size * sizeof(double));

    /* Подготовка данных для scatterv */
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        counts[i] = chunk_size + (i < remainder ? 1 : 0);
        displs[i] = i ? displs[i-1] + counts[i-1] : 0;
    }
    
    /* Распределение данных между процессами */
    MPI_Scatterv(a, counts, displs, MPI_DOUBLE, local_a, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, counts, displs, MPI_DOUBLE, local_b, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    /* Выделение памяти под локальные результаты */
    double *local_sum = malloc(local_size * sizeof(double));
    double *local_diff = malloc(local_size * sizeof(double));
    double *local_prod = malloc(local_size * sizeof(double));
    double *local_quot = malloc(local_size * sizeof(double));
    
    /* Выполнение операций над локальными данными */
    for (int i = 0; i < local_size; i++) {
        local_sum[i] = local_a[i] + local_b[i];
        local_diff[i] = local_a[i] - local_b[i];
        local_prod[i] = local_a[i] * local_b[i];
        if (local_b[i] != 0.0) {
            local_quot[i] = local_a[i] / local_b[i];
        } else {
            local_quot[i] = 0.0;
        }
    }
    
    /* Сбор результатов в root процессе */
    MPI_Gatherv(local_sum, local_size, MPI_DOUBLE, sum_par, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_diff, local_size, MPI_DOUBLE, diff_par, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_prod, local_size, MPI_DOUBLE, prod_par, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_quot, local_size, MPI_DOUBLE, quot_par, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double end_par = MPI_Wtime();
    
    /* 8. Вывод результатов и завершение */
    if (rank == 0) {
        printf("Sequential time: %.5f seconds\n", end_seq - start_seq);
        printf("Parallel time: %.5f seconds\n", end_par - start_par);
        
        free(a); free(b); 
        free(sum_seq); free(diff_seq); free(prod_seq); free(quot_seq);
        free(sum_par); free(diff_par); free(prod_par); free(quot_par);
    }
    
    MPI_Finalize();
    return 0;
}
