#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define ROWS 320
#define COLS 320
#define TOTAL_ELEMENTS (ROWS * COLS)

void sequential_operations(double *a, double *b, double *add, double *sub, double *mul, double *div) {
    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        add[i] = a[i] + b[i];
        sub[i] = a[i] - b[i];
        mul[i] = a[i] * b[i];
        div[i] = (b[i] != 0.0) ? a[i] / b[i] : 0.0;
    }
}

void parallel_operations(double *a, double *b, double *add, double *sub, double *mul, double *div, 
                         int start, int end) {
    for (int i = start; i < end; i++) {
        add[i] = a[i] + b[i];
        sub[i] = a[i] - b[i];
        mul[i] = a[i] * b[i];
        div[i] = (b[i] != 0.0) ? a[i] / b[i] : 0.0;
    }
}

void initialize_array(double *arr) {
    for (int i = 0; i < TOTAL_ELEMENTS; i++) {
        arr[i] = (double)rand() / RAND_MAX * 100.0 + 1.0; // 1.0 to 101.0 to avoid division by zero
    }
}

void print_results(double *arr, const char *name) {
    printf("%s (first 10 elements):\n", name);
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    double *a, *b, *add, *sub, *mul, *div;
    double start_time, end_time;
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Allocate memory
    a = (double *)malloc(TOTAL_ELEMENTS * sizeof(double));
    b = (double *)malloc(TOTAL_ELEMENTS * sizeof(double));
    add = (double *)malloc(TOTAL_ELEMENTS * sizeof(double));
    sub = (double *)malloc(TOTAL_ELEMENTS * sizeof(double));
    mul = (double *)malloc(TOTAL_ELEMENTS * sizeof(double));
    div = (double *)malloc(TOTAL_ELEMENTS * sizeof(double));
    
    if (rank == 0) {
        srand(time(NULL));
        initialize_array(a);
        initialize_array(b);
        
        // Sequential version
        start_time = MPI_Wtime();
        sequential_operations(a, b, add, sub, mul, div);
        end_time = MPI_Wtime();
        
        printf("Sequential execution time: %.6f seconds\n", end_time - start_time);
        print_results(add, "Addition");
        print_results(sub, "Subtraction");
        print_results(mul, "Multiplication");
        print_results(div, "Division");
    }
    
    // Broadcast arrays to all processes
    MPI_Bcast(a, TOTAL_ELEMENTS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, TOTAL_ELEMENTS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Parallel version
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    // Calculate workload for each process
    int chunk_size = TOTAL_ELEMENTS / size;
    int remainder = TOTAL_ELEMENTS % size;
    int start = rank * chunk_size + (rank < remainder ? rank : remainder);
    int end = start + chunk_size + (rank < remainder ? 1 : 0);
    
    parallel_operations(a, b, add, sub, mul, div, start, end);
    
    MPI_Gather(add + start, end - start, MPI_DOUBLE, 
               add, end - start, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(sub + start, end - start, MPI_DOUBLE, 
               sub, end - start, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(mul + start, end - start, MPI_DOUBLE, 
               mul, end - start, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(div + start, end - start, MPI_DOUBLE, 
               div, end - start, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    if (rank == 0) {
        printf("\nParallel execution time (%d processes): %.6f seconds\n", size, end_time - start_time);
        print_results(add, "Addition (parallel)");
        print_results(sub, "Subtraction (parallel)");
        print_results(mul, "Multiplication (parallel)");
        print_results(div, "Division (parallel)");
    }
    
    free(a);
    free(b);
    free(add);
    free(sub);
    free(mul);
    free(div);
    
    MPI_Finalize();
    return 0;
}
