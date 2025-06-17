#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quick_sort_parallel(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        #pragma omp task shared(arr)
        {
            quick_sort_parallel(arr, low, pi - 1);
        }

        #pragma omp task shared(arr)
        {
            quick_sort_parallel(arr, pi + 1, high);
        }

        #pragma omp taskwait
    }
}

int is_sorted(int arr[], int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return 0;
        }
    }
    return 1;
}

void fill_array(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % size;
    }
}

int main() {
    const int size = 1000000;
    int* arr = (int*)malloc(size * sizeof(int));

    srand(time(NULL));
    fill_array(arr, size);

    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quick_sort_parallel(arr, 0, size - 1);
    }
    double end = omp_get_wtime();

    printf("Parallel quick sort took %.3f seconds\n", end - start);
    printf("Array is %s\n", is_sorted(arr, size) ? "sorted" : "not sorted");

    free(arr);
    return 0;
}
