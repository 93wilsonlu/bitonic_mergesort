#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <valarray>

bool cmp(const double& l, const double& r) {
    return l > r;
}

void printresult(double* arr, int size, double elapsed_time, int rank) {
    printf("Proc %d Elapsed time : %lf\n", rank, elapsed_time);
    for (int i = 0; i < size; i++) {
        if (arr[i] == __DBL_MAX__) {
            printf("inf ");
        } else {
            printf("%lf ", arr[i]);
        }
    }
    puts("");
}

void merge(double* chunk, int chunksize, int descending) {
    int half_n = chunksize >> 1;
#pragma omp parallel for
    for (int i = 0; i < half_n; i++) {
        if (descending ? chunk[i] < chunk[i + half_n]
                       : chunk[i] > chunk[i + half_n]) {
            std::swap(chunk[i], chunk[i + half_n]);
        }
    }
}

void sort_from_bitonic_array(double* chunk, int chunksize, int descending) {
    for (int k = chunksize >> 1; k > 0; k >>= 1) {
#pragma omp parallel for
        for (int i = 0; i < chunksize; i++) {
            int j = i ^ k;
            if (i < j && j < chunksize &&
                (descending ? chunk[i] < chunk[j] : chunk[i] > chunk[j])) {
                std::swap(chunk[i], chunk[j]);
            }
        }
    }
}

void bitonic_sort(double* chunk, int chunksize, int descending) {
    for (int step = 2; step <= chunksize; step <<= 1) {
        for (int k = step >> 1; k > 0; k >>= 1) {
#pragma omp parallel for
            for (int i = 0; i < chunksize; i++) {
                int j = i ^ k;
                if (i < j && j < chunksize &&
                    ((step == chunksize ? descending : i & step)
                         ? chunk[i] < chunk[j]
                         : chunk[i] > chunk[j])) {
                    std::swap(chunk[i], chunk[j]);
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // cpu_set_t cpu_set;
    // sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // int ncpus = CPU_COUNT(&cpu_set);
    // printf("CPU: %d\n", ncpus);
    int rank, size;
    long long ARRAY_SIZE = atoll(argv[1]), expanded_size = 1;
    while (expanded_size < ARRAY_SIZE) {
        expanded_size <<= 1;
    }

    char *input_filename = argv[2], *output_filename = argv[3];
    MPI_File input_file, output_file;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long chunksize = expanded_size / size, display = rank * chunksize;
    double* chunk = (double*)malloc(chunksize * 2 * sizeof(double));
    std::fill_n(chunk, chunksize, __DBL_MAX__);

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY,
                  MPI_INFO_NULL, &input_file);
    if (display < ARRAY_SIZE) {
        MPI_File_read_at(input_file, sizeof(double) * display, chunk, chunksize,
                         MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&input_file);
    if (display < ARRAY_SIZE) {
        bitonic_sort(chunk, chunksize, rank & 1);
    }
    MPI_Barrier(MPI_COMM_WORLD); // used to profiling

    start_time = MPI_Wtime();

    for (int step = 2; step <= size; step <<= 1) {
        for (int k = step >> 1; k > 0; k >>= 1) {
            if (rank & k) {
                MPI_Sendrecv_replace(chunk, chunksize >> 1, MPI_DOUBLE,
                                     rank - k, 0, rank - k, 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);

                merge(chunk, chunksize, rank & step);

                MPI_Sendrecv_replace(chunk, chunksize >> 1, MPI_DOUBLE,
                                     rank - k, 0, rank - k, 0, MPI_COMM_WORLD,
                                     MPI_STATUS_IGNORE);
            } else {
                MPI_Sendrecv_replace(chunk + (chunksize >> 1), chunksize >> 1,
                                     MPI_DOUBLE, rank + k, 0, rank + k, 0,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                merge(chunk, chunksize, rank & step);

                MPI_Sendrecv_replace(chunk + (chunksize >> 1), chunksize >> 1,
                                     MPI_DOUBLE, rank + k, 0, rank + k, 0,
                                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        sort_from_bitonic_array(chunk, chunksize, rank & step);
    }

    MPI_Barrier(MPI_COMM_WORLD); // used to profiling

    MPI_File_open(MPI_COMM_WORLD, output_filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                  &output_file);
    if (display < ARRAY_SIZE) {
        MPI_File_write_at(output_file, sizeof(double) * display, chunk,
                          std::min(chunksize, ARRAY_SIZE - display), MPI_DOUBLE,
                          MPI_STATUS_IGNORE);
    }
    MPI_File_close(&output_file);

    // if (rank == 0) {
    //     printresult(chunk, chunksize, end_time - start_time, rank);
    // }

    MPI_Finalize();
    return 0;
}
