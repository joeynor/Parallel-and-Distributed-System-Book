/* Compile: cc -std=c11 -O2 -Wall -Wextra -pthread parallel_sum.c -o parallel_sum */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N = 1003, WORKERS = 4 };

typedef struct {
    const int *data;       /* Shared, read-only after thread creation. */
    size_t begin, end;     /* This worker owns [begin, end). */
    long long partial;     /* Only this worker writes its own result. */
} Work;

static void *sum_range(void *argument)
{
    Work *work = argument;
    long long sum = 0;
    for (size_t i = work->begin; i < work->end; ++i)
        sum += work->data[i];
    work->partial = sum;
    return NULL;
}

int main(void)
{
    int data[N];
    pthread_t threads[WORKERS];
    Work work[WORKERS];

    for (size_t i = 0; i < N; ++i)
        data[i] = (int)i + 1;

    for (size_t t = 0; t < WORKERS; ++t) {
        /* Quotient/remainder gives balanced, disjoint blocks without t*N. */
        size_t q = N / WORKERS, r = N % WORKERS;
        work[t] = (Work){data, t*q + (t < r ? t : r),
                         (t+1)*q + (t+1 < r ? t+1 : r), 0};
        int rc = pthread_create(&threads[t], NULL, sum_range, &work[t]);
        if (rc != 0) {
            fprintf(stderr, "pthread_create: %s\n", strerror(rc));
            return EXIT_FAILURE;
        }
    }

    long long total = 0;
    for (size_t t = 0; t < WORKERS; ++t) {
        int rc = pthread_join(threads[t], NULL);
        if (rc != 0) {
            fprintf(stderr, "pthread_join: %s\n", strerror(rc));
            return EXIT_FAILURE;
        }
        total += work[t].partial; /* Join makes the worker's write available. */
        printf("worker %zu: [%zu,%zu) partial=%lld\n",
               t, work[t].begin, work[t].end, work[t].partial);
    }
    printf("total=%lld expected=%lld\n", total,
           (long long)N * (N + 1) / 2);
    return total == (long long)N * (N + 1) / 2 ? EXIT_SUCCESS : EXIT_FAILURE;
}
