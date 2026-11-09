/* Build: gcc-16 -std=c11 -O2 -Wall -Wextra -Werror -fopenmp \
 *        examples/chapter7/worksharing_phases.c -o worksharing_phases
 * Run:   OMP_NUM_THREADS=4 ./worksharing_phases
 */
#include <stdio.h>
#include <stdlib.h>

enum { N = 16 };

int main(void) {
    int a[N], b[N], c[N];

    #pragma omp parallel default(none) shared(a, b, c)
    {
        /* Phase A and phase B do not read each other's outputs. */
        #pragma omp for nowait schedule(static)
        for (int i = 0; i < N; ++i)
            a[i] = i + 1;

        /* The implicit barrier at this loop's end finishes A and B. */
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i)
            b[i] = 2 * (i + 1);

        /* b[N-1-i] may have been written by a different thread. */
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i)
            c[i] = a[i] + b[N - 1 - i];
    }

    for (int i = 0; i < N; ++i)
        if (c[i] != 2 * N + 1 - i) return EXIT_FAILURE;
    printf("verified=%d last=%d\n", N, c[N - 1]);
    return EXIT_SUCCESS;
}
