/* Build: gcc-16 -std=c11 -O2 -Wall -Wextra -Werror -fopenmp \
 *        examples/chapter7/openmp_examples.c -o /tmp/openmp_examples
 * Run:   OMP_NUM_THREADS=4 /tmp/openmp_examples
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

enum { N = 1003, TASKS = 8 };

/* Each loop iteration writes one distinct output element. */
static void scale(const int *a, int *b, int n) {
    #pragma omp parallel for default(none) shared(a, b, n) schedule(static)
    for (int i = 0; i < n; ++i)
        b[i] = 2 * a[i];
}

/* Reduction gives each thread a private partial before combining them. */
static long long sum(const int *a, int n) {
    long long total = 0;
    #pragma omp parallel for default(none) shared(a, n) reduction(+:total)
    for (int i = 0; i < n; ++i)
        total += a[i];
    return total;
}

/* Deliberately simple serial kernel; each larger limit costs more work. */
static int count_primes(int limit) {
    int count = 0;
    for (int x = 2; x <= limit; ++x) {
        int prime = 1;
        for (int d = 2; d <= x / d; ++d)
            if (x % d == 0) { prime = 0; break; }
        count += prime;
    }
    return count;
}

/* One task per independently owned slot; tasks have unequal costs. */
static void task_prime_counts(int *out) {
    #pragma omp parallel default(none) shared(out)
    {
        #pragma omp single
        {
            for (int i = 0; i < TASKS; ++i) {
                int slot = i; /* firstprivate captures this iteration's slot */
                #pragma omp task default(none) firstprivate(slot) shared(out)
                out[slot] = count_primes(5000 * (slot + 1));
            }
            #pragma omp taskwait
        }
    }
}

int main(void) {
    int a[N], b[N], prime_counts[TASKS];
    const int expected_primes[TASKS] =
        {669, 1229, 1754, 2262, 2762, 3245, 3732, 4203};
    for (int i = 0; i < N; ++i) a[i] = i + 1;
    scale(a, b, N);
    for (int i = 0; i < N; ++i)
        if (b[i] != 2 * a[i]) return EXIT_FAILURE;

    long long actual = sum(a, N);
    long long expected = (long long)N * (N + 1) / 2;
    if (actual != expected) return EXIT_FAILURE;

    task_prime_counts(prime_counts);
    for (int i = 0; i < TASKS; ++i)
        if (prime_counts[i] != expected_primes[i]) return EXIT_FAILURE;

    printf("scaled=%d sum=%lld expected=%lld tasks=%d\n",
           N, actual, expected, TASKS);
    return EXIT_SUCCESS;
}
