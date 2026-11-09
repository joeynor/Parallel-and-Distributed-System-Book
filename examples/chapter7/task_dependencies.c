/* Build: gcc-16 -std=c11 -O2 -Wall -Wextra -Werror -fopenmp \
 *        examples/chapter7/task_dependencies.c -o task_dependencies
 * Run:   OMP_NUM_THREADS=4 ./task_dependencies
 */
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int left = 0, right = 0, total = 0;

    #pragma omp parallel default(none) shared(left, right, total)
    {
        #pragma omp single
        {
            #pragma omp taskgroup
            {
                #pragma omp task default(none) shared(left) depend(out: left)
                left = 6 * 6;

                #pragma omp task default(none) shared(right) depend(out: right)
                right = 10 * 10;

                /* This task waits for both producers, regardless of order. */
                #pragma omp task default(none) shared(left, right, total) \
                                 depend(in: left, right) depend(out: total)
                total = left + right;
            }
        }
    }

    if (left != 36 || right != 100 || total != 136)
        return EXIT_FAILURE;
    printf("left=%d right=%d total=%d\n", left, right, total);
    return EXIT_SUCCESS;
}
