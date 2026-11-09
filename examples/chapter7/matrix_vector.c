/* Build: gcc-16 -std=c11 -O2 -Wall -Wextra -Werror -fopenmp \
 *        examples/chapter7/matrix_vector.c -o matrix_vector
 * Run:   OMP_NUM_THREADS=4 ./matrix_vector
 */
#include <stdio.h>
#include <stdlib.h>

enum { ROWS = 4, COLS = 3 };

int main(void) {
    const int a[ROWS][COLS] = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}
    };
    const int x[COLS] = {1, 2, 3};
    const int expected[ROWS] = {14, 32, 50, 68};
    int doubled[ROWS][COLS];
    int y[ROWS];

    /* The collapsed iterations still own distinct matrix elements. */
    #pragma omp parallel for collapse(2) default(none) shared(a, doubled)
    for (int i = 0; i < ROWS; ++i)
        for (int j = 0; j < COLS; ++j)
            doubled[i][j] = 2 * a[i][j];

    /* Row i is the only writer of y[i]; sum is private to that iteration. */
    #pragma omp parallel for default(none) shared(a, x, y)
    for (int i = 0; i < ROWS; ++i) {
        int sum = 0;
        for (int j = 0; j < COLS; ++j)
            sum += a[i][j] * x[j];
        y[i] = sum;
    }

    for (int i = 0; i < ROWS; ++i) {
        if (y[i] != expected[i]) return EXIT_FAILURE;
        for (int j = 0; j < COLS; ++j)
            if (doubled[i][j] != 2 * a[i][j]) return EXIT_FAILURE;
    }
    printf("matvec=[%d,%d,%d,%d] doubled=ok\n",
           y[0], y[1], y[2], y[3]);
    return EXIT_SUCCESS;
}
