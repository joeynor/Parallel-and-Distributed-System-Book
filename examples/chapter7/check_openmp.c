/* Build with an OpenMP-capable GCC and -fopenmp. */
#include <omp.h>
#include <stdio.h>

#ifndef _OPENMP
#error OpenMP is disabled: compile with -fopenmp
#endif

int main(void) {
    int team = 0;
    int counted = 0;

    #pragma omp parallel default(none) shared(team) reduction(+:counted)
    {
        #pragma omp single
        team = omp_get_num_threads();
        counted += 1;
    }

    printf("team=%d counted=%d\n", team, counted);
    return team == counted && team > 0 ? 0 : 1;
}
