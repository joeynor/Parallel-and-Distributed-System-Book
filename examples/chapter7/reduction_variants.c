#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int a[] = {1, 2, 3, 4};
    int sum = 0, product = 1, low = INT_MAX, high = INT_MIN;
    int all_positive = 1, any_even = 0;
    unsigned bits_and = ~0u, bits_or = 0u, bits_xor = 0u;
    int balance = 100;

    #pragma omp parallel for default(none) shared(a) \
        reduction(+:sum,balance) reduction(*:product) \
        reduction(min:low) reduction(max:high) \
        reduction(&&:all_positive) reduction(||:any_even) \
        reduction(&:bits_and) reduction(|:bits_or) reduction(^:bits_xor)
    for (int i = 0; i < 4; ++i) {
        sum += a[i];
        balance -= a[i];
        product *= a[i];
        if (a[i] < low) low = a[i];
        if (a[i] > high) high = a[i];
        all_positive = all_positive && (a[i] > 0);
        any_even = any_even || (a[i] % 2 == 0);
        bits_and &= (unsigned)a[i];
        bits_or |= (unsigned)a[i];
        bits_xor ^= (unsigned)a[i];
    }
    printf("sum=%d product=%d min=%d max=%d balance=%d\n",
           sum, product, low, high, balance);
    printf("all_positive=%d any_even=%d and=%u or=%u xor=%u\n",
           all_positive, any_even, bits_and, bits_or, bits_xor);
    return sum == 10 && product == 24 && low == 1 && high == 4
        && balance == 90 && all_positive == 1 && any_even == 1
        && bits_and == 0u && bits_or == 7u && bits_xor == 4u
        ? EXIT_SUCCESS : EXIT_FAILURE;
}
