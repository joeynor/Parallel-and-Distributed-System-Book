/* Compare mutex-protected and C11 atomic counters. */
#include <pthread.h>   /* Threads and mutexes. */
#include <stdatomic.h> /* Atomic counter operations. */
#include <stdio.h>
#include <stdlib.h>
/* Each worker adds ITERATIONS to both counters. */
enum { WORKERS = 2, ITERATIONS = 100000 };
static long locked_count = 0; /* Access only while holding count_lock. */
static atomic_long atomic_count = ATOMIC_VAR_INIT(0);
static pthread_mutex_t count_lock = PTHREAD_MUTEX_INITIALIZER; /* Unlocked. */
/* Relaxed ordering is enough: the atomic count publishes no other data. */
static void *worker(void *unused) {
    (void)unused; /* No per-worker argument is needed. */
    for (int i = 0; i < ITERATIONS; ++i) {
        if (pthread_mutex_lock(&count_lock) != 0) abort();
        ++locked_count; /* The mutex makes this increment exclusive. */
        if (pthread_mutex_unlock(&count_lock) != 0) abort();
        atomic_fetch_add_explicit(&atomic_count, 1, memory_order_relaxed);
    }
    return NULL; /* Worker has finished. */
}
/* Join every worker before reading the final counters. */
int main(void) {
    pthread_t threads[WORKERS]; /* Handles for create and join. */
    for (int i = 0; i < WORKERS; ++i)
        if (pthread_create(&threads[i], NULL, worker, NULL) != 0) abort();
    for (int i = 0; i < WORKERS; ++i)
        if (pthread_join(threads[i], NULL) != 0) abort();
    printf("mutex=%ld atomic=%ld expected=%ld\n",
           locked_count,
           atomic_load_explicit(&atomic_count, memory_order_relaxed),
           (long)WORKERS * ITERATIONS); /* Independent expected value. */
    if (pthread_mutex_destroy(&count_lock) != 0) abort();
    return 0;
}
