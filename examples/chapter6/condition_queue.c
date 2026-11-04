/* Compile: cc -std=c11 -O2 -Wall -Wextra -pthread condition_queue.c -o condition_queue */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

enum { CAPACITY = 3, ITEMS = 12 };

typedef struct {
    int values[CAPACITY];
    size_t head, tail, count;
    int closed;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty, not_full;
} Queue;

static Queue queue = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .not_empty = PTHREAD_COND_INITIALIZER,
    .not_full = PTHREAD_COND_INITIALIZER
};

/* Only the producer calls put. The mutex protects all queue fields. */
static void put(int value)
{
    pthread_mutex_lock(&queue.mutex);
    while (queue.count == CAPACITY)
        pthread_cond_wait(&queue.not_full, &queue.mutex);
    queue.values[queue.tail] = value;
    queue.tail = (queue.tail + 1) % CAPACITY;
    ++queue.count;
    pthread_cond_signal(&queue.not_empty);
    pthread_mutex_unlock(&queue.mutex);
}

/* Return 0 only when closed AND drained; output is set only on success. */
static int take(int *output)
{
    pthread_mutex_lock(&queue.mutex);
    while (queue.count == 0 && !queue.closed)
        pthread_cond_wait(&queue.not_empty, &queue.mutex);
    if (queue.count == 0) {
        pthread_mutex_unlock(&queue.mutex);
        return 0;
    }
    *output = queue.values[queue.head];
    queue.head = (queue.head + 1) % CAPACITY;
    --queue.count;
    pthread_cond_signal(&queue.not_full);
    pthread_mutex_unlock(&queue.mutex);
    return 1;
}

static void *produce(void *unused)
{
    (void)unused;
    for (int value = 1; value <= ITEMS; ++value)
        put(value);
    pthread_mutex_lock(&queue.mutex);
    queue.closed = 1;
    pthread_cond_broadcast(&queue.not_empty);
    pthread_mutex_unlock(&queue.mutex);
    return NULL;
}

static void *consume(void *result)
{
    long long *sum = result;
    int value;
    while (take(&value))
        *sum += value; /* Only this consumer writes sum. */
    return NULL;
}

int main(void)
{
    pthread_t producer, consumer;
    long long sum = 0;
    int rc = pthread_create(&consumer, NULL, consume, &sum);
    if (rc != 0) { fprintf(stderr, "consumer create failed: %d\n", rc); return 1; }
    rc = pthread_create(&producer, NULL, produce, NULL);
    if (rc != 0) { fprintf(stderr, "producer create failed: %d\n", rc); return 1; }
    rc = pthread_join(producer, NULL);
    if (rc != 0) { fprintf(stderr, "producer join failed: %d\n", rc); return 1; }
    rc = pthread_join(consumer, NULL);
    if (rc != 0) { fprintf(stderr, "consumer join failed: %d\n", rc); return 1; }
    printf("sum=%lld expected=%d\n", sum, ITEMS * (ITEMS + 1) / 2);
    pthread_cond_destroy(&queue.not_empty);
    pthread_cond_destroy(&queue.not_full);
    pthread_mutex_destroy(&queue.mutex);
    return sum == ITEMS * (ITEMS + 1) / 2 ? 0 : 1;
}
