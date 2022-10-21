
#include <stdlib.h>
#include <pthread.h>
#include "async.h"
#include "utlist.h"

#include <stdio.h>
#include <unistd.h>

my_queue_t *queue;

void *work_container(void *args){
    while(1){
        pthread_mutex_lock(&(*queue).lock);
        while((*queue).size == 0)
            pthread_cond_wait(&(*queue).not_empty, &(*queue).lock);
        my_item_t *item = (*queue).head;
        DL_DELETE((*queue).head, item);
        (*queue).size--;
        pthread_mutex_unlock(&(*queue).lock);
        item->fx(item->args);
        free(item);
    }
}

void async_init(int num_threads) {
    pthread_t thread;
    queue = malloc(sizeof(my_queue_t));
    (*queue).size = 0;
    (*queue).head = NULL;
    (*queue).lock = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    (*queue).not_empty = (pthread_cond_t)PTHREAD_COND_INITIALIZER;
    for (int i = 0; i < num_threads; i++) {
        pthread_create(&thread, NULL, work_container, NULL);
        pthread_detach(thread);
    }
    return;
}

void async_run(void (*hanlder)(int), int args) {
    pthread_mutex_lock(&(*queue).lock);
    my_item_t *item = malloc(sizeof(my_item_t));
    item->fx = hanlder;
    item->args = args;
    DL_APPEND((*queue).head, item);
    (*queue).size++;
    pthread_mutex_unlock(&(*queue).lock);
    pthread_cond_signal(&(*queue).not_empty);
}