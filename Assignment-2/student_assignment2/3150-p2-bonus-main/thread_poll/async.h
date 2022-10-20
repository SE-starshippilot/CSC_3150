#ifndef __ASYNC__
#define __ASYNC__

#include <pthread.h>

typedef void (*worker_func)(int args); // the ACTUAL function that will be executed

typedef struct my_item {
  worker_func fx;         // the function to be executed
  int args;             // the arguments of that function
  struct my_item *next;
  struct my_item *prev;
} my_item_t;

typedef struct my_queue {
  int size;
  my_item_t *head;
  pthread_mutex_t lock;       // mutex lock
  pthread_cond_t not_empty;   // conditional var
} my_queue_t;


void async_init(int);
void async_run(void (*fx)(int), int args);

#endif
