#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int i=0;  // the global variable in the parent process != the global variable in the child process
void do_something(char *name) {
  const int N = 5;
  for(; i<N; i++){
    sleep((rand() % 3) + 1);
    printf("[%s]This is %d\n", name, i);
  }
}


int main(int argc, char *argv[]) {
  printf("I am %d\n", (int) getpid());

  pid_t pid = fork();
  printf("fork returned %d\n", (int) pid);
  if (pid < 0){
    perror("fork failed");
  }
  if (pid == 0) {
    printf("I am the child %d, and now I will count to 4.\n", (int) getpid());
    do_something("Child");
    printf("Child is exiting\n");
    exit(42);
  }
  printf("I am the parent %d, waiting child to end\n", (int) getpid());
  do_something("Parent");
  int status=0;
  pid_t child_pid =  wait(&status);
  int childReturnValue = WEXITSTATUS(status);
  printf("Parent is aware that child %d is done with status %d.\n", (int) child_pid, childReturnValue);
  return 0;
}
