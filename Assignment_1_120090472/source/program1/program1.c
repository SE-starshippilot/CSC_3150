#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


int main(int argc, char *argv[]) {
  /* fork a child process */
  printf("Process start to fork\n");
  pid_t pid = fork();
  int status = 0;
  if (pid < 0) {
    fprintf(stderr, "Fork Failed");
    return 1;
  }
  /* execute test program */
  if (pid == 0) {
    /* child process */
    printf("I am the Child Process, my pid = %d\n", getpid());
    char *args[argc];
    for (int i = 0; i < argc - 1; i++) {
      args[i] = argv[i + 1];
    }
    args[argc - 1] = NULL;
    printf("Child process start to execute test program:\n");
    execve(args[0], args, NULL);
    printf("Child process should not reach here\n");
  } else {
    /* parent process */
    printf("I am the Parent Process, my pid = %d\n", getpid());
    pid_t ret_pid;
    ret_pid = waitpid(pid, &status, WUNTRACED);
    if (ret_pid < 0) {
      perror("Unsuccessful Return!");
      exit(1);
    }
    /* check child process'  termination status */
    printf("Parent process received SIGCHILD signal\n");
    if (WIFEXITED(status)) {
      printf("Normal termination with EXIT STATUS = %d\n", WEXITSTATUS(status));
    } else if (WIFSTOPPED(status)) {
      printf("Child process STOPPED by signal %d\n", WSTOPSIG(status));
    } else if (WIFSIGNALED(status)) {
      printf("Child process TERMINATED by signal %d\n", WTERMSIG(status));
    } else if (WIFCONTINUED(status)) {
      printf("Child process CONTINUED\n");
    }
  }
  return 0;
}
