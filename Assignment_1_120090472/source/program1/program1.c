#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


typedef struct sig_name_prompt {
  char *name;
  char *prompt;
} snp;

snp getsig(const int sig) {
  switch (sig) {
    case 0:
      return (snp){.name = "SIGNULL", .prompt = "NULL"};
    case SIGHUP:
      return (snp){.name = "SIGHUP", .prompt = "hung up"};
    case SIGINT:
      return (snp){.name = "SIGINT", .prompt = "is interrupted"};
    case SIGQUIT:
      return (snp){.name = "SIGQUIT", .prompt = "is quitted"};
    case SIGILL:
      return (snp){.name = "SIGILL", .prompt = "contains illegal instruction"};
    case SIGTRAP:
      return (snp){.name = "SIGTRAP", .prompt = "is trapped"};
    case SIGABRT:
      return (snp){.name = "SIGABRT", .prompt = "is aborted"};
    case SIGBUS:
      return (snp){.name = "SIGBUS", .prompt = "contains bus error"};
    case SIGFPE:
      return (snp){.name = "SIGFPE", .prompt = "contains floating point error"};
    case SIGKILL:
      return (snp){.name = "SIGKILL", .prompt = "is killed"};
    case SIGUSR1:
      return (snp){.name = "SIGUSR1",
                   .prompt = "contains user defined signal 1"};
    case SIGSEGV:
      return (snp){.name = "SIGSEGV", .prompt = "contains segmentation fault"};
    case SIGUSR2:
      return (snp){.name = "SIGUSR2",
                   .prompt = "contains user defined signal 2"};
    case SIGPIPE:
      return (snp){.name = "SIGPIPE", .prompt = "has broken pipe"};
    case SIGALRM:
      return (snp){.name = "SIGALRM", .prompt = "received alarm clock signal"};
    case SIGTERM:
      return (snp){.name = "SIGTERM", .prompt = "is terminated"};
    case SIGSTKFLT:
      return (snp){.name = "SIGSTKFLT", .prompt = "has stack fault"};
    case SIGCHLD:
      return (snp){.name = "SIGCHLD", .prompt = "has child process terminated"};
    case SIGCONT:
      return (snp){.name = "SIGCONT", .prompt = "is continued"};
    case SIGSTOP:
      return (snp){.name = "SIGSTOP", .prompt = "is stopped"};
    case SIGTSTP:
      return (snp){.name = "SIGTSTP", .prompt = "is stopped from tty"};
    case SIGTTIN:
      return (snp){.name = "SIGTTIN", .prompt = "is stopped from tty input"};
    case SIGTTOU:
      return (snp){.name = "SIGTTOU", .prompt = "is stopped from tty output"};
    case SIGURG:
      return (snp){.name = "SIGURG",
                   .prompt = "has urgent condition on socket"};
    case SIGXCPU:
      return (snp){.name = "SIGXCPU", .prompt = "has exceeded CPU time limit"};
    case SIGXFSZ:
      return (snp){.name = "SIGXFSZ", .prompt = "has exceeded file size limit"};
    case SIGVTALRM:
      return (snp){.name = "SIGVTALRM",
                   .prompt = "received virtual alarm clock signal"};
    case SIGPROF:
      return (snp){.name = "SIGPROF", .prompt = "received profiling signal"};
    case SIGWINCH:
      return (snp){.name = "SIGWINCH", .prompt = "has window size changed"};
    case SIGIO:
      return (snp){.name = "SIGIO", .prompt = "has I/O is possible"};
    case SIGPWR:
      return (snp){.name = "SIGPWR", .prompt = "has power failure"};
    case SIGSYS:
      return (snp){.name = "SIGSYS", .prompt = "contains bad system call"};
    default:
      return (snp){.name = "UNKOWN", .prompt = "is unknown"};
  }
}

void sigchld_handler(int sig){
  printf("Parent received %s signal.\n", getsig(sig).name);
}

int main(int argc, char *argv[]) {
  /* fork a child process */
  printf("Process start to fork\n");
  pid_t pid = fork();
  int status = 0;
  signal(SIGCHLD, sigchld_handler);
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
    printf("Status of return:%d\n", status);
    if (WIFEXITED(status)) {
      printf("Normal termination with EXIT STATUS = %d\n", WEXITSTATUS(status));
      return 0;
    } else if (WIFSTOPPED(status)) {
      snp sig = getsig(WSTOPSIG(status));
      printf(
          "Child process STOPPED by signal %s.\n\tReason: child process %s.\n",
          sig.name, sig.prompt);
      return 1;
    } else if (WIFSIGNALED(status)) {
      snp sig = getsig(WTERMSIG(status));
      printf(
          "Child process TERMINATED by signal %s.\n\tReason: child process "
          "%s.\n",
          sig.name, sig.prompt);
      return 1;
    } else if (WIFCONTINUED(status)) {
      printf("Child process CONTINUED\n");
      return 1;
    } else {
      printf("Child process has Unknown termination status\n");
      return 1;
    }
  }
}
