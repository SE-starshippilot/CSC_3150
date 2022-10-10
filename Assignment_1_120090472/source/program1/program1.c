#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

char *getsig(const int sig)
{
	switch (sig) {
	case SIGHUP:
		return "SIGHUP";
	case SIGINT:
		return "SIGINT";
	case SIGQUIT:
		return "SIGQUIT";
	case SIGILL:
		return "SIGILL";
	case SIGTRAP:
		return "SIGTRAP";
	case SIGABRT: // or SIGIOT, same in 5.10.15
		return "SIGABRT";
	case SIGBUS:
		return "SIGBUS";
	case SIGFPE:
		return "SIGFPE";
	case SIGKILL:
		return "SIGKILL";
	case SIGUSR1:
		return "SIGUSR1";
	case SIGSEGV:
		return "SIGSEGV";
	case SIGUSR2:
		return "SIGUSR2";
	case SIGPIPE:
		return "SIGPIPE";
	case SIGALRM:
		return "SIGALRM";
	case SIGTERM:
		return "SIGTERM";
	case SIGSTKFLT:
		return "SIGSTKFLT";
	case SIGCHLD:
		return "SIGCHLD";
	case SIGCONT:
		return "SIGCONT";
	case SIGSTOP:
		return "SIGSTOP";
	case SIGTSTP:
		return "SIGTSTP";
	case SIGTTIN:
		return "SIGTTIN";
	case SIGTTOU:
		return "SIGTTOU";
	case SIGURG:
		return "SIGURG";
	case SIGXCPU:
		return "SIGXCPU";
	case SIGXFSZ:
		return "SIGXFSZ";
	case SIGVTALRM:
		return "SIGVTALRM";
	case SIGPROF:
		return "SIGPROF";
	case SIGWINCH:
		return "SIGWINCH";
	case SIGIO:
		return "SIGIO";
	case SIGPWR:
		return "SIGPWR";
	case SIGSYS: // or SIGUNUSED, same in 5.10.15
		return "SIGSYS";
	default:
		return "UNKNOWN SIGNAL";
	}
}

void sigchld_handler(int sig)
{
	printf("Parent received %s signal\n", getsig(sig));
}

int main(int argc, char *argv[])
{
	/* fork a child process */
	printf("Process start to fork\n");
	pid_t pid    = fork();
	int   status = 0;
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
		if (WIFEXITED(status)) {
			printf("Normal termination with EXIT STATUS = %d\n",
			       WEXITSTATUS(status));
			return 0;
		} else if (WIFSTOPPED(status)) {
			printf("Child process get %s signal\n",
			       getsig(WSTOPSIG(status)));
		} else if (WIFSIGNALED(status)) {
			printf("Child process get %s signal\n",
			       getsig(WTERMSIG(status)));
		} else if (WIFCONTINUED(status)) {
			printf("Child process CONTINUED\n");
			return 1;
		} else {
			printf("Child process has Unknown termination status\n");
			return 1;
		}
	}
}
