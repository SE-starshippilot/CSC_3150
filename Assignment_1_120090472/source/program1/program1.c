#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[])
{
	/* fork a child process */
	printf("Process start to fork\n");
	pid_t pid = fork();
	int status = 0;
	if (pid < 0) {
		fprintf(stderr, "Fork Failed");
		return 1;
	}
	/* execute test program */
	char *args[argc - 1];
	for (int i = 1; i <= argc - 1; i++) {
		args[i - 1] = argv[i];
	}
	if (pid == 0) {
		/* child process */
		execvp(args[0], args);
	} else {
		/* parent process */
		wait(&status); /* wait for child process terminates */
		pid_t ret_pid;
		ret_pid = waitpid(pid, &status, 0);
		if (ret_pid < 0) {
			perror("Unsuccessful Return!");
			exit(1);
		}
		/* check child process'  termination status */
		printf("Parent process received %s signal", sys_signame[status]);
		if (WIFEXITED(status)) {
			printf("Normal termination with exit status = %d\n",
			       WEXITSTATUS(status));
		}
	}
	return 0;
}
