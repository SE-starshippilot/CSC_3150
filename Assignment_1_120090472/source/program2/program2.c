#include <linux/err.h>
#include <linux/fs.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/kmod.h>
#include <linux/kthread.h>
#include <linux/module.h>
#include <linux/pid.h>
#include <linux/printk.h>
#include <linux/sched.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");

/* If WIFEXITED(STATUS), the low-order 8 bits of the status.  */
#define __WEXITSTATUS(status) (((status)&0xff00) >> 8)

/* If WIFSIGNALED(STATUS), the terminating signal.  */
#define __WTERMSIG(status) ((status)&0x7f)

/* If WIFSTOPPED(STATUS), the signal that stopped the child.  */
#define __WSTOPSIG(status) __WEXITSTATUS(status)

/* Nonzero if STATUS indicates normal termination.  */
#define __WIFEXITED(status) (__WTERMSIG(status) == 0)

/* Nonzero if STATUS indicates termination by a signal.  */
#define __WIFSIGNALED(status) (((signed char)(((status)&0x7f) + 1) >> 1) > 0)

/* Nonzero if STATUS indicates the child is stopped.  */
#define __WIFSTOPPED(status) (((status)&0xff) == 0x7f)

typedef struct wait_queue_entry wait_queue_entry_t;
// clang-format off
struct wait_opts {
	enum pid_type 		wo_type;
	int 				wo_flags;
	struct pid 			*wo_pid;
	struct signifo 		*wo_info;
	int 				wo_stat;
	struct rusage 		*wo_rusage;
	wait_queue_entry_t 	child_wait;
	int notask_error;
}; // cannot export struct, defined here.
// clang-format on
extern int do_execve(struct filename *filename,
		     const char __user *const __user *__argv,
		     const char __user *const __user *__envp);
extern long
do_wait(struct wait_opts *
		wo); // exported from
		     // [kernel/exit.c](https://elixir.bootlin.com/linux/v5.15.50/source/kernel/exit.c#L1482)
extern struct filename *getname_kernel(
	const char *
		filename); // exported from
			   // [fs/namei.c](https://elixir.bootlin.com/linux/v5.15.50/source/fs/namei.c#L215)

void my_wait(pid_t pid, int *status)
{
	pid_t		 ret_pid;
	struct pid *     wo_pid = find_get_pid(pid);
	enum pid_type    type   = PIDTYPE_PID;
	struct wait_opts wo     = { .wo_type   = type,
				    .wo_pid    = wo_pid,
				    .wo_flags  = WEXITED | WUNTRACED,
				    .wo_info   = NULL,
				    .wo_stat   = *status,
				    .wo_rusage = NULL };

	ret_pid = do_wait(&wo);
	*status = wo.wo_stat;
	put_pid(wo_pid);
	return;
}

char *getsig(const int __user sig)
{
	switch (sig) {
	case 0:
		return "SIGNULL";
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
	case SIGABRT:
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
	case SIGSYS:
		return "SIGSYS";
	default:
		return "UNKOWN";
	}
}

void kthread_sig_handler(int sig)
{
	printk("[program2] : get %s signal.\n", getsig(sig));
}

int my_exec(void *p)
{
	const char *     path_to_file = "/tmp/test";
	struct filename *files_stat_struct;
	int		 ret_exec;
	files_stat_struct = getname_kernel(path_to_file);
	ret_exec	  = do_execve(
		 files_stat_struct, NULL,
		 NULL); // refrence:
		       // https://piazza.com/class/l7r9eyyrpo86ds/post/79_f3;
	return ret_exec;
}

// implement fork function
int my_fork(void *argc)
{
	int			 i;
	int			 status = 0;
	pid_t			 pid;
	struct kernel_clone_args args = {
		.flags = ((lower_32_bits(SIGCHLD) | CLONE_VM | CLONE_UNTRACED) &
			  ~CSIGNAL),
		.exit_signal = (lower_32_bits(SIGCHLD) & CSIGNAL),
		.stack       = (unsigned long)&my_exec,
		.stack_size  = (unsigned long)NULL,
	};
	struct k_sigaction *k_action = &current->sighand->action[0];
	// set default sigaction for current process
	printk("[program2] : module_init kthread start\n");
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler  = SIG_DFL;
		k_action->sa.sa_flags    = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	/* fork a process using kernel_clone or kernel_thread */
	pid = kernel_clone(&args);
	printk("[program2] : The child process has pid = %d\n", (int)pid);
	printk("[program2] : This is the parent process, pid = %d\n",
	       current->pid);
	if (pid) {
		printk("[program2] : child process\n");
		/* parent process */
		/* execute a test program in child process */
	} else {
		printk("[program2] : Unsuccessful creation of child process\n");
	}
	/* wait until child process terminates */
	my_wait(pid, &status);
	if (__WIFEXITED(status)) {
		printk("[program2] : child process normal exit with status:%d\n",
		       __WEXITSTATUS(status));
	} else if (__WIFSTOPPED(status)) {
		printk("[program2] : get %s signal.\n",
		       getsig(__WSTOPSIG(status)));
		printk("[program2] : child process stopped\n");
		printk("[program2] : the return signal is %d\n",
		       __WSTOPSIG(status));
	} else if (__WIFSIGNALED(status)) {
		printk("[program2] : get %s signal.\n",
		       getsig(__WTERMSIG(status)));
		printk("[program2] : child process terminated\n");
		printk("[program2] : the return signal is %d\n",
		       __WTERMSIG(status));
	}
	return 0;
}

static int __init program2_init(void)
{
	/* write your code here */
	static struct task_struct *my_task;
	printk("[program2] : module_init Tianhao SHI 120090472\n");
	/* create a kernel thread to run my_fork */
	my_task = kthread_create(my_fork, NULL, "program2");
	if (!my_task) {
		printk("[program2] : kthread_run failed\n");
		return -1;
	} else {
		printk("[program2] : module_init create kthread start\n");
		wake_up_process(my_task);
	}
	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
