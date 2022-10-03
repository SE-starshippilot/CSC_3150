#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

extern do_execve(struct filename *filename, const char *const argv[], const char *const envp[]);

#define FORK(stack_start)                                                      \
	do_fork(SIGCHLD, stack_start, NULL, 0, NULL,                           \
		NULL) // the do_fork is extern

struct wait_ops {
	enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct waitid_info *wo_info;
	int wo_stat;
	struct rusage *wo_rusage;
	wait_queue_t child_wait;
	int notask_error;
}

extern long do_wait(struct wait_ops *wo);
extern struct filename *getname(const char __user *filename);

int my_exec(void){
	
	return 0;
}

//implement fork function
int my_fork(void *argc)
{
	//set default sigaction for current process
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for (i = 0; i < _NSIG; i++) {
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}

	/* fork a process using kernel_clone or kernel_thread */

	/* execute a test program in child process */

	/* wait until child process terminates */

	return 0;
}

void my_wait(pid_t pid, int *status)
{
	W_OPS wo;
}

static int __init program2_init(void)
{
	printk("[program2] : Module_init Tianhao SHI 120090472\n");

	/* write your code here */

	/* create a kernel thread to run my_fork */
	static struct *task_struct my_fork_task;
	my_fork_task = kthread_run(my_fork, NULL, "my_fork");
	if (my_fork_task){
		
	}
	
	return 0;
}

static void __exit program2_exit(void)
{
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
