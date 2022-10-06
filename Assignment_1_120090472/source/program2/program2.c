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

struct wait_opts {
  enum pid_type wo_type;
  int wo_flags;
  struct pid *wo_pid;
  struct signifo *wo_info;
  int wo_stat;
  struct rusage *wo_rusage;
  wait_queue_entry_t child_wait;
  int notask_error;
};  // cannot export struct, defined here.

struct waitid_info {
  pid_t pid;
  uid_t uid;
  int status;
  int cause;
};

extern int do_execve(struct filename *filename,
                     const char __user *const __user *__argv,
                     const char __user *const __user *__envp);
extern long do_wait(
    struct wait_opts *
        wo);  // exported from
              // [kernel/exit.c](https://elixir.bootlin.com/linux/v5.15.50/source/kernel/exit.c#L1482)
extern struct filename *getname_kernel(
    const char *
        filename);  // exported from
                    // [fs/namei.c](https://elixir.bootlin.com/linux/v5.15.50/source/fs/namei.c#L215)

void my_wait(pid_t pid, int __user *status) {
  pid_t ret_pid;
  struct wait_opts wo;
  struct pid *wo_pid;
  enum pid_type type = PIDTYPE_PID;
  wo_pid = find_get_pid(pid);
  wo.wo_type = type;
  wo.wo_pid = wo_pid;
  wo.wo_flags = WEXITED;
  wo.wo_info = NULL;
  wo.wo_stat = *status;
  wo.wo_rusage = NULL;

  ret_pid = do_wait(&wo);
  *status = wo.wo_stat;
  printk("Return value of do_wait: %d\n", (int) ret_pid);
  printk("status is: %d\n", *status);
  printk("wo.wo_stat is: %d\n", wo.wo_stat);
  printk("Return signal of status: %d\n", __WEXITSTATUS(wo.wo_stat));
  put_pid(wo_pid);
  return;
}

int my_exec(void) {
  const char *path_to_file =
      "/home/vagrant/CSC_3150/Assignment_1_120090472/source/program2/test";
  struct filename *files_stat_struct;
  // const char __user *const __user argv[] = {NULL};
  // const char __user *const __user envp[] = {NULL};
  files_stat_struct = getname_kernel(path_to_file);
  // printk("Return value of do_execve:%d\n", return_status);
  return do_execve(
      getname_kernel(path_to_file), NULL,
      NULL);  // refrence: https://piazza.com/class/l7r9eyyrpo86ds/post/79_f3;
}

// implement fork function
int my_fork(void *argc) {
  int i;
  int status = 0;
  pid_t pid;
  struct kernel_clone_args args;
  struct k_sigaction *k_action = &current->sighand->action[0];
  // set default sigaction for current process
  printk("[program2] : module_init kthread start\n");
  for (i = 0; i < _NSIG; i++) {
    k_action->sa.sa_handler = SIG_DFL;
    k_action->sa.sa_flags = 0;
    k_action->sa.sa_restorer = NULL;
    sigemptyset(&k_action->sa.sa_mask);
    k_action++;
  }
  args.flags = SIGCHLD;
  args.stack = (unsigned long)&my_exec;
  args.stack_size = 0;
  args.parent_tid = NULL;
  args.child_tid = NULL;
  args.tls = 0;
  /* fork a process using kernel_clone or kernel_thread */
  pid = kernel_clone(&args);
  printk("[program2] : The child process has pid = %d\n", (int)pid);
  if (pid) {
    /* parent process */
    printk("[program2] : This is the parent process, pid = %d\n", current->pid);
    /* execute a test program in child process */
    my_exec();
  } else {
    printk("[program2] : Unsuccessful creation of child process\n");
  }
  /* wait until child process terminates */
  my_wait(pid, &status);
  printk("status after:%d\n", __WEXITSTATUS(status));
  return 0;
}

static int __init program2_init(void) {
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

static void __exit program2_exit(void) { printk("[program2] : Module_exit\n"); }

module_init(program2_init);
module_exit(program2_exit);
