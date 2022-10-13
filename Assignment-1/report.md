# CSC 3150 Assignment 1 Report

Tianhao SHI, 120090472

Oct 2022

****

## Environment

### Environment Info

- **Linux Kernel Version**: 5.10
- **GCC Version**: 5.4.0
- **Ubuntu Distribution Version**: 16.04

![image-20221008230513455](/Users/shitianhao/Library/CloudStorage/OneDrive-CUHK-Shenzhen/Year 3 Term 1/CSC 3150/Assignment_1_120090472/image-20221008230513455.png)

### Environment Setup

The following steps are executed during my VM setup:

1. Expand the disk space

   1. Make a copy of the default ==vmdk== file in the format of ==vdl== 
   2. Modify the size of ==vdl== file
   3. Convert the format back to ==vmdk== and update VM setting to boot from new disk

2. Updating/ Upgrading packages (`sudo apt-get update && sudo apt`)

   - Apt source are modified for faster download

3. Install essential packages

4. Download kernel source code

5. Modify the source code and export some symbols using `EXPORT_SYMBOL()`

   - `do_wait()`[ from kernel/exit.c](https://elixir.bootlin.com/linux/latest/source/kernel/exit.c#L1486)
   - `do_execve()` [from fs/exec.c](https://elixir.bootlin.com/linux/latest/source/fs/exec.c#L2010)

   - `getname_kernel()` [from fs/namei.c]()

   - `kernel_clone()`[from kernel/fork.c](https://elixir.bootlin.com/linux/latest/source/kernel/fork.c#L2630)

5. Compile kernel source code and reboot
   1. save a snapshot of VM
   2. switch into root user `sudo su`
   3. copy the original `.config` file to the downloaded source code folder
   4. `make mrproper`
   5. `make clean`
   6. `make menuconfig` load and save the config file
   7. `make bzImage -j$(nproc)`
   8. `make modules -j$(nproc)`
   9. `make modules_install`
   10. `make install`
   11. `reboot`

## Code design

The output of the tasks can be found in the appendix

### Task 1

#### Fork a child process

​	In task 1, a child is created by calling the `fork()` function. If the child process is created successfully, the program will enter a conditional statement. The child will execute codes in the `pid==0` branch.

#### Execute the program

​	In the previous step, the child process will execute the code enclosed in `pid==0` branch. In this branch, the `execve()` fuction is called. This function takes in 3 arguments: the *filename*,  which is the path to the executable file (machine code); the *arguments* that the exeutable file takes (there should be none in this assignment, however); and the *environment variable* array used during the execution. It should be noted that the path to the program to be executed is passed as an argument, and is stored at *argv[1]*.  The *arguments* corresponds to `argv[2:]` in the arguments passed to `program1.c`, and should end with a `NULL`. The child will then load and execute the specified file.

#### Parent receive SIGCHLD

​	In order to receive SIGCHLD sent from the child process, a signal handler isneeded. I defined a function `sigchld_handler` function in `program1.c` and configed the `signal()` function so that whenever SIGCHLD is raised, a message will be printed.

#### Print out the termination status of child process

​	In the first step, while child process executes code in `pid==0` branch, parent process will go into the `else` branch. It will call the `wait_pid(pid_t pid, int* status, int options)` function. From the parent process's point of view, the `pid` will be the child process's pid. The `status`, passed by refrence, will record the termination status. In task 1, the `options` is set as `WUNTRACED` because it is possible that the child will be killed/stopped.

​	Macros defined in nolib.c can be used to analyze the exit status

- Whether the child process is exited can be checked by `WIFEXITED`
  - The return status can be required by `WEXITSTATUS`
- Whether the child process is terminated by a signal can be checked by `WIFSIGNALED`
  - The signal can be further achieved analyzed by `WTERMSIG`
- Whether the child process is stopped by a signal can be checked by `WIFSTOPPED`
  - The signal can be further acquired by `WSTOPSIG`
- Although not covered in the testcase, whether a process is resumed can be checked by `WIFCONTINUED`

​	To make the output more semantic, I wrote a function called `getsig(int sig)` that returns the string of the signal name. The underlying value of each process can be found [here](https://elixir.bootlin.com/linux/v5.10.146/source/arch/x86/include/uapi/asm/signal.h). It should be noted that the signals are platform-dependent. In this assignment I selected x86 architecture. ==I assume the testing environment should be x86 as well.==

<img src="/Users/shitianhao/Library/Application Support/typora-user-images/image-20221009141136650.png" alt="image-20221009141136650" style="zoom:50%;" />



### Task 2

#### Create a kernel thread to run my_fork

​	A new kernel thread will be created using the `kthread_create(threadfn, data, namefmt, arg...)`. The first argument is the self-implemented function `my_fork`. The `data` argument is set to NULL, and `namefmt` is just a name for the thread.

​	The `kthread_create` only create a new thread, and a `task_struct` struct is returned., to start the thread, I used `wake_up_process(struct taskstruct *p)`, and passed a pointer to the newly created task_struct.

#### Fork a process to execute the test program

​	As of version 5.10.50, the system call `fork` is done through calling the `kernel_thread()` function, which is just a wrapper function of `kernel_clone()`. Therefore, I only exported  the latter. This function takes in a `kernel_clone_args` struct, which describes how the child process is forked. To guarantee same behaviour, my args are modified from the settings in the `kernel_thread` function:

```c
struct kernel_clone_args args = {
		.flags		= ((lower_32_bits(flags) | CLONE_VM |
				    CLONE_UNTRACED) & ~CSIGNAL),
		.exit_signal	= (lower_32_bits(flags) & CSIGNAL),
		.stack		= (unsigned long)fn,
		.stack_size	= (unsigned long)arg,
	};
```

In the context of task 2, the flags are set to **SIGCHLD**, the fn is a self-written **`my_exec`**, which is responsible for executing the test program. The arg is simply set as NULL.

#### Print the process id of both the parent and child process

​	The parent process's pid is readily achieved though the `current` pointer (current->pid). The child process's pid, assuming successful creation, is simply the return value of `kernel_clone`.

#### Exectue the test program

​	The execution is completed by calling the `do_execve` function, which is the kernel-space version of the `execve` function. It also takes three arguments. The first argument is a `struct filename` pointer. The actual path to the file needs to be transferred to this type by calling the `getname_kernel()` function (as discussed in the piazza forum, the getname function will not work in 5.10.50). The other two arguments, `argv` and `envp` are simply set to NULL. 

#### Parent wait for child process and capture signal

​	The waiting process is achieved by calling the `my_wait` function, which takes 2 arguments: the target process's pid ( in this case, the pid of child process), and an integer status (passed by refrence). This function is a wrapper function for the acutal `do_wait` function. To configure the waiting process, a `wait_opts` struct is defined:

```c
struct wait_opts {
	enum pid_type					wo_type;
	int										wo_flags;
	struct pid						*wo_pid;
	struct waitid_info		*wo_info;
	int										wo_stat;
	struct rusage					*wo_rusage;
	wait_queue_entry_t		child_wait;
  int										notask_error;
};
```

It should be noted that wo_stat is the `status` argument. After the `do_wait` completion, the status should be updated. The wo_flags should be WEXITED | WUNTRACED because we want to trace not only normal exit of child process, but also whether it is beed stopped or termnated by signals. The wo_pid is achieved through looking up a hash table using the `find_get_pid` function.

#### Catching signal and Parsing exit status

​	The `status` variable is passed by refrence to the `my_wait` function. Its value at the end of function execution contains information about the child process. However, the macros used in task 1 cannot be used in kernel space. Therefore, in program2.c, I copied these macros so that they can be readily applied to the analyze the status. The macros used are:

- __WEXITSTATUS
- \__WTERMSIG
- \__WSTOPSIG
- \__WIFEXITED
- \__WIFSIGNALED
- \__WIFSTOPPED

​	For semantic output, as I did in task 1, I used the `getsig` to output the signal name as string

### Bonus

​	The bonus program asked us to implement the `pstree` command in linux system. 

#### Key data structure

##### N-ary tree	

An n-ary tree is used as the data structure. Each node of the tree is defined as a struct:

```c++
struct proc_node{
	proc_info info;

	proc_node *parent;
  proc_node *first_child;
  proc_node *next_sibling;
} ;
```

The `proc_info` field contains information abut the process. The `parent` is a pointer pointing to the process's parent process. The `first_child` is a pointer to the first child process node, and any sibling process can be get by traversing the `next_sibling` pointer.

The proc_info is also a struct:

```c++
struct proc_info{
	pid_t pid;
	pid_t ppid;
	std::string name;
  std::string cmdline;
};
```

#### Map

​	To guaranteen quick access, a <int, proc_node*> map is created. The address of a given node can be accessed in constant time complexity.

#### Basic Information

​	In linux system, the `/proc` folder in Linux contains information about the current state of kernel. The folder contains nunmbered folders, e.g: */proc/2*, which contains information about a process (in this example, process with PID=2). The 'status' file in this folder contains the PID, PPID, name of this process.

#### Workflow

1. Parse the arguments and config the output stype
2. Scan the **/proc** directory, use regex to match all process folder
3. For each process foder, if it's not in the map, put it in the map
   1. Access the /proc/PID/status file and the /proc/PID/cmdline
   2. Acquire the PID, PPID, Name from the file
   3. Create a node using the above-mentioned information
   4. Traverse using PPID until a parent node is already present in the map
4. After step 2, the tree is created.
5. Print the tree out using Depth-first-search

#### Arguments I implemented

- -V: print out the verbose information about pstree
- -p: print out the PID of each process
- -c: print out the commandline of each process
- -s: show the parent process

## Learning Outcome

​	Both task 1 and task 2 allows me to gain hand-on experience in C programming. They allow me to understand how Linux processes are created both from the user space and the kernel space. Moreover, I learned what it mean to 'execute' a binary file.

​	After this assgnment, I learned how to write a basic linux kernel module and insert it into the running kernel using LKM. I also learned to patiently read the source code of Linux Kernel and understand the APIs(where a symbol is defined and refrenced etc.).

## Appendix

### Program output for Task 1

<img src="/Users/shitianhao/Library/Application Support/typora-user-images/image-20221009005237381.png" style="zoom:50%;" />

<img src="/Users/shitianhao/Library/Application Support/typora-user-images/image-20221009005240632.png" alt="image-20221009005240632" style="zoom:50%;" />

<img src="/Users/shitianhao/Library/Application Support/typora-user-images/image-20221009005253931.png" alt="image-20221009005253931" style="zoom:50%;" />

<img src="/Users/shitianhao/Library/Application Support/typora-user-images/image-20221009005302079.png" alt="image-20221009005302079" style="zoom:50%;" />

<img src="/Users/shitianhao/Library/Application Support/typora-user-images/image-20221009005310937.png" alt="image-20221009005310937" style="zoom:50%;" />

### Program output for Task 2

![image-20221009224427475](/Users/shitianhao/Library/Application Support/typora-user-images/image-20221009224427475.png)

![image-20221009224722749](/Users/shitianhao/Library/Application Support/typora-user-images/image-20221009224722749.png)

![image-20221009225055413](/Users/shitianhao/Library/Application Support/typora-user-images/image-20221009225055413.png)

![image-20221009225253925](/Users/shitianhao/Library/Application Support/typora-user-images/image-20221009225253925.png)



### Modified Makefile for batch running Task 1

```makefile
CFILES:= $(shell ls|grep .c)
PROGS:=$(patsubst %.c,%,$(CFILES))
all: $(PROGS)

%:%.c
	$(CC) -o $@ $<

clean:$(PROGS)
	rm $(PROGS)

run: $(PROGS)
	@for testcase in $(PROGS) ; do \
        if [ $$testcase != "program1" ]; then \
			echo "************Performing Testcase For $$testcase ************\n"; \
			./program1 $$testcase; \
			echo "\n"; \
		fi \
	done
```

### Bonus program output

![image-20221010235613486](/Users/shitianhao/Library/Application Support/typora-user-images/image-20221010235613486.png)
