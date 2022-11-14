# CSC 3150 Assignment 3 Report

Tianhao Shi, 120090472

Nov, 2022

****

## Environment

![Screenshot 2022-11-09 at 05.47.30](/Users/shitianhao/Desktop/Screenshot 2022-11-09 at 05.47.30.png)

- CUDA Version: 11.6

  <img src="/Users/shitianhao/Desktop/Screenshot 2022-11-09 at 20.18.09.png" alt="Screenshot 2022-11-09 at 20.18.09" style="zoom:25%;" />

  Also tested on the school's cluster

  

## Execution

### Method 1: Use makefile

```sh
cd Assignment_3_120090472/source
make #(compile+run)
# make build #(build only)
```

### Method 2: Batch file

```sh
cd Assignment_3_120090472/source
bash ./slurm.sh
```



## Design of Key Components

### Inverted Page Table

A page table is a one-to-one mapping betwen the virtual address and the physical address. In an inverted table, the index is the physical address (frames) and the virtual page id is stored as elements of the table. During address translation, the CPU issues a virtual address. The address is divided into two parts, **page id** and **page offset**, using the following rule:

​	For a virtual address with *n* bits, if the page table entry is of size *2^m^*, the upper *m* bits are used as page id, and the rest *n-m* bits are used as page offset.

<img src="/Users/shitianhao/Library/Application Support/typora-user-images/image-20221109083700296.png" alt="image-20221109083700296" style="zoom:50%;" />

 

### LRU

LRU, or "Least Recently Used", is a page replacement policy that OS use to decide which page to swap out in the inverted page table. This policy always choose, as the name suggests, the least recently used page to swap out of RAM (and page table). In my design, the underlying data structure is a **doubly linked list**. I store two pointers in the VirtualMemory struct, which tracks the *head* and *tail* of the linked list. The *head* points to the newest item in queue, while the tail points to the oldest. The update of LRU is not directly relevant to page hit/page miss. Instead, LRU is update when:

1. Page hit, but the requested page is not the latest used page.
2. Page miss, which means the page is previously not in the LRU list.

However, there are minor differences between the two scenarios. Namely, in the page hit scenario,all we need to do is traverse the LRU list and **perform a linear search**. The found node is moved to the head of the linked list.

A page miss indicates the requested page is not in the RAM. Therefore, there is no need to search. We can simply **append the node to the head** and update the list size. Notice that there is a limit for LRU size. When overflow occurs, the extraneous node is the LRU node and is removed from the linked list. 

### Swap table

Our virtual memory size is 160KB, but the RAM size is only 32KB. This means for any input larger than the RAM size, there must be some data that resides in disk. To find the physical address of a page in disk, a swap table is used. The structure of swap table is similar to the inverted page table. The disk address is also partitioned into 32 byte pages, giving the swap table 4096 entries. Whenever swapping happens, the page id of the queried virtual address is searched in the swap table. 

## Program Flow

![Program Flow.drawio](/Users/shitianhao/Library/CloudStorage/OneDrive-CUHK-Shenzhen/Year 3 Term 1/CSC 3150/Assignment-3/Program Flow.drawio.png)

The `vm_write` and `vm_read` function are almost identical. Here, I use `vm_write` as an example to illustrate the flow of my program.

For any virtual address, it is pointing to one of the following places:

1. A physical address in RAM
2. A physical address in Disk
3. Unallocated

If a virtual address is pointing to RAM, then the page_id must be stored in the inverted page table. Therefore, if the return value of querying page table is not -1, then we can construct the physical address using the corresponding frame_id.

If the page_id is not stored in RAM, **a page fault occurs**, and we must increment the page_fault counter.

If the page table is not full, then we can always insert the page to the first vacant place (there is no point swapping if we can store data in RAM).

However, if the page table is full, we **must move victim a page from RAM to disk**. The selection of this 'victim' is conducted using the previously mentioned **LRU** algorithm. If the swap table contains the requested page_id, then **the desired page is moved from disk to RAM**.



`vm_read` is almost identical except it reads from the final physical address.

## Bonus Design

Due to the limitted time, I implemented the bonus using the second version described in the additional note. The code is almost identical compared to the code in the first task.

However, notibaly the following things are added:

1. In `main.cu`, we need to launch four threads. Therefore, we call `mykernel <<<1, 4, INVERT_PAGE_TABLE_SIZE >>> (input_size);`
2. It seems resolving race condition is not very straightforward (hard to implement `mutex` used in assignment 2). However, the CUDA library do provide a `__syncthreads()` function. This function serves as a barrier, blocking threads until every thread reches this function. Using this function, the threads are executed consequtively, in the order of 0->1->2->3. Each time, the vm is initialized. However, the page fault pointer is not cleaned and is continuously updating, allowing the program to count the overall fault number.![image-20221109185829336](/Users/shitianhao/Library/Application Support/typora-user-images/image-20221109185829336.png)

## Output and Analysis:

**For testcase 1, the output of my program is 8193:**

![Screenshot 2022-11-09 at 06.04.44](/Users/shitianhao/Library/Application Support/typora-user-images/Screenshot 2022-11-09 at 06.04.44.png)

snapshot output

![Screenshot 2022-11-09 at 09.48.02](/Users/shitianhao/Desktop/Screenshot 2022-11-09 at 09.48.02.png)

Cmp command:

![Screenshot 2022-11-09 at 09.49.27](/Users/shitianhao/Desktop/Screenshot 2022-11-09 at 09.49.27.png)

Analysis:

​	The user program of testcase 1 does the following things:

		1. Write [0, 132k) of the data.bin to VM starting at address 0 in ascending order.
		1. Read virtual address [132k-1, 96k-1] 
		1. Take a snapshot of the virtual memory with offset 0

During these procedures, the first step contributes 132k/32=4k page faults, as they are all new pages. The second step only contributes to 1 page fault, because it reads all the content stored in the RAM (page hit), plus one element in the disk. The final step also contributes 4k page faults, because it reads from the beginning of VM. This means the pages loaded in step 2 are all replaced starting from address 0.

Threfore, totally we have $8\times2^{10}+1=8193$ page faults.

A visualization is as followed

**For testcase 2, the output of my program is 9125**

![Screenshot 2022-11-09 at 07.14.34](/Users/shitianhao/Desktop/Screenshot 2022-11-09 at 07.14.34.png)

Analysis:

​	In this program the following things are done:

	1. Write [0, 132k) of the data.bin to VM starting at address 32k in ascending order
	1. Write [32k, 64k-1] of the data.bin to VM stasrting at address 0 in ascending order
	1. Take a snapshot of the virtual memory with offset 32k

The first step populates the VM with 132K data, resulting in 4k page faults. The second step causes 1k-1 page faults, and the snapshot causes 4k page faults. Therefore, in total we have $9*2^{10}-1=9215$ page faults.

**Bonus**

Only testcase 1 is used to test the bonus case. Since my code can be view as reading the .bin file four times, the total sum is 4*8193=32772 times.![Screenshot 2022-11-09 at 20.15.39](/Users/shitianhao/Library/Application Support/typora-user-images/Screenshot 2022-11-09 at 20.15.39.png)

## Problems Encountered & Solutions

### Implementation of LRU

At first I tried to use array as the underlying datastructure for implementing LRU. Because the vm_init function seems to suggest us to use the second 1k of inverted page table as the LRU array. This idea sounds easy, as the top element is the most recently used one. However, whenever we need to update the LRU array, we always need O(N) to move elements one by one. Moreover, when I tried to debug my programs, it seems that CUDA-GDB is having some trouble displaying value of the members' values of  `struct VirtualMemory`. In the end I choose to replace the data structure with doubly linked list, which allowed the code to do insertion in O(1) time.

### Debugging Mechanism

Considering the large input size and heterogeneous nature of CUDA, finding an efficient debugging program is very difficult. I came up with three ways to debug my program:

1. Create the `tasks.json` and `launch.json` by referring to VSCode and Nsight studio doccumentation. I uploaded these two files on [github](https://github.com/SE-starshippilot/CSC3150-Cuda-Debug.git). This method allows me to debug program using VSCode GUI in a step-wise manner. Moreover, at anytime I can examine the value of expressions to get accurate information. However, this method has two serious drawbacks. First, although VSCode attaches the program to the Cuda-GDB, the running speed of program is significantly slower than other debugging methods. Second, I find that the content of `printf` is not updated immediantly. I later found out that, according to [this post]([How to print from CUDA : 15-418 Spring 2013 (cmu.edu)](http://15418.courses.cs.cmu.edu/spring2013/article/15)) this is a feature of CUDA and there is nothing I can do about it. Therefore I also use the following methods to facilitate the debugging procedure.

2. Using CUDA-GDB directly from command line. Although I am pretty new to GDB debugging and I don't know many of the commands and features, CUDA-GDB allows me to locate bugs more specifically (e.g, if SEGFAULT happens, I know which line triggers this).

3. Use rich `printf` in source code and run the executable file from the command line. Redirect stdout and stderr to a file and look for desired information in the file.

   <img src="/Users/shitianhao/Library/Application Support/typora-user-images/Screenshot 2022-11-09 at 06.55.40.png" alt="Screenshot 2022-11-09 at 06.55.40" style="zoom:33%;" />

## Learning Outcome

This process allowed me to get a more thorugh  and in-depth understanding of the paging policy, as well as the implementation and usage of virtual memory. It also allows me to gain some first hand-on experience on using CUDA. It is a very challenging project in terms of:

1. I have to come up how to implement a flawless paging policy from scratch
2. The input size of the binary file manes it hard to expose issues
3. Previously I have little experience in using GDB debugger (now I know how to use some of the basic features using CLI commands)
4. The CUDA-GDB itself has bugs, causing me to rely heavily on printf statements (on the bright side, I learned how to search for useful information in the log)

However, when I saw that I am able to handle 13000+ data without any mistake, I feel very proud of myself.

### Miscellaneous

I kept all the `printf` commands in my code, so it may print out a lot of information. Also, I included the outut.txt as a proof that my program DO run (hard to fake a file with ten thousad lines).

