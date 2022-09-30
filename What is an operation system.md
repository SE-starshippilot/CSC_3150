What is an operation system

- Referee
  - Manage protection, isolation, sharing of resources
- Illusionsit
  - Provide lean, easy-to-use abstraction of physial resource
    - infinite memory
    - Higher level of objects (files, uesers, messages)
    - Masking limitations
- Glue





printf, provided by libc, declaired in stdio

printf is a wrapper for some syscall

​	syscall: user mode -> kernel mode by interrupt 

​	

four fundamental OS Concepts:

process -> thread -> stack



Thread: Execution Content

- PC, Reg, Execution Flags, Stack
- TCB (Thread Control Block), saves PC, Reg, Execution Flag in the memory （preserved for OS)
- Context Switcher: reserve TCB

Address Spae

- the set of memory address accessible to program (virtual memory)

Process

- protected address space + some threads

Dual mode operation

- user mode vs kernel mode
- 

​	