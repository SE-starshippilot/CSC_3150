#define PROCDIR "/proc"
#define MAX_NAME_LENGTH 128

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <sys/types.h>
#include <string>
#include <iostream>

#include <map>
typedef struct {
	pid_t pid;
	pid_t ppid;
	std::string name;

	proc_node *parent;
    proc_node *first_child;
    proc_node *next_sibling;
} proc_node;

void compileRegex(regex_t *regex);

void createProcNode(std::map<int, proc_node*>* p_map, int pid);
