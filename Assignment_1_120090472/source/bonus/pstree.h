#define PROCDIR "/proc"
#define MAX_NAME_LENGTH 128
#define PROC_FOLDER_PATTERN "^[0-9]+$"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <regex>
#include <sys/types.h>
#include <string>
#include <iostream>

#include <map>

std::regex reg(PROC_FOLDER_PATTERN);

struct proc_node{
	pid_t pid;
	pid_t ppid;
	std::string name;

	proc_node *parent;
    proc_node *first_child;
    proc_node *next_sibling;
} ;

void createProcNode(std::map<int, proc_node*>* p_map, int pid);
