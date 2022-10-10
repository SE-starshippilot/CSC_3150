#define PROCDIR "/proc"
#define MAX_NAME_LENGTH 128
#define PROC_FOLDER_PATTERN "^[0-9]+$"
#define OPSTRING "hpV"

#include <dirent.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <regex>
#include <sys/types.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <map>

std::regex reg(PROC_FOLDER_PATTERN);
const char* verbose = 
"pstree (PSmisc) version 0.0.1\n"
"Copyright (C) 2022 Tianhao SHI\n"
"\n"
"pstree comes with ABSOLUTELY NO WARRANTY.\n"
"This is free software, and you are welcome to redistribute it under\n"
"the terms of the GNU General Public License.\n";

struct proc_info{
	pid_t pid;
	pid_t ppid;
	std::string name;
	std::string cmdline;
};

struct proc_node{
	proc_info info;

	proc_node *parent;
    proc_node *first_child;
    proc_node *next_sibling;
} ;

void createProcNode(std::map<int, proc_node *> *p_map, proc_info info);

static inline void trim(std::string *s);

proc_info recursiveProcInfo(std::map<int, proc_node *> *p_map, proc_info info);

void insertProcNode(std::map<int, proc_node *> *p_map, int ppid, int pid);

void traverseTree(proc_node *node, int level, std::string prefix);

bool show_pid;