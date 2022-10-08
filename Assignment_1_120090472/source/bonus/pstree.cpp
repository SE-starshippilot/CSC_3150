#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <sys/types.h>
#include <map>
#include <string>
#include <iostream>

#include "pstree.h"




void compileRegex(regex_t *regex){
    int reg_comp_status;
    char msgbuf[100];

    reg_comp_status = regcomp(regex, "^[0-9]+$", 0);
    if (reg_comp_status) {
        fprintf(stderr, "Could not compile regex!\n");
        exit(1);
    }
    return;
}

// int catalog_process(char *dirname)
// {
// 	char filename[256];
// 	char linebuf[256];
// 	char procname[256];
// 	char pid[32];
// 	char ppid[32];
// 	char *key;
// 	char *value;
// 	FILE *p_file;
// 	strcpy(filename, dirname);
// 	strcat(filename, "/status");
// 	p_file = fopen(filename, "r");
// 	if (p_file == NULL) {
//         fprintf(stderr, "Could not find status file in process folder!");
// 		return 1; /* just ignore, this is fine I guess */
// 	}
// 	while (fgets(linebuf, sizeof(linebuf), p_file) != NULL) {
// 		key = strtok(linebuf, ":");
// 		value = strtok(NULL, ":");
// 		if (key != NULL && value != NULL) {
// 			trim(key); trim(value);
// 			if (strcmp(key, "Pid") == 0) {
// 				strcpy(pid, value);
// 			} else if (strcmp(key, "PPid") == 0) {
// 				strcpy(ppid, value);
// 			} else if (strcmp(key, "Name") == 0) {
// 				strcpy(procname, value);
// 			}
// 		}
// 	}

// 	return ll_create_and_insert(&procname[0], atoi(pid), atoi(ppid));
// }

void createProcNode(std::map<int, proc_node*>* p_map, int pid){
    proc_node *temp = new proc_node{(pid_t) pid, 0, nullptr, nullptr};
    p_map->insert(std::pair<int, proc_node*>(pid, temp));
}

// void insertProcNode(std::map<int, proc_node>* p_map, int ppid, int pid){
//     proc_node parent = (*p_map)[ppid];
//     proc_node children = (*p_map)[pid];

// }

int main(int argc, char *argv[])
{
	DIR *dir;
	struct dirent *sd;
    regex_t regex;
    int reg_comp_status;
    int val;
	int proc_num = 0;
    std::map<int, proc_node*> proc_map;
    reg_comp_status = regcomp(&regex, "^[0-9]", 0);
    if (reg_comp_status) {
        fprintf(stderr, "Could not compile regex!\n");
        exit(1);
    }
	dir = opendir(PROCDIR);
	if (dir == NULL) {
		perror("opendir");
		return 1;
	}
	while ((sd = readdir(dir)) != NULL) {
		if (sd->d_type == DT_DIR && regexec(&regex, sd->d_name, 0, NULL, 0) == 0) {
			proc_num++;
            int pid = atoi(sd->d_name);
            createProcNode(&proc_map, pid);
            // if (proc_map.find(pid) != proc_map.end()){
                
            // }
			// printf("%s\n", sd->d_name);
		}
	}
	printf("Total number of process:%d\n", proc_num);
	closedir(dir);
	return 0;
}