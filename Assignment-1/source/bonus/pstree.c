#define PROCDIR "/proc"
#define MAX_CHILD_PROC 256

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <regex.h>
#include <sys/types.h>

typedef struct node{
	pid_t pid;
	pid_t ppid;
	char name[256];
	struct node *parent;
	struct node *childrens[MAX_CHILD_PROC];
} proc_node;

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

typedef struct {
    pid_t pid;
    uid_t uid;

    pid_t ppid;
    char name[256];
} proc_t;


int main(int argc, char *argv[])
{
    DIR *dir;
    dir = opendir(PROCDIR);
    if (dir == NULL) {
        perror("opendir");
        return 1;
    }
    closedir(dir);
	return 0;
}