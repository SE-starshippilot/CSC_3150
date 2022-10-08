#define PROCDIR "/proc"

#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

typedef struct {
    pid_t pid;
    uid_t uid;

    pid_t ppid;
    char name[256];
} proc_t;


int main(int argc, char *argv[])
{
    /* parse args*/
    for (int i=1; i<argc; i++) {
        printf("argument %d is %s\n", i,  argv[i]);
    }
    DIR *dir;
    dir = opendir(PROCDIR);
    if (dir == NULL) {
        perror("opendir");
        return 1;
    }
    closedir(dir);

    pid_t pid = 0;
    uid_t uid = 0;
    
	return 0;
}