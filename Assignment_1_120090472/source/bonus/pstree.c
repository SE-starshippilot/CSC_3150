#define PROCDIR "/proc"

#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

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