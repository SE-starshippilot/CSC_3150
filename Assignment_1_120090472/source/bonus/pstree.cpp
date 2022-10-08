#include "pstree.h"

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

int getProcInfo(std::string proc){
	std::string proc_status_path = PROCDIR + std::string("/") + proc + std::string("/status");
	std::cout << proc_status_path << std::endl;
    // std::ifstream  proc_status_file;
    // proc_status_file.open(proc_status_path);
    // if (!proc_status_file.is_open()){
    //     std::cout << "Error opening file";
    //     return 1;
    // } else {
    //     std::string line;
    //     while (std::getline(proc_status_file, line)){
    //         std::cout << line << std::endl;
    //     }
    // }
    return 0;
}

void createProcNode(std::map<int, proc_node *> *p_map, int pid)
{
	proc_node *temp = new proc_node{ (pid_t)pid, 0,	      "test",
					 nullptr,    nullptr, nullptr };
	p_map->insert(std::pair<int, proc_node *>(pid, temp));
}

void insertProcNode(std::map<int, proc_node *> *p_map, int ppid, int pid)
{
	proc_node *parent = (*p_map)[ppid];
	proc_node *child = (*p_map)[pid];

	child->parent = parent;
	child->ppid = parent->pid;

	if (parent->first_child) {
		proc_node *_sibling = parent->first_child;
		do {
			_sibling = _sibling->next_sibling;
		} while (_sibling->next_sibling);
		_sibling->next_sibling = child;
	} else {
		parent->first_child = child;
	}
}

int main(int argc, char *argv[])
{
	DIR *dir;
	struct dirent *sd;
	int reg_comp_status;
	int val;
	int proc_num = 0;

	/*parse options*/
	while ((val = getopt(argc, argv, OPSTRING)) != -1) {
		switch (val) {
		case 'h':
			std::cout << "Usage: pstree [-h] [pid | name | -]\n" << std::endl;
			return 0;
		case 'V':
			std::cout << verbose << std::endl;
			return 0;
		default:
			std:: cout <<"Usage: pstree [-h] [pid | name | -]\n" << std::endl;
			return 1;
		}
	}

	std::map<int, proc_node *> proc_map;
	dir = opendir(PROCDIR);
	if (dir == NULL) {
		perror("opendir");
		return 1;
	}
	while ((sd = readdir(dir)) != NULL) {
		if (sd->d_type == DT_DIR &&
		    std::regex_match(sd->d_name,
				     std::regex(PROC_FOLDER_PATTERN))) {
			proc_num++;
			int info = getProcInfo(sd->d_name);
			int pid = atoi(sd->d_name);
			createProcNode(&proc_map, pid);
			if (proc_map.find(pid) != proc_map.end()){
			} else {
				int ppid = proc_map[pid]->ppid;
				insertProcNode(&proc_map, pid, pid);
			}
			printf("%s\n", sd->d_name);
		}
	}
	printf("Total number of process:%d\n", proc_num);
	closedir(dir);
	return 0;
}