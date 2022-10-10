#include "pstree.h"

proc_info getProcInfo(std::string proc){
	if (proc == "0"){
		return proc_info{0, 0, "swapper"};
	}
	std::string proc_status_path = PROCDIR + std::string("/") + proc + std::string("/status");
    std::ifstream  proc_status_file;
	proc_info _info;
	std::string delim = ":";
    proc_status_file.open(proc_status_path);
    if (!proc_status_file.is_open()){
        std::cout << "Error opening file"<<std::endl;
        exit(EXIT_FAILURE);
    } else {
        std::string line;
        while (std::getline(proc_status_file, line)){
			std::string attr = line.substr(0, line.find(delim));
			if (attr == "Pid"){
				_info.pid = std::stoi(line.substr(line.find(delim) + 1));
			} else if (attr == "PPid"){
				_info.ppid = std::stoi(line.substr(line.find(delim) + 1));
			} else if (attr == "Name"){
				_info.name = line.substr(line.find(delim) + 1);
				trim(&_info.name);
			}
		}
		proc_status_file.close();
		return _info;
	}
}

proc_info recursiveInsertNode(std::map<int, proc_node *> *p_map, proc_info info){
	if (info.pid == 0){
		// std::cout << "swapper" << std::endl;
		return info;
	} else {
		bool new_flag = false;
		if (p_map->find(info.pid) == p_map->end()){
			createProcNode(p_map, info);
			new_flag = true;
		}
		proc_info parent_info = getProcInfo(std::to_string(info.ppid));
		// std::cout << info.name << std::string("->");
		if (p_map->find(parent_info.pid) == p_map->end()){
			createProcNode(p_map, parent_info);
			new_flag = true;
		}
		if (new_flag) insertProcNode(p_map, info.ppid, info.pid);
		return recursiveInsertNode(p_map, parent_info);
	}
}

void createProcNode(std::map<int, proc_node *> *p_map, proc_info info)
{
	proc_node *temp_node = new proc_node{info, nullptr, nullptr, nullptr};
	p_map->insert(std::pair<int, proc_node *>(info.pid, temp_node));
}

void insertProcNode(std::map<int, proc_node *> *p_map, int ppid, int pid)
{
	proc_node *parent = (*p_map)[ppid];
	proc_node *child = (*p_map)[pid];

	child->parent = parent;

	if (parent->first_child) {
		proc_node *_sibling = parent->first_child;
		while (_sibling->next_sibling){
			_sibling = _sibling->next_sibling;
		}
		_sibling->next_sibling = child;
	} else {
		parent->first_child = child;
	}
}

void traverseTree(proc_node *node, int level)
{
	if (node) {
		for (int i = 0; i < level-1; i++) {
			std::cout << "\t";
		}
		// if (level != 0) std::cout << "|-";
		std::cout << node->info.name ;
		if (node->first_child) {
			std::cout << std::string("-");
			if (node->first_child->next_sibling) {
				std::cout << std::string("+-");
			} else {
				std::cout << std::string("--");
			}
			traverseTree(node->first_child, level+1);
		}
		if (node->next_sibling){
			
		}
		traverseTree(node->next_sibling, level);
	}
}

static inline void trim(std::string *s){
	std::stringstream ss(*s);
	ss >> *s;
}

int main(int argc, char *argv[])
{
	DIR *dir;
	struct dirent *sd;
	int reg_comp_status;
	int val;
	int proc_num = 0;
	bool show_pid = false;
	/*parse options*/
	while ((val = getopt(argc, argv, OPSTRING)) != -1) {
		switch (val) {
		case 'h':
			std::cout << "Usage: pstree [-h] [pid | name | -]\n" << std::endl;
			return 0;
		case 'V':
			std::cout << verbose << std::endl;
			return 0;
		case 'p':
			show_pid = true;
			break;
		default:
			std:: cout <<"Usage: pstree [-h] [pid | name | -]\n" << std::endl;
			return 1;
		}
	}

	/*create swapper process*/
	std::map<int, proc_node *> proc_map;
	proc_info swapper_info={0, 0, "swapper"};
	proc_node *swapper = new proc_node{swapper_info, nullptr, nullptr, nullptr};
	proc_map.insert(std::pair<int, proc_node *>(0, swapper));
	/*open /proc directory*/
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
			int curr_pid = atoi(sd->d_name);
			if (proc_map.find(curr_pid) == proc_map.end()){
				recursiveInsertNode(&proc_map, getProcInfo(sd->d_name));
			}
		}
	}
	printf("Total number of process:%d\n", proc_num);
	traverseTree(swapper, 0);
	closedir(dir);
	return 0;
}