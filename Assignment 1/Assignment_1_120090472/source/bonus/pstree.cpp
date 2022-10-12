#include "pstree.h"

proc_info getProcInfo(std::string proc){
	if (proc == "0"){
		return proc_info{0, 0, "swapper"};
	}
	std::string proc_status_path = PROCDIR + std::string("/") + proc + std::string("/status");
	std::string proc_cmdline_path = PROCDIR + std::string("/") + proc + std::string("/cmdline");
    std::ifstream  proc_status_file, proc_cmdline_file;
	proc_info _info;
	std::string delim = ":";
    proc_status_file.open(proc_status_path);
	proc_cmdline_file.open(proc_cmdline_path);
	if (!proc_status_file.is_open()){
		std::cerr << "Error: cannot open " << proc_status_path << std::endl;
		exit(EXIT_FAILURE);
	} else {
		std::string line;
		std::getline(proc_cmdline_file, line);
		_info.cmdline = line;
	}
	proc_cmdline_file.close();
    if (!proc_status_file.is_open()){
        std::cerr << "Error opening file"<<std::endl;
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

void printTree(proc_node *node, int level, std::string prefix)
{
	if (node) {
		std::string name;
		name = node->info.name;
		if (show_pid){
			name = node->info.name + std::string("(") + std::to_string(node->info.pid) + std::string(")");
		} 
		std::string blank(name.length()+1, ' ');
		std::string n_prefix;
		if (node->first_child) {
			std::cout << name  << std::string("-");
			if (node->first_child->next_sibling) {
				std::cout << std::string("+-");
				proc_node *tmp = node->first_child->first_child;
				bool flag = false;
				while(tmp){
					if (tmp->next_sibling){
						flag = true;
						break;
					}
					tmp = tmp->first_child;
				}
				if(flag){
					prefix = blank + std::string("| ");
				} else {
					prefix = blank;
				}
			} else {
				std::cout << std::string("--");
			}
			printTree(node->first_child, level+1, prefix);
		} else {
			std::cout << name <<std::endl;
		}
		if (node->next_sibling){
			std::cout << prefix;
			if (node->next_sibling->next_sibling){
				std::cout << std::string("|-");	
			} else {
				std::cout << std::string("`-");	
			}
			printTree(node->next_sibling, level, prefix);
		} 
	}
}

static inline std::string createBlankString(int num){
	std::ostringstream oss;
	oss << std::setw(num);
	return oss.str();
}

static inline void trim(std::string *s){
	std::stringstream ss(*s);
	ss >> *s;
}

void createTestTree(proc_node *HEAD){
	HEAD = new proc_node{{0, 0, "Alpha", "test"}, nullptr, nullptr, nullptr};
	HEAD->first_child = new proc_node{{1, 0, "Beta", "test"}, HEAD, nullptr, nullptr};
	HEAD->first_child->first_child = new proc_node{{2, 1, "Gamma"}, HEAD->first_child, nullptr, nullptr};
	HEAD->first_child->first_child->next_sibling = new proc_node{{3, 1, "Delta", "test"}, HEAD->first_child, nullptr, nullptr};
	HEAD->first_child->first_child->next_sibling->next_sibling = new proc_node{{4, 1, "Epsilon", "test"}, HEAD->first_child, nullptr, nullptr};
	HEAD->first_child->next_sibling = new proc_node{{6, 0, "Zeta", "test"}, HEAD, nullptr, nullptr}; 
	printTree(HEAD, 0, "");
}

int main(int argc, char *argv[])
{
	DIR *dir;
	struct dirent *sd;
	int reg_comp_status;
	int val;
	int proc_num = 0;
	/*parse options*/
	show_pid = false;
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
	std::string curr_pid = argv[argc-1];
	proc_info swapper_info={0, 0, "swapper", "swapper"};
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

	/*draw tree*/
	proc_node *curr = proc_map[atoi(curr_pid.c_str())];
	printTree(curr, 0, std::string(""));
	closedir(dir);
	return 0;
}