#define PROCDIR "/proc"
#define MAX_CHILD_PROC 256
#include <map>
typedef struct node{
	pid_t pid;
	pid_t ppid;

	struct node *parent;
	struct node *childrens[MAX_CHILD_PROC];
} proc_node;

void createProcNode(std::map<int, proc_node*>* p_map, int pid);