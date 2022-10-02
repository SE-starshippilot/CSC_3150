#include <stdlib.h>
#include <unistd.h>

int main() {
  int pid = fork();
  if (pid == 0) {
    exit(0);
  }
  return 0;
}
