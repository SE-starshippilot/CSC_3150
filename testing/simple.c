#include <stdio.h>

int sum(int a, int b)
{
	return a + b;
}

int main(int argc, char *argv[])
{
	int a = 2;
	int b = 2;
	int c = sum(a, b);
	printf("sum(%d, %d)=%d\n", a, b, c);
	return 0;
}