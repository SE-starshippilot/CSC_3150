#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <random>

#include "hw2.h"

#define ROW 10
#define COLUMN 50 
#define LOG_LENGTH 15

pthread_mutex_t eventmutex;
pthread_cond_t eventcond;
struct Node {
	int row, col;
	Node(int _row, int _col) : row(_row), col(_col) {};
	Node() {};
} frog;

/*
 +--------> COLUMN++ (col++)
 |
 |
 v
 ROW++
 (row++)
*/
char map[ROW + 10][COLUMN];
int logs_head_arr[ROW];
int status = 1;
int test_log = 20;
// Determine a keyboard is hit or not. (If yes, return 1. If not, return 0. )
int kbhit(void) {
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if (ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void draw_map() {
	system("clear");
	for (int i = 1; i < ROW; ++i) {
		for (int j = 0; j < COLUMN; ++j)
			map[i][j] = ' ';
	} //canvas

	for (int j = 0; j < COLUMN; ++j) {
		map[ROW][j] = map[0][j] = '|';
	}// Upper bank and lower bank

	for (int i = 1; i < ROW; ++i) {
		if (i % 2) {
			if (logs_head_arr[i - 1] - LOG_LENGTH + 1 < 0) {
				for (int j = 0; j <= logs_head_arr[i - 1]; ++j)
					map[i][j] = '=';
				for (int j = logs_head_arr[i - 1] - LOG_LENGTH + 1 + COLUMN; j < COLUMN; ++j)
					map[i][j] = '=';
			}
			else {
				for (int j = logs_head_arr[i - 1] - LOG_LENGTH + 1; j <= logs_head_arr[i - 1]; ++j)
					map[i][j] = '=';
			}
		}
		else {
			if (logs_head_arr[i - 1] + LOG_LENGTH - 1 >= COLUMN) {
				for (int j = logs_head_arr[i - 1]; j < COLUMN; ++j)
					map[i][j] = '=';
				for (int j = 0; j <= logs_head_arr[i - 1] + LOG_LENGTH - 1 - COLUMN; ++j)
					map[i][j] = '=';
			}
			else {
				for (int j = logs_head_arr[i - 1]; j <= logs_head_arr[i - 1] + LOG_LENGTH - 1; ++j)
					map[i][j] = '=';
			}
		}
	}// Logs
	map[frog.row][frog.col] = '0';
	for (int i = 0; i <= ROW; ++i) {
		for (int j = 0; j < COLUMN; ++j)
			printf("%c", map[i][j]);
		printf("\n");
	}
}

void* logs_move(void* t) {
	while (status == 1) {
		pthread_mutex_lock(&eventmutex);
		/* move the logs by 1 column*/
		for (int i = 0; i < ROW; ++i) {
			if (i % 2) {
				logs_head_arr[i]--;
				if (logs_head_arr[i] < 0) logs_head_arr[i] = COLUMN - 1;
			}
			else {
				logs_head_arr[i]++;
				if (logs_head_arr[i] == COLUMN) logs_head_arr[i] = 0;
			}
		}

		/* frog will move with log by 1 column if it's on one*/
		if (frog.row % 2) {
			if (
				(logs_head_arr[frog.row-1] - LOG_LENGTH + 1 >= 0 &&
					frog.col >= logs_head_arr[frog.row-1] - LOG_LENGTH + 1 && frog.col <= logs_head_arr[frog.row-1])
				|| (logs_head_arr[frog.row-1] - LOG_LENGTH + 1 < 0 &&
					(frog.col >= logs_head_arr[frog.row-1] - LOG_LENGTH + 1 + COLUMN || frog.col <= logs_head_arr[frog.row-1]))
				) {
				frog.col++;
			}
			else {
				status = 0;
			}
		}
		else if (frog.row != 0 && frog.row != ROW) {
			if (
				(logs_head_arr[frog.row-1] + LOG_LENGTH - 1 < COLUMN &&
					frog.col >= logs_head_arr[frog.row-1] && frog.col <= logs_head_arr[frog.row-1] + LOG_LENGTH - 1)
				|| (logs_head_arr[frog.row-1] + LOG_LENGTH - 1 >= COLUMN &&
					(frog.col >= logs_head_arr[frog.row-1] || frog.col <= logs_head_arr[frog.row-1] + LOG_LENGTH - 1 - COLUMN))
				) {
				frog.col--;
			}
			else {
				status = 0;
			}
		}
		draw_map();
		pthread_mutex_unlock(&eventmutex);
		usleep(400000);
		/*  Move the logs  */
		/*  Check game's status  */
		// pthread_exit(NULL);
	}
	printf("logs_move thread exit\n");
	pthread_exit(NULL);
}
void* listen_keyboard(void* t) {
	while (status == 1) {
		pthread_mutex_lock(&eventmutex);
		if (kbhit()) {
			bool moved = false;
			char c = (char)getchar();
			if (c == 'w' || c == 'W') {
				frog.row--;
				moved = true;
			}
			else if (c == 's' || c == 'S') {
				if (frog.row != ROW)frog.row++; // make sure frog is not below the bottom row
				moved = true;
			}
			else if (c == 'a' || c == 'A') {
				frog.col--;
				moved = true;
			}
			else if (c == 'd' || c == 'D') {
				frog.col++;
				moved = true;
			}
			else if (c == 'q' || c == 'Q') {
				status = 3; // quit
				pthread_mutex_unlock(&eventmutex);
			}
			if (moved) draw_map();
			/* judge game status */
			if (!frog.row) status = 2; // win
			if (!(!frog.row || frog.row == ROW)				// the frog is between the banks
				&& (!frog.col || frog.col == COLUMN - 1)) 	/* and is at the border*/status = 0; // lose
		}
		pthread_mutex_unlock(&eventmutex);
	}
	pthread_exit(NULL);
}

void initialize_logs() {
	for (int i = 0; i < ROW - 1; ++i) {
		logs_head_arr[i] = rand() % (COLUMN);
	}
}

int main(int argc, char* argv[]) {
	pthread_t logs_thread, kb_thread;
	memset(map, ' ', sizeof(map));
	memset(logs_head_arr, 0, sizeof(logs_head_arr));
	frog = Node(ROW, (COLUMN - 1) / 2);// Frog initially at the lower bank, in the middle.
	initialize_logs();
	draw_map();
	pthread_mutex_init(&eventmutex, NULL);
	pthread_create(&logs_thread, NULL, logs_move, NULL);
	pthread_create(&kb_thread, NULL, listen_keyboard, NULL);
	pthread_join(kb_thread, NULL);
	pthread_join(logs_thread, NULL);
	system("clear");
	switch (status) {
	case 0:
		std::cout << "You lose!" << std::endl;
		break;
	case 2:
		std::cout << "You win!" << std::endl;
		break;
	case 3:
		std::cout << "You quit!" << std::endl;
		break;
	}
	pthread_mutex_destroy(&eventmutex);
	pthread_exit(NULL);
	return 0;
}
