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
#define LOG_SPEED 200000
#define log_right(log_left) (log_left + LOG_LENGTH - 1) % COLUMN


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
char map[ROW + 1][COLUMN];
int logs_left_arr[ROW - 1];
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
		if (log_right(logs_left_arr[i - 1]) < logs_left_arr[i - 1]) {
			for (int j = 0; j <= log_right(logs_left_arr[i - 1]); ++j)
				map[i][j] = '=';
			for (int j = logs_left_arr[i - 1]; j < COLUMN; ++j)
				map[i][j] = '=';
		}
		else {
			for (int j = logs_left_arr[i - 1]; j <= log_right(logs_left_arr[i - 1]); ++j)
				map[i][j] = '=';
		}
	}// Logs

	map[frog.row][frog.col] = '0';//frog

	for (int i = 0; i <= ROW; ++i) {
		for (int j = 0; j < COLUMN; ++j)
			printf("%c", map[i][j]);
		printf("\n");
	}
}

void* logs_move(void*) {
	while (status == 1) {
		pthread_mutex_lock(&eventmutex);
		/* move the logs by 1 column*/
		for (int i = 0; i < ROW; ++i) {
			if (i % 2) {
				logs_left_arr[i]--;
				if (logs_left_arr[i] < 0) logs_left_arr[i] = COLUMN - 1;
			}
			else {
				logs_left_arr[i]++;
				if (logs_left_arr[i] == COLUMN) logs_left_arr[i] = 0;
			}
		}

		/* frog will move with log by 1 column if it's on one*/
		if (frog.row % 2)
			frog.col++;
		else if (frog.row != 0 && frog.row != ROW)
			frog.col--;
		draw_map();
		pthread_mutex_unlock(&eventmutex);
		usleep(LOG_SPEED);
	}
	pthread_exit(NULL);
}

void* frog_move(void*) {
	while (1) {
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
				break;
				pthread_mutex_unlock(&eventmutex);
			}

			/* judge game status */
			update_game_status();
			if (status != 1) {
				pthread_mutex_unlock(&eventmutex);
				break;
			}

			/* draw map */
			if (moved) draw_map();

		}
		pthread_mutex_unlock(&eventmutex);
	}
	pthread_exit(NULL);
}

void initialize_logs() {
	for (int i = 0; i < ROW - 1; ++i) {
		logs_left_arr[i] = rand() % (COLUMN);
	}
}

void update_game_status() {
	if (frog.row == 0) {
		status = 2;
	}
	else if (frog.row < ROW) {
		if (frog.col == 0 || frog.col == COLUMN - 1) {
			status = 0;
		} // hit the border
		if ((logs_left_arr[frog.row - 1] < log_right(logs_left_arr[frog.row - 1]) && (frog.col < logs_left_arr[frog.row - 1] || frog.col > log_right(logs_left_arr[frog.row - 1])))
			|| (logs_left_arr[frog.row - 1] > log_right(logs_left_arr[frog.row - 1]) && (frog.col < logs_left_arr[frog.row - 1] && frog.col > log_right(logs_left_arr[frog.row - 1])))) {
			status = 0;
		} // fall into the water
	}
}


int main(int argc, char* argv[]) {
	pthread_t logs_thread, frog_thread;
	memset(map, ' ', sizeof(map));
	memset(logs_left_arr, 0, sizeof(logs_left_arr));
	frog = Node(ROW, (COLUMN - 1) / 2);// Frog initially at the lower bank, in the middle.
	initialize_logs();
	draw_map();
	pthread_mutex_init(&eventmutex, NULL);
	pthread_create(&logs_thread, NULL, logs_move, NULL);
	pthread_create(&frog_thread, NULL, frog_move, NULL);
	pthread_join(frog_thread, NULL);
	pthread_join(logs_thread, NULL);
	system("clear");
	switch (status) {
	case 0:
		std::cout << "You lose the game!" << std::endl;
		break;
	case 2:
		std::cout << "You win the game!" << std::endl;
		break;
	case 3:
		std::cout << "You exit the game!" << std::endl;
		break;
	}
	pthread_mutex_destroy(&eventmutex);
	return 0;
}
