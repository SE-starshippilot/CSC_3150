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
struct Node{
	int row , col; 
	Node( int _row , int _col ) : row( _row ) , col( _col ) {}; 
	Node(){} ; 
} frog ; 

/*
 +--------> COLUMN++ (col++)
 |
 |
 v
 ROW++
 (row++) 
*/
char map[ROW][COLUMN] ; 
int status=1;
int test_log=20;
// std::vector<int> logs_head(9);
// Determine a keyboard is hit or not. (If yes, return 1. If not, return 0. )
int kbhit(void){
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

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void draw_map(){
	/*  Draw the map  */
	int i, j;
	system("clear");
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN ; ++j )	
			map[i][j] = ' ' ;  
	} //canvas
	for( j = 0; j < COLUMN; ++j ){
		map[ROW][j] = map[0][j] ='|' ;
	}// Upper bank and lower bank
	printf("Frog's location: (%d,%d)\n", frog.row, frog.col);
}


void *logs_move(){
	while (status==1){
		pthread_mutex_lock(&eventmutex);
		draw_map();
		pthread_mutex_unlock(&eventmutex);
		usleep(100000);
		
	}
	if (pthread_mutex_trylock(&eventmutex)==0) pthread_mutex_unlock(&eventmutex);
	printf("logs_move thread exit\n");
	pthread_exit(NULL);
	/*  Move the logs  */
	// std::vector<int> log_length(ROW-2,LOG_LENGTH);
	/*  Check keyboard hits, to change frog's position or quit the game. */

	
	/*  Check game's status  */
	// pthread_exit(NULL);
}

void *listen_keyboard( void *t ){
	while (status == 1){
		pthread_mutex_lock(&eventmutex);
		if (kbhit()){
			bool moved = false;
			char c = (char) getchar();
			if (c=='w' || c=='W'){
				frog.row--;
				moved = true;
			}
			else if (c=='s' || c=='S'){
				frog.row++; // make sure frog is not below the bottom row
				moved = true;
			}
			else if (c=='a' || c=='A'){
				frog.col--;
				moved = true;
			}
			else if (c=='d' || c=='D'){
				frog.col++;
				moved = true;
			}
			else if (c=='q' || c=='Q'){
				status=3; // quit
				pthread_mutex_unlock(&eventmutex);	
			}
			if (moved) draw_map();
		}
		pthread_mutex_unlock(&eventmutex);	
	}
	pthread_exit(NULL);
}

int main( int argc, char *argv[] ){
	pthread_t /*logs_thread,*/ kb_thread;
	memset(map, 0, sizeof(map));
	frog = Node( ROW, (COLUMN-1) / 2 ) ;// Frog initially at the lower bank, in the middle.
	draw_map();
	pthread_mutex_init(&eventmutex, NULL);
	pthread_create(&kb_thread, NULL, listen_keyboard, NULL);
	pthread_join(kb_thread, NULL);
	system("clear");
	switch (status){
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
	printf("main thread exit\n");
	pthread_mutex_destroy(&eventmutex);
	pthread_exit(NULL);
	return 0;
}
