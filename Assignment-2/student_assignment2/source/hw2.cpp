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

void *logs_move( void *t ){
	while (status==1){
		pthread_cond_broadcast(&eventcond);
		test_log = (test_log+1)%COLUMN;
		usleep(100000);
	}

	/*  Move the logs  */
	// std::vector<int> log_length(ROW-2,LOG_LENGTH);
	/*  Check keyboard hits, to change frog's position or quit the game. */

	
	/*  Check game's status  */
	// pthread_exit(NULL);
}

void *frog_move( void *t ){
	/*  Move the frog  */
	while (status==1){
		if (kbhit()){
			switch (getchar()){
				case 'w':
					if(frog.row) frog.row--;
					if(!frog.row) status = 0;//win
					break;
				case 's':
					if(frog.row < ROW ) frog.row++;
					break;
				case 'a':
					if(frog.col)frog.col--;
					if(!frog.col) status = 2;//lose, hit left border
					break;
				case 'd':
					if(frog.col < COLUMN ) frog.col++;
					if(frog.col == COLUMN ) status = 2;//lose, hit right border
					break;
				case 'q':
					status = 0;
					break;
			}
			pthread_cond_broadcast(&eventcond);
		}
	}
	pthread_exit(NULL);
}

void *draw_map(void *t){
	/*  Draw the map  */
	int i, j;
	while (status==1){
		pthread_cond_wait(&eventcond, &eventmutex);
		system("clear");
		for( i = 1; i < ROW; ++i ){	
			for( j = 0; j < COLUMN ; ++j )	
				map[i][j] = ' ' ;  
		} //canvas
		for( j = 0; j < COLUMN; ++j ){
			map[ROW][j] = map[0][j] ='|' ;
		}// Upper bank and lower bank
		if(test_log - LOG_LENGTH + 1< 0){
			for( j = 0; j <= test_log; ++j ){
				map[3][j]  ='=' ;
			}
			for ( j = (test_log - LOG_LENGTH - 1 + COLUMN) % 50; j < COLUMN; ++j){
				map[3][j] ='=' ;
			}
		}
		else{
			for( j = test_log - LOG_LENGTH+1; j <= test_log; ++j ){
				map[3][j] = '=' ;
			}
		}//test log
		map[frog.row][frog.col] = '0';//frog
		for( int i = 0; i <= ROW; ++i){	
			for (int j = 0; j < COLUMN; ++j){
			std::cout << map[i][j];
			}
			std::cout << std::endl;
		}
	}
	system("clear");
	if (status == 0){
		printf("You Win!\n");
	} else if (status == 2){
		printf("Game Over!\n");
	}
	pthread_exit(NULL);
}


int main( int argc, char *argv[] ){
	pthread_t logs_thread, frog_thread, kb_thread, display_thread;
	frog = Node( ROW, (COLUMN-1) / 2 ) ;// Frog initially at the lower bank, in the middle.
	for(int i = 1; i < ROW; ++i ){	
		for(int j = 0; j < COLUMN ; ++j )	
			map[i][j] = ' ' ;  
	} //ca
	for(int j = 0; j < COLUMN; ++j ){
		map[ROW][j] = map[0][j] ='|' ;// Upper bank and lower bank	
	}	
	map[frog.row][frog.col] = '0';

	pthread_mutex_init(&eventmutex, NULL);
	pthread_cond_init(&eventcond, NULL);
	pthread_create(&display_thread, NULL, draw_map, NULL);
	pthread_create(&logs_thread, NULL, logs_move, NULL);
	pthread_create(&frog_thread, NULL, frog_move, NULL);
	pthread_join(logs_thread, NULL);
	pthread_join(display_thread, NULL);
	pthread_join(frog_thread, NULL);
	/*  Create pthreads for wood move and frog control.  */
	/*  Display the output for user: win, lose or quit.  */
	pthread_mutex_destroy(&eventmutex);
	pthread_cond_destroy(&eventcond);
	return 0;
}
