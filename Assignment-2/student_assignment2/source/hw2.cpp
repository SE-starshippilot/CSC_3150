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

#define ROW 10
#define COLUMN 50 
#define LOG_LENGTH 15
#define KB_EVENT 0b01
#define LOG_EVENT 0b10

int event_id = 0b00;
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
std::vector<int> logs_head(9);
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
		pthread_mutex_lock(&eventmutex);
		event_id |= LOG_EVENT;
		pthread_cond_broadcast(&eventcond);
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
		if (kbhit()){
			pthread_mutex_lock(&eventmutex);
			event_id |= KB_EVENT;
			pthread_cond_broadcast(&eventcond);
			pthread_mutex_unlock(&eventmutex);	
		}
	}
	printf("listen_keyboard thread exit\n");
	return NULL;
}

void *draw_map(void *t){
	/*  Draw the map  */
	int i, j;
	while (status==1){
		pthread_cond_wait(&eventcond, &eventmutex);
		system("clear");
		/* move the elements on canvas */
		if (event_id & KB_EVENT){
			char c = getchar();
			if (c=='w'){
				frog.row--;
				if (!frog.row) status=2; // win
			}
			else if (c=='s'){
				if (frog.row<ROW) frog.row++; // make sure frog is not below the bottom row
			}
			else if (c=='a'){
				frog.col--;
				if (!frog.col) status=0; // lose (hit the left border)
			}
			else if (c=='d'){
				frog.col++;
				if (frog.col==COLUMN - 1) status=0; // lose (hit the right border)
			}
			else if (c=='q'){
				status=3; // quit
			}
		}
		if (event_id & LOG_EVENT){
			// for (i=0;i<ROW;i++){
			// 	for (j=0;j<COLUMN;j++){
			// 		map[i][j]=' ';
			// 	}
			// }
			// for (i=0;i<ROW-2;i++){
			// 	for (j=0;j<LOG_LENGTH;j++){
			// 		map[i+1][(logs_head[i]+j)%COLUMN]='-';
			// 	}
			// }
			// map[frog.row][frog.col]='@';
			// for (i=0;i<ROW;i++){
			// 	for (j=0;j<COLUMN;j++){
			// 		printf("%c",map[i][j]);
			// 	}
			// 	printf("\n");
			// }
		}

		/*Draw the board*/
		for( i = 1; i < ROW; ++i ){	
			for( j = 0; j < COLUMN ; ++j )	
				map[i][j] = ' ' ;  
		} //canvas
		for( j = 0; j < COLUMN; ++j ){
			map[ROW][j] = map[0][j] ='|' ;
		}// Upper bank and lower bank
		event_id = 0b00;
		printf("Frog's location: (%d,%d)\n", frog.row, frog.col);
		// if(test_log - LOG_LENGTH + 1< 0){
		// 	for( j = 0; j <= test_log; ++j ){
		// 		map[3][j]  ='=' ;
		// 	}
		// 	for ( j = (test_log - LOG_LENGTH - 1 + COLUMN) % 50; j < COLUMN; ++j){
		// 		map[3][j] ='=' ;
		// 	}
		// }
		// else{
		// 	for( j = test_log - LOG_LENGTH+1; j <= test_log; ++j ){
		// 		map[3][j] = '=' ;
		// 	}
		// }//test log
		// map[frog.row][frog.col] = '0';//frog
		// for( int i = 0; i <= ROW; ++i){	
		// 	for (int j = 0; j < COLUMN; ++j){
		// 	std::cout << map[i][j];
		// 	}
		// 	std::cout << std::endl;
		// }
	}

	printf("draw_map thread exit\n");
	return NULL;
}


int main( int argc, char *argv[] ){
	pthread_t logs_thread, kb_thread, display_thread;
	// std::random_device rd;
	// std::mt19937 me{rd()};
	// std::uniform_int_distribution<int> distrib(0, COLUMN-1);
	// std::generate(logs_head.begin(), logs_head.end(), [&distrib, &me](){return distrib(me);});
	frog = Node( ROW, (COLUMN-1) / 2 ) ;// Frog initially at the lower bank, in the middle.
	// for(int i = 1; i < ROW; ++i ){	
	// 	for(int j = 0; j < COLUMN ; ++j )	
	// 		map[i][j] = ' ' ;  
	// } //ca
	// for(int j = 0; j < COLUMN; ++j ){
	// 	map[ROW][j] = map[0][j] ='|' ;// Upper bank and lower bank	
	// }	
	// map[frog.row][frog.col] = '0';

	pthread_mutex_init(&eventmutex, NULL);
	pthread_cond_init(&eventcond, NULL);
	pthread_create(&display_thread, NULL, draw_map, NULL);
	pthread_create(&logs_thread, NULL, logs_move, NULL);
	pthread_create(&kb_thread, NULL, listen_keyboard, NULL);
	pthread_join(logs_thread, NULL);
	pthread_join(display_thread, NULL);
	pthread_join(kb_thread, NULL);
	/* Evaluate exit status */
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
	/*  Create pthreads for wood move and frog control.  */
	/*  Display the output for user: win, lose or quit.  */
	pthread_cond_destroy(&eventcond);
	pthread_mutex_destroy(&eventmutex);
	return 0;
}
