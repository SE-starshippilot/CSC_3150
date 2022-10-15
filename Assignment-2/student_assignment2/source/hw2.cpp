#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 
#define LOG_LENGTH 15

pthread_mutex_t kbmutex;
pthread_cond_t kbcond;
struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 

/*
 +--------> COLUMN++
 |
 |
 v
 ROW++ 
*/
char map[ROW+10][COLUMN] ; 

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch, retval;
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
		retval = ch;
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void *logs_move( void *t ){

	/*  Move the logs  */


	/*  Check keyboard hits, to change frog's position or quit the game. */

	
	/*  Check game's status  */


	/*  Print the map on the screen  */

	return t;
}

void *frog_move( void *t ){

	/*  Move the frog  */
	char ch;
	pthread_mutex_lock(&kbmutex);
	while (true){
		pthread_cond_wait(&kbcond, &kbmutex);
		ch = getchar();
	}

}
void kb_listen( void *t ){
	pthread_mutex_lock(&kbmutex);
	while (true){

	}
}
int main( int argc, char *argv[] ){
	pthread_mutex_init(&kbmutex, NULL);
	pthread_cond_init(&kbcond, NULL);
	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	// Initiazlize the river
	int i , j , ch; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	
	// Initialize the river bank
	for( j = 0; j < COLUMN - 1; ++j )// Upper bank	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )// Lower bank	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ;// Frog initially at the lower bank, in the middle.
	map[frog.x][frog.y] = '0' ; 

	//Print the map into screen
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );


	/*  Create pthreads for wood move and frog control.  */
	// pthread_t frog_thread, kb_thread, log_thread, monitor_thread;

	// pthread_join(monitor_thread, NULL ) ;
	/*  Display the output for user: win, lose or quit.  */

	// pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
	// pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
	return 0;

}
