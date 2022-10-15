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
pthread_mutex_t eventmutex;
pthread_cond_t eventcond;
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
char map[ROW][COLUMN] ; 
int status=1;
int gametime=0;
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

	/*  Move the logs  */


	/*  Check keyboard hits, to change frog's position or quit the game. */

	
	/*  Check game's status  */
	while (status){
		sleep(1);
		pthread_cond_broadcast(&eventcond);
	}
	pthread_exit(NULL);
}

void *frog_move( void *t ){
	/*  Move the frog  */
	while (status){
		if (kbhit()){
			if (!frog.y) map[frog.x][frog.y] = '|';
			else map[frog.x][frog.y] = ' ';
			switch (getch()){
				case 'w':
					frog.x--;
					break;
				case 's':
					frog.x++;
					break;
				case 'a':
					frog.y--;
					break;
				case 'd':
					frog.y++;
					break;
				case 'q':
					status = 0;
					pthread_exit(NULL);
					break;
			}
			pthread_cond_broadcast(&eventcond);
		}
	}
}

void *draw_map(void *t){
	/*  Draw the map  */
	while (status){
		pthread_cond_wait(&eventcond, &eventmutex);
		system("clear");
		memset(map, 0, sizeof(map));	
		for(int j = 0; j < COLUMN - 1; ++j )	
			map[ROW][j] = map[0][j] = '|' ;
		map[frog.x][frog.y] = '0';
		for( int i = 0; i <= ROW; ++i)	
			puts( map[i] );
	}
	pthread_exit(NULL);
}


int main( int argc, char *argv[] ){
	pthread_t logs_thread, frog_thread, kb_thread, display_thread;
	frog = Node( ROW, (COLUMN-1) / 2 ) ;// Frog initially at the lower bank, in the middle.
	pthread_mutex_init(&kbmutex, NULL);
	pthread_cond_init(&kbcond, NULL);
	pthread_mutex_init(&eventmutex, NULL);
	pthread_cond_init(&eventcond, NULL);
	// pthread_create(&display_thread, NULL, draw_map, NULL);
	// pthread_create(&logs_thread, NULL, logs_move, NULL);
	pthread_create(&frog_thread, NULL, frog_move, NULL);
	// pthread_join(logs_thread, NULL);
	// pthread_join(display_thread, NULL);
	pthread_join(frog_thread, NULL);
	/*  Create pthreads for wood move and frog control.  */
	/*  Display the output for user: win, lose or quit.  */
	pthread_mutex_destroy(&kbmutex);
	pthread_mutex_destroy(&eventmutex);
	pthread_cond_destroy(&eventcond);
	pthread_cond_destroy(&kbcond);
	return 0;
}
