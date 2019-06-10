#include "book.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define MAX_STEPS 32


void readFile();
int cpuPathTest(int limitSteps, unsigned long long *tid);
void printMaze();
void printPath(unsigned long long tid, int steps);
void printPathMaze(unsigned long long tid, int steps);
void setTime0();
void getExeTime();

struct Maze
{
	char maze[99][99];
	int rows, cols, s_x, s_y, e_x, e_y;
};

struct Maze maze;
FILE *MAZE;
struct timespec t_start, t_end;
double elapsedTime;

const int threadsPerBlock = 1024;
const int blocksPerGrid = 1024;

__global__ void testPath(int *limitSteps, struct Maze *maze, int *workDone , unsigned long long *path)
{
	unsigned long long tid = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned long long bias = blockDim.x * gridDim.x;
	unsigned long long maxRoute = 0xffffffffffffffff - (bias - 1);//max length of path
	maxRoute >>= (MAX_STEPS - *limitSteps) * 2;//(32 - 1) * 2 = 62 =>0~011 only 3 steps : right up left 
	
	while(tid <= maxRoute)
	{
		if(*workDone) break;
		
		int x = maze->s_x, y = maze->s_y;
		unsigned long long temp = tid;
		
		int i = *limitSteps;
		int steps = 0;
		do
        {
            //GetMoveDirection
            steps++;
            int direction = temp & 3;//mask
            temp >>= 2;
            //Move
            switch(direction)
            {
            case 0 :
                x += 1;
                break;
            case 1 :
                y -= 1;
                break;
            case 2 :
                x -= 1;
                break;
            case 3 :
                y += 1;
                break;
            }
            //if at Target, print path ,else keep going, if no way then break
            if(maze->maze[y][x] == '$')
            {
				*workDone = 1;
				*path = tid;
                break;
            }
            else if(maze->maze[y][x] != '.')
            {
                break;
            }
        }
        while(i--);
		
		tid += bias;
	}
}


int main()
{
	printf("GPU Version\n");
    readFile();
    printMaze();
	
	// allocate the memory on the GPU
	struct Maze *maze_ptr;
	HANDLE_ERROR(cudaMalloc((void**)&maze_ptr, sizeof(struct Maze)));
	int *limitSteps;
	HANDLE_ERROR(cudaMalloc((void**)&limitSteps, sizeof(int)));
	int *workDone;
	HANDLE_ERROR(cudaMalloc((void**)&workDone, sizeof(int)));
	unsigned long long *path;
	HANDLE_ERROR(cudaMalloc((void**)&path, sizeof(unsigned long long)));
	// copy to the GPU
	HANDLE_ERROR(cudaMemcpy( maze_ptr, &maze, sizeof(struct Maze), cudaMemcpyHostToDevice ));
	
	
	// Get start time event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	
	// start
	int *isdone = (int*)malloc( sizeof(int) );
	unsigned long long *path_ptr = (unsigned long long*)malloc( sizeof(unsigned long long) );
	int i;

	for(i = 1;i < MAX_STEPS; i++)
	{
		*isdone = 0;
		// copy to the GPU for every turn
		HANDLE_ERROR(cudaMemcpy( limitSteps, &i, sizeof(int), cudaMemcpyHostToDevice ));
		HANDLE_ERROR(cudaMemcpy( workDone, isdone, sizeof(int), cudaMemcpyHostToDevice ));
		testPath<<<blocksPerGrid,threadsPerBlock>>>(limitSteps, maze_ptr, workDone, path);
		// copy back from the GPU to the CPU
		
		HANDLE_ERROR( cudaMemcpy( isdone, workDone, sizeof(int), cudaMemcpyDeviceToHost ) );
		HANDLE_ERROR( cudaMemcpy( path_ptr, path, sizeof(unsigned long long), cudaMemcpyDeviceToHost ) );
		
		if(*isdone)
		{
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop); 
			break;
		}
		
		// Get stop time event    
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop); 
	}
	printf("search length: %d\n", i);
	printPath(*path_ptr, i);
	printPathMaze(*path_ptr, i);
	//check cuda error
    cudaError_t status = cudaGetLastError();
    if ( cudaSuccess != status ){
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(status));
        exit(1) ;
    }
	
    // Compute execution time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU time: %13f msec\n", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	
	printf("\nCPU Version\n");
	readFile();
	

	unsigned long long tid;
	setTime0();
	for(i = 1; i < MAX_STEPS; i++)
	{
		if(cpuPathTest(i, &tid)) break;
	}
	getExeTime();
	printf("search length: %d\n", i);
	printPath(tid, i);
	printPathMaze(tid, i);
    fclose(MAZE);
    return 0;
}

int cpuPathTest(int limitSteps, unsigned long long *tid)
{
	*tid = 0;
	unsigned long long bias = 1;
	unsigned long long maxRoute = 0xffffffffffffffff - (bias - 1);//max length of path
	maxRoute >>= (MAX_STEPS - limitSteps) * 2;//(32 - 1) * 2 = 62 =>0~011 only 3 steps : right up left 
	
	while(*tid <= maxRoute)
	{
		int x = maze.s_x, y = maze.s_y;
		unsigned long long temp = *tid;
		
		int i = limitSteps;
		int steps = 0;
		do
        {
            //GetMoveDirection
            steps++;
            int direction = temp & 3;//mask
            temp >>= 2;
            //Move
            switch(direction)
            {
            case 0 :
                x += 1;
                break;
            case 1 :
                y -= 1;
                break;
            case 2 :
                x -= 1;
                break;
            case 3 :
                y += 1;
                break;
            }
            //if at Target, print path ,else keep going, if no way then break
            if(maze.maze[y][x] == '$')
            {
				return 1; //this moment tid not change
            }
            else if(maze.maze[y][x] != '.')
            {
                break;
            }
        }
        while(i--);
		
		*tid += bias;
	}
	return 0;
}




void setTime0()
{
    clock_gettime(CLOCK_REALTIME, &t_start);
}

void getExeTime()
{
    clock_gettime(CLOCK_REALTIME, &t_end);
	elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
	elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
	printf("CPU time: %13f msec\n", elapsedTime);
}

void printPath(unsigned long long tid, int steps)
{
	int x = maze.s_x, y = maze.s_y;
    printf("path = %llx\n", tid);
    do
    {
        int direction = tid & 3;//mask
        tid >>= 2;
        switch(direction)
        {
        case 0:
			x += 1;
            printf("> ");
            break;
        case 1:
			y -= 1;
            printf("^ ");
            break;
        case 2:
			x -= 1;
            printf("< ");
            break;
        case 3:
			y += 1;
            printf("v ");
            break;
        }
		
		if(maze.maze[y][x] == '$')
        {
			break;
        }
    }
    while(steps--);
    printf("\n");
    return;
}

void printPathMaze(unsigned long long tid, int steps)
{
	int x = maze.s_x, y = maze.s_y;
    do
    {
        int direction = tid & 3;//mask
        tid >>= 2;
        switch(direction)
        {
        case 0:				
			maze.maze[y][x] = '>';
			x += 1;
            break;
        case 1:			
			maze.maze[y][x] = '^';
			y -= 1;
            break;
        case 2:			
			maze.maze[y][x] = '<';
			x -= 1;
            break;
        case 3:
			maze.maze[y][x] = 'v';
			y += 1;
            break;
        }
		
		if(maze.maze[y][x] == '$')
        {
			break;
        }
		
    }
    while(steps--);
	
	printMaze();

    return;
}


void readFile()
{
    MAZE = fopen("maze.txt", "r");
    fscanf(MAZE, "%d %d", &maze.rows, &maze.cols);
    fgetc(MAZE);
    printf("rows = %d, cols = %d\n", maze.rows, maze.cols);

    int i = 0, j = 0;
    char c = fgetc(MAZE);
    while((c = fgetc(MAZE)) != EOF)
    {
        if(c == '\n')
        {
            i++;
            j = 0;
        }
        else
        {
            maze.maze[i][j] = c;
            if(c == '*')
            {
                maze.s_x = j;
                maze.s_y = i;
            }
            else if(c == '$')
            {
                maze.e_x = j;
                maze.e_y = i;
            }
            j++;
        }
    }
    return;
}

void printMaze()
{
    printf("start = (%d %d) \nend=(%d %d) \n", maze.s_x, maze.s_y, maze.e_x, maze.e_y);
    int i,j;
    for(i = 0; i < maze.rows; i++)
    {
        for(j = 0; j < maze.cols; j++)
        {
            printf("%c ", maze.maze[i][j]);
        }
        printf("\n");
    }
    return;
}


/*
8 13
#############
#*#...#...#.#
#.#.#.#.#.#.#
#.#.#.#.#.#.#
#...#.......#
##########.##
#$..........#
#############
0xAAAAAF003F05400
*/
