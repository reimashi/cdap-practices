#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include "gputimer.h"

#define TILE_DIM 32
#define MATRIX_DIM 97
#define NITERS 1000
#define NITERS2 100

#define funcCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf( "Failed to run stmt %d ", __LINE__);                       \
            printf( "Got CUDA error ...  %s ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void matrixMult(int* A, int* B, int* C, int nT)
{
	int CValue = 0;

	int Col = blockIdx.x*TILE_DIM + threadIdx.x;
	int Row = blockIdx.y*TILE_DIM + threadIdx.y;


	__shared__ int As[TILE_DIM][TILE_DIM];
	__shared__ int Bs[TILE_DIM][TILE_DIM];


	for (int iter = 0; iter < NITERS2; iter++) {
		for (int k = 0; k < nT; k++) {

			if (k*TILE_DIM + threadIdx.x < MATRIX_DIM && Row < MATRIX_DIM)
				As[threadIdx.x][threadIdx.y] = A[Row*MATRIX_DIM + k*TILE_DIM + threadIdx.x];
			else
				As[threadIdx.x][threadIdx.y] = 0.0;

			if (k*TILE_DIM + threadIdx.y < MATRIX_DIM && Col < MATRIX_DIM)
				Bs[threadIdx.x][threadIdx.y] = B[(k*TILE_DIM + threadIdx.y)*MATRIX_DIM + Col];
			else
				Bs[threadIdx.x][threadIdx.y] = 0.0;

			__syncthreads();

			for (int n = 0; n < TILE_DIM; n++)
				CValue += As[threadIdx.x][n] * Bs[n][threadIdx.y];

			__syncthreads();
		}

		if (Row < MATRIX_DIM && Col < MATRIX_DIM)
			C[((blockIdx.x * TILE_DIM + threadIdx.x)*MATRIX_DIM) + (blockIdx.y*TILE_DIM) + threadIdx.y] = CValue;
	}

}


void randomInit(int *data, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; ++i)
	{
		data[i] = rand() % 20 - 10;  // random between -10 and 10
	}
}

void zeroInit(int *data, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; ++i)
	{
		data[i] = 0;
	}
}
int main(int argc, char ** argv) {
	GpuTimer timer;
	int * h_A; // The A matrix
	int * h_B; // The B matrix
	int * h_C; // The output C matrix
	int * d_A;
	int * d_B;
	int * d_C;

	int numTiles = (int)ceil((float)MATRIX_DIM / TILE_DIM);

	printf("Matrix dimension: %d x %d elements\n", MATRIX_DIM, MATRIX_DIM);
	printf("Block dimension: %d x %d threads\n", TILE_DIM, TILE_DIM);
	printf("Grid dimension: %d x %d blocks\n", numTiles, numTiles);
	printf("Grid dimension: %d x %d threads\n", numTiles*TILE_DIM, numTiles*TILE_DIM);
	printf("Number of iterations: %d\n", NITERS*NITERS2);



	h_A = (int *)malloc(sizeof(int)*MATRIX_DIM*MATRIX_DIM);
	h_B = (int *)malloc(sizeof(int)*MATRIX_DIM*MATRIX_DIM);
	randomInit(h_A, MATRIX_DIM*MATRIX_DIM);
	randomInit(h_B, MATRIX_DIM*MATRIX_DIM);




	h_C = (int *)malloc(sizeof(int)*MATRIX_DIM*MATRIX_DIM);
	zeroInit(h_C, MATRIX_DIM*MATRIX_DIM);

	printf("Matrices were initialized\n");

	// Allocating GPU memory
	funcCheck(cudaMalloc((void **)&d_A, sizeof(int)*MATRIX_DIM*MATRIX_DIM));
	funcCheck(cudaMalloc((void **)&d_B, sizeof(int)*MATRIX_DIM*MATRIX_DIM));
	funcCheck(cudaMalloc((void **)&d_C, sizeof(int)*MATRIX_DIM*MATRIX_DIM));

	// Copy memory to the GPU 
	funcCheck(cudaMemcpy(d_A, h_A, sizeof(int)*MATRIX_DIM*MATRIX_DIM, cudaMemcpyHostToDevice));
	funcCheck(cudaMemcpy(d_B, h_B, sizeof(int)*MATRIX_DIM*MATRIX_DIM, cudaMemcpyHostToDevice));

	// Initialize the grid and block dimensions 
	dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
	dim3 dimGrid(numTiles, numTiles, 1);

	//@@ Launch the GPU Kernel here
	timer.Start();
	for (int niter = 0; niter < NITERS; niter++)
		matrixMult << <dimGrid, dimBlock >> >(d_A, d_B, d_C, numTiles);
	timer.Stop();
	printf("Tiled kernel processing time: %f millisec.\n", timer.Elapsed());

	// Copy the results in GPU memory back to the CPU    
	funcCheck(cudaMemcpy(h_C, d_C, sizeof(int)*MATRIX_DIM*MATRIX_DIM, cudaMemcpyDeviceToHost));

	// Free the GPU memory
	funcCheck(cudaFree(d_A));
	funcCheck(cudaFree(d_B));
	funcCheck(cudaFree(d_C));

	free(h_A);
	free(h_B);
	free(h_C);

	getchar();

	return 0;
}