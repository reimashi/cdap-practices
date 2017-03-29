#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include "gputimer.h"

#include "Constants.h"

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

	int Row = blockIdx.y * TILE_DIM + threadIdx.y;
	int Col = blockIdx.x * TILE_DIM + threadIdx.x;

	__shared__ int As[TILE_DIM][TILE_DIM];
	__shared__ int Bs[TILE_DIM][TILE_DIM];

	for (int k = 0; k < nT; k++) {
		if (k * TILE_DIM + threadIdx.x < MATRIX_DIM && Row < MATRIX_DIM)
			As[threadIdx.y][threadIdx.x] = A[Row * MATRIX_DIM + k * TILE_DIM + threadIdx.x];
		else
			As[threadIdx.y][threadIdx.x] = 0.0;

		if (k*TILE_DIM + threadIdx.y < MATRIX_DIM && Col < MATRIX_DIM)
			Bs[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y) * MATRIX_DIM + Col];
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0;

		__syncthreads();

		for (int n = 0; n < TILE_DIM; n++) {
			CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
		}

		__syncthreads();
	}

	C[threadIdx.x + TILE_DIM] = CValue;
}

// Inicializa una matriz con valores aleatorios
void randomInit(int *data, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; ++i)
	{
		data[i] = rand() % 20 - 10;  // random between -10 and 10
	}
}

// Inicializa una matriz con ceros
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
	int numTiles = (int)ceil((float)TILE_DIM / TILE_DIM);

	printf("Matrix dimension: %d x %d elements\n", MATRIX_DIM, MATRIX_DIM);
	printf("Block dimension: %d x %d threads\n", TILE_DIM, TILE_DIM);
	printf("Grid dimension: %d x %d blocks\n", numTiles, numTiles);
	printf("Grid dimension: %d x %d threads\n", numTiles*TILE_DIM, numTiles*TILE_DIM);
	printf("Number of iterations: %d\n", NITERS*NITERS2);

	int memoryPos = TILE_DIM * TILE_DIM;
	int memorySize = sizeof(int) * memoryPos;

	h_A = (int *)malloc(memorySize);
	h_B = (int *)malloc(memorySize);
	randomInit(h_A, memoryPos);
	randomInit(h_B, memoryPos);

	h_C = (int *)malloc(memorySize);
	zeroInit(h_C, memoryPos);

	printf("Matrices were initialized\n");

	// Allocating GPU memory
	funcCheck(cudaMalloc((void **)&d_A, memorySize));
	funcCheck(cudaMalloc((void **)&d_B, memorySize));
	funcCheck(cudaMalloc((void **)&d_C, memorySize));

	// Copy memory to the GPU 
	funcCheck(cudaMemcpy(d_A, h_A, memorySize, cudaMemcpyHostToDevice));
	funcCheck(cudaMemcpy(d_B, h_B, memorySize, cudaMemcpyHostToDevice));

	// Initialize the grid and block dimensions 
	dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
	dim3 dimGrid(numTiles, numTiles, 1);

	//@@ Launch the GPU Kernel here
	timer.Start();
	for (int niter = 0; niter < NITERS*NITERS2; niter++) {
		matrixMult << <dimGrid, dimBlock >> > (d_A, d_B, d_C, numTiles);
	}
	timer.Stop();
	printf("Tiled & coalescend kernel processing time: %f millisec.\n", timer.Elapsed());

	// Copy the results in GPU memory back to the CPU    
	funcCheck(cudaMemcpy(h_C, d_C, memorySize, cudaMemcpyDeviceToHost));

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