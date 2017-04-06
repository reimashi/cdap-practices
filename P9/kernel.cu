#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <exception>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CudaException.h"

#define VECTOR_DIM 2048

void randomInit(int *data, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++)
	{
		data[i] = (int)(rand() % 21) - 10;
	}
}

void showVector(char *str, int *data, int size) {
	printf("%s:", str);
	for (int i = 0; i < size; i++) {
		printf("[%d] %d ", i, data[i]);
	}
	printf("\n");
}

__global__ void scan(int *g_idata, int *g_odata, int n, int blockSize) {
	__shared__ int temp[VECTOR_DIM]; // allocated on invocation  
	int thid = threadIdx.x;
	int offset = 1;

	temp[thid] = g_idata[thid + (blockIdx.x * blockSize)]; // load input into shared memory  

	for (int d = n >> 1; d > 0; d >>= 1)                    // build sum in place up the tree  
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) { temp[n - 1] = 0; } // clear the last element  
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	g_odata[thid + (blockIdx.x * blockSize)] = temp[thid]; // write results to device memory  
}

// Suma un entero a cada elemento del array
__global__ void add(int *g_iodata, int toAdd) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	g_iodata[index] += toAdd;
}

// Algoritmo scan secuencial por CPU
void scanCPU(int* output, int* input, int length) {
	output[0] = 0;
	for (int j = 1; j < length; ++j) {
		output[j] = input[j - 1] + output[j - 1];
	}
}

int main() {
	try {
		cudaError_t cudaStatus;

		// Se obtiene la GPU a usar
		std::cout << "Escaneando dispositivos..." << std::endl;
		int deviceId;
		cudaStatus = cudaGetDevice(&deviceId);
		if (cudaStatus != cudaSuccess) throw CudaException("No se puede detectar ninguna GPU Cuda.", cudaStatus);

		// Se obtiene la información de dicha GPU
		cudaDeviceProp deviceInfo;
		cudaStatus = cudaGetDeviceProperties(&deviceInfo, deviceId);
		if (cudaStatus != cudaSuccess) throw CudaException("No se ha podido acceder a la información de la GPU.", cudaStatus);
		std::cout << "Se usara la GPU: " << deviceInfo.name << std::endl;
		std::cout << "Esta GPU puede ejecutar " << deviceInfo.maxThreadsPerBlock << " hilos por bloque." << std::endl;

		// Calculamos los tamaños de bloque
		int blockElements = deviceInfo.maxThreadsPerBlock;
		int blocks = VECTOR_DIM / blockElements;
		std::cout << "El vector se dividira en " << blocks << " bloques." << std::endl;

		// Se comprueba que el tamaño del vector sea un multiplo del numero de hilos por bloque
		if (VECTOR_DIM % blockElements != 0) {
			throw std::exception("El vector a escanear debe ser un multiplo del numero de hilos por bloque.");
		}

		// Reservamos memoria para los vectores
		int *d_A, *d_B;
		int *h_A = (int *)malloc(VECTOR_DIM * sizeof(int));
		int *h_B = (int *)malloc(VECTOR_DIM * sizeof(int));
		int *h_SumA = (int *)malloc(blocks * sizeof(int));
		int *h_SumScanned = (int *)malloc(blocks * sizeof(int));

		cudaMalloc((void **)&d_A, VECTOR_DIM * sizeof(int));
		cudaMalloc((void **)&d_B, VECTOR_DIM * sizeof(int));

		// Inicializamos con valores aleatorios el vector de entrada
		randomInit(h_A, VECTOR_DIM);
		showVector((char *) "Vector to scan\n", h_A, VECTOR_DIM);

		// Copiamos el vector a memoria del dispositivo
		cudaMemcpy(d_A, h_A, VECTOR_DIM * sizeof(int), cudaMemcpyHostToDevice);

		// Realizamos el scan por bloques
		for (int i = 0; i < blocks; i++) {
			long memoryOffset = i * blockElements;
			scan << <1, blockElements >> > (d_A + memoryOffset, d_B + memoryOffset, VECTOR_DIM, blockElements);
		}

		// Copiamos el resultado a memoria principal
		cudaMemcpy(h_B, d_B, VECTOR_DIM * sizeof(int), cudaMemcpyDeviceToHost);

		// Obtenemos el vector de mascara para los scan parciales
		for (int i = 0; i < blocks; i++) {
			int offset = (i * blockElements) + (blockElements - 1);
			h_SumA[i] = h_A[offset] + h_B[offset];
		}

		// Realizamos scan al vector de mascara.
		// NOTA: Se podría hacer por GPU pero complica el ejercicio. Ademas, tendríamos que volver a considerar
		// el numero de hilos por bloque y hacer una funcion recursiva para manejar cualquier tamaño de vector.
		scanCPU(h_SumScanned, h_SumA, blocks);

		// Sumamos las mascaras al vector con el scan por bloques
		for (int i = 0; i < blocks; i++) {
			long memoryOffset = i * blockElements;
			add<<<1, blockElements>>> (d_B + memoryOffset, h_SumScanned[i]);
		}

		// Pasamos el resultado final a memoria principal y lo mostramos
		cudaMemcpy(h_B, d_B, VECTOR_DIM * sizeof(int), cudaMemcpyDeviceToHost);
		showVector((char *) "Vector after scan\n", h_B, VECTOR_DIM);

		// Liberamos memoria
		free(h_A);
		free(h_B);
		free(h_SumA);
		free(h_SumScanned);
		cudaFree(d_A);
		cudaFree(d_B);

		getchar();
		return 0;
	}
	catch (std::exception e) {
		std::cerr << e.what() << std::endl;
		getchar();
		return 1;
	}
}