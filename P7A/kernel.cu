#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include "PPMImage.h"
#include "CudaException.h"

#define COLOR_FILTER_R 1.0
#define COLOR_FILTER_G 0.0
#define COLOR_FILTER_B 0.75

// Kernel que procesa cada color del pixel
__global__ void colorFilterKernel(unsigned char *data, long dataSize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int col = i % 3;
	data[i] = data[i] * (col == 0 ? COLOR_FILTER_R : (col == 1 ? COLOR_FILTER_G : COLOR_FILTER_B));
}

// Función que procesa la imagen para hacer un filtro de color
cudaError_t colorFilterWithCuda(PPMImage *image)
{
	cudaError_t cudaStatus;
	unsigned char *imageData_d;

	try {
		// Elegimos la primera GPU (No tenemos en cuenta sistemas multi GPU)
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) throw new CudaException("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", cudaStatus);

		cudaStatus = cudaMalloc((void**)&imageData_d, image->dataSize);
		if (cudaStatus != cudaSuccess) throw new CudaException("cudaMalloc failed!", cudaStatus);

		cudaStatus = cudaMemcpy(imageData_d, &image->data, image->dataSize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) throw new CudaException("cudaMemcpy failed!", cudaStatus);

		int numBlocks_h = image->dataSize / 32;
		colorFilterKernel << <numBlocks_h, 32 >> > (imageData_d, image->dataSize);

		// Detectamos el error de ejecución
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) throw new CudaException("colorFilterKernel launch failed!", cudaStatus);

		// Espera a que termine todas las ejecuciones y detecta errores
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) throw new CudaException("cudaDeviceSynchronize returned error code %d after launching addKernel!", cudaStatus);

		// Copiamos el vector resultado a RAM
		cudaStatus = cudaMemcpy(image->data, imageData_d, image->dataSize, cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) throw new CudaException("cudaMemcpy failed!", cudaStatus);
	}
	catch (CudaException e) {
		cudaFree(imageData_d);
		throw e;
	}

	return cudaStatus;
}

int main()
{
	try {
		std::cout << "Leyendo imagen... ";

		PPMImage sourceImage;
		loadPpmImage("lena.ppm", &sourceImage);

		std::cout << "correcto!" << std::endl;
		std::cout << "Procesando imagen... ";

		cudaError_t cudaStatus = colorFilterWithCuda(&sourceImage);
		if (cudaStatus != cudaSuccess) throw new CudaException("Ha habido un error al convertir la imagen con CUDA.", cudaStatus);

		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) throw new CudaException("Error al reiniciar el estado del dispositivo.", cudaStatus);

		std::cout << "correcto!" << std::endl;
		std::cout << "Guardando imagen... ";

		savePpmImage("lenaFiltered.ppm", &sourceImage);

		std::cout << "correcto!" << std::endl;

		return 0;
	}
	catch (std::exception e) {
		std::cout << "error!" << std::endl;
		std::cerr << e.what() << std::endl;
		std::cout << "Pulsa ENTER para salir..." << std::endl;
		int x = std::cin.get();

		return 1;
	}
}
