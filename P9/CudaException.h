#pragma once
#include "cuda_runtime.h"
#include <exception>
#include <string>

class CudaException : public std::exception
{
protected:
	cudaError_t error;
	std::string errorMsg;

public:
	CudaException(const std::string& message, cudaError_t status);
	~CudaException() throw ();

	virtual const char* what() const throw ();
	virtual const cudaError_t getError();
};