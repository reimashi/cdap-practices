#include "CudaException.h"

CudaException::CudaException(const std::string& message, cudaError_t status) : error(status), errorMsg(message)
{
}

CudaException::~CudaException() throw () {}

const char* CudaException::what() const throw () {
	std::string message = std::string(cudaGetErrorString(this->error));
	return (this->errorMsg + " " + message).c_str();
}

const cudaError_t CudaException::getError() {
	return this->error;
}