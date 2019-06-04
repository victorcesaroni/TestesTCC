#include "cuda_timer.h"

CudaTimer::CudaTimer()
{
    cudaEventCreate(&starte);
    cudaEventCreate(&stope);
}

CudaTimer::~CudaTimer()
{
    cudaEventDestroy(starte);
    cudaEventDestroy(stope);
}

void CudaTimer::start()
{
    cudaEventRecord(starte);
    
    elapsedMs=0;
}

float CudaTimer::stop()
{
    cudaEventRecord(stope);
    cudaEventSynchronize(stope);
    cudaEventElapsedTime(&elapsedMs, starte, stope);

    return elapsedMs;
}
