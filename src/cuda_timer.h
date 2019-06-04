#ifndef _CUDA_TIMER_H
#define _CUDA_TIMER_H

#include <cuda.h>
#include <cuda_runtime.h>

class CudaTimer
{
  public:
    CudaTimer();
    ~CudaTimer();

    void start();
    float stop();
    
    cudaEvent_t starte, stope;
    float elapsedMs;
};

#endif
