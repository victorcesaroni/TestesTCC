#ifndef _COMMON_H
#define _COMMON_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <nvfunctional>

#define DEBUG_VERBOSE_ERROR(fmt, ...) printf("[ERROR] %s:%d %s "fmt"\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#define DEBUG_VERBOSE_WARNING(fmt, ...) printf("[WARNING] %s:%d %s "fmt"\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#define CHECK_CUDA_ERROR(e)                                                                                       \
    {                                                                                                             \
        if (e != cudaError_t::cudaSuccess)                                                                        \
        {                                                                                                         \
            printf("[CUDA ERROR] %s: %s\n",  __PRETTY_FUNCTION__, cudaGetErrorString(e)); \
        }                                                                                                         \
    }

#define UNSIGNED_IN_BOUNDS(x, y, z, ox, oy, oz, width, height, depth)   \
    ((ox >= 0 ? true : x >= -ox) /*x + ox >= 0*/ && x + ox < width) &&  \
    ((oy >= 0 ? true : y >= -oy) /*y + oy >= 0*/ && y + oy < height) && \
    ((oz >= 0 ? true : z >= -oz) /*z + oz >= 0*/ && z + oz < depth)

#define DIVUP(a,b) (a+b-1)/b

template<class F>
__global__ void lambda_invoker(F func)
{
    func(blockIdx, blockDim, threadIdx);
}

#endif