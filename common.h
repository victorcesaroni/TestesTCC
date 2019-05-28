#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define DEBUG_VERBOSE_ERROR(fmt, ...) printf("[ERROR] %s:%d %s "fmt"\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#define DEBUG_VERBOSE_WARNING(fmt, ...) printf("[WARNING] %s:%d %s "fmt"\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, ##__VA_ARGS__)
#define SAFE_CUDA_CALL(call)\
    {\
        cudaError_t e = call;\
        if (e != cudaSuccess) {\
            DEBUG_VERBOSE_ERROR("%s", cudaGetErrorString(e));\
        }\
    }

#define UNSIGNED_IN_BOUNDS(x, y, z, ox, oy, oz, width, height, depth)\
    ((ox >= 0 ? true : x >= -ox) /*x + ox >= 0*/ && x + ox < width) &&\
    ((oy >= 0 ? true : y >= -oy) /*y + oy >= 0*/ && y + oy < height) &&\
    ((oz >= 0 ? true : z >= -oz) /*z + oz >= 0*/ && z + oz < depth)

class CudaTimer {
  public:
    CudaTimer() {
        cudaEventCreate(&starte);
        cudaEventCreate(&stope);
    }
    ~CudaTimer() {
        cudaEventDestroy(starte);
        cudaEventDestroy(stope);
    }
    void start() {
        cudaEventRecord(starte);
        elapsedMs=0;
    }
    float stop() {
        cudaEventRecord(stope);
        cudaEventSynchronize(stope);
        cudaEventElapsedTime(&elapsedMs, starte, stope);
        return elapsedMs;
    }
    cudaEvent_t starte, stope;
    float elapsedMs;
};

template <typename T>
class MatrixAccessor 
{
  public:
    // Row major matrix acessor
    MatrixAccessor(T *data, cudaExtent size)
        : data(data), size(size)
    {        
    }

    __forceinline__ T& Get(size_t x, size_t y, size_t z)
    {
#ifdef __CUDA_ARCH__
        // device code (gpu)
        return data[z*size.depth*y*size.height + y*size.width + x];
#else
        // host code (cpu)
        return data[z*size.depth*y*size.height + y*size.width + x];
#endif
    }

    bool CheckOffsetInBounds(size_t x, size_t y, size_t z, int ox, int oy, int oz)
    {
        return UNSIGNED_IN_BOUNDS(x, y, z, ox, oy, oz, size.width, size.height, size.depth);
    }

    T *data;
    cudaExtent size;
};
