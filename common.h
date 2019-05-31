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

template<class F> __global__ void lambda_invoker(F func) {
    func(blockIdx, blockDim, threadIdx);
}

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

class MemoryManager
{
  public:

    template <typename T>
    static void AllocGrayScaleImageGPU(T **ptr, cudaExtent size)
    {
        size_t bytes = size.depth * size.height * size.width * sizeof(T);
        Alloc(ptr, bytes, true);
    }

    template <typename T>
    static void AllocGrayScaleImageCPU(T **ptr, cudaExtent size)
    {
        size_t bytes = size.depth * size.height * size.width * sizeof(T);
        Alloc(ptr, bytes, false);
    }

    template <typename T>
    static void CopyGrayScaleImageToGPU(T *ptr, T *device_ptr, cudaExtent size)
    {
        size_t bytes = size.depth * size.height * size.width * sizeof(T);
        auto e = cudaMemcpy(device_ptr, ptr, bytes, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(e);
    }

    template <typename T>
    static void CopyGrayScaleImageFromGPU(T *ptr, T *device_ptr, cudaExtent size)
    {
        size_t bytes = size.depth * size.height * size.width * sizeof(T);
        auto e = cudaMemcpy(ptr, device_ptr, bytes, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(e);
    }

    static void FreeGPU(void *ptr)
    {
        Free(ptr, true);
    }

    static void FreeCPU(void *ptr)
    {
        Free(ptr, false);
    }

    template <typename T>
    static void Alloc(T **ptr, size_t bytes, bool gpu)
    {
        if (gpu)
        {
            auto e = cudaMalloc(&(*ptr), bytes);
            CHECK_CUDA_ERROR(e);
        }
        else
        {
            (*ptr) = (T *)malloc(bytes);            
        }        
    }

    static void Free(void *ptr, bool gpu)
    {
        if (gpu)
        {
            auto e = cudaFree(ptr);
            CHECK_CUDA_ERROR(e);
        }
        else
        {   
            free(ptr);     
        }      
    }
};

class FileManager
{
  public:
    template <typename T>
    static void Read(T *ptr, const char *filePath, cudaExtent size)
    {
        size_t fileSize = size.depth * size.height * size.width * sizeof(T);

        FILE *file = fopen(filePath, "r");
        size_t read = fread(ptr, fileSize, 1, file);        
        fclose(file);
    }

    template <typename original_t, typename as_t>
    static void ReadAs(as_t *ptr, const char *filePath, cudaExtent size)
    {
        size_t fileSize = size.depth * size.height * size.width * sizeof(original_t);
        unsigned char *tmp = (unsigned char*)malloc(fileSize);

        FILE *file = fopen(filePath, "r");
        size_t read = fread(tmp, fileSize, 1, file);
        fclose(file);

        size_t j = 0;
        for (size_t i = 0; i < fileSize; i += sizeof(original_t))
            ptr[j++] = (as_t)(*(original_t*)&tmp[i]);        

        free(tmp);
    }

    template <typename T>
    static void Write(T *ptr, const char *filePath, cudaExtent size)
    {
        size_t fileSize = size.depth * size.height * size.width * sizeof(T);

        FILE *file = fopen(filePath, "w+");
        size_t write = fwrite(ptr, fileSize, 1, file);
        fclose(file);
    }
};
