#include "common.h"

class MemoryManager
{
  public:

    template <typename T>
    static void AllocGrayScaleImageGPU(T **ptr, cudaExtent size)
    {
        size_t bytes = size.depth * size.height * size.width * sizeof(T);
        Alloc(ptr, bytes, true);
        cudaMemset(*ptr, 0, bytes);
    }

    template <typename T>
    static void AllocGrayScaleImageCPU(T **ptr, cudaExtent size)
    {
        size_t bytes = size.depth * size.height * size.width * sizeof(T);
        Alloc(ptr, bytes, false);
        memset(*ptr, 0, bytes);
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

            if (NULL == *ptr)
                printf("[ERROR] %s:%d %s allocating %lu bytes\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, bytes);
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
            if (NULL != ptr)
                free(ptr);     
        }      
    }
};
