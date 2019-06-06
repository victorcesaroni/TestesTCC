#include "common.h"

template <typename T>
class MatrixAccessor 
{
  public:
    // Row major matrix acessor
    __host__ __device__ MatrixAccessor(T *data, cudaExtent size)
        : data(data), size(size)
    {        
    }

    __host__ __device__ __forceinline__ T& Get(size_t x, size_t y, size_t z)
    {
#ifdef __CUDA_ARCH__
        // device code (gpu)
        return data[z*size.height*size.width + y*size.width + x];
#else
        // host code (cpu)
        return data[z*size.height*size.width + y*size.width + x];
#endif
    }

    __host__ __device__ __forceinline__ bool CheckOffsetInBounds(size_t x, size_t y, size_t z, int ox, int oy, int oz)
    {
        return UNSIGNED_IN_BOUNDS(x, y, z, ox, oy, oz, size.width, size.height, size.depth);
    }

    T *data;
    cudaExtent size;
};
