#include "common.h"
#include "filters.h"
#include "matrix_accessor.h"

template <typename val_t>
void MaxFilterCPU(val_t *pInput, val_t *pOutput, cudaExtent size, MaxFilterParams params)
{
    MatrixAccessor<val_t> input(pInput, size);
    MatrixAccessor<val_t> output(pOutput, size);

    for (size_t z = 0; z < size.depth; z++)
    for (size_t y = 0; y < size.height; y++)
    for (size_t x = 0; x < size.width; x++)
    {
        val_t maxVal = input.Get(x, y, z);

        for (int oy = -params.radius; oy <= params.radius; oy++)
        for (int ox = -params.radius; ox <= params.radius; ox++)
        {
            if (input.CheckOffsetInBounds(x, y, z, ox, oy, 0))
                maxVal = max(maxVal, input.Get(x+ox, y+oy, z));
        }
        
        output.Get(x, y, z) = maxVal;
    }
}

template <typename val_t>
void MaxFilterParallel(val_t *pInput, val_t *pOutput, cudaExtent size, MaxFilterParams params, bool gpu)
{
    auto kernel = [pInput, pOutput, size, params] __host__ __device__ (dim3 blockIdx, dim3 blockDim, dim3 threadIdx)
    {
        MatrixAccessor<val_t> input(pInput, size);
        MatrixAccessor<val_t> output(pOutput, size);

        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t z = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (x < size.width && y < size.height && z < size.depth)
        {
            val_t maxVal = input.Get(x, y, z);

            for (int oy = -params.radius; oy <= params.radius; oy++)
            for (int ox = -params.radius; ox <= params.radius; ox++)
            {
                if (input.CheckOffsetInBounds(x, y, z, ox, oy, 0))
                    maxVal = max(maxVal, input.Get(x+ox, y+oy, z));
            }            

            output.Get(x, y, z) = maxVal;
        }
    };

    if (gpu)
    {
        const size_t BLOCK_SIZE = 8;

        dim3 blocks(DIVUP(size.width, BLOCK_SIZE), DIVUP(size.height, BLOCK_SIZE), DIVUP(size.depth, BLOCK_SIZE));
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

        lambda_invoker<<<blocks, threads>>>(kernel);

        auto e = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(e);
    }
    else
    {        
        const size_t BLOCK_SIZE = 512;

        dim3 blocks(DIVUP(size.width, BLOCK_SIZE), DIVUP(size.height, BLOCK_SIZE), DIVUP(size.depth, BLOCK_SIZE));
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

        for (size_t bz = 0; bz < blocks.z; bz++)
        for (size_t by = 0; by < blocks.y; by++)
        for (size_t bx = 0; bx < blocks.x; bx++)
        {
            dim3 blockIdx(bx, by, bz);

            for (size_t tz = 0; tz < threads.z; tz++)
            for (size_t ty = 0; ty < threads.y; ty++)
            for (size_t tx = 0; tx < threads.x; tx++)
            {
                dim3 threadIdx(tx, ty, tz);
                dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
                kernel(blockIdx, blockDim, threadIdx);
            }
        }
    }    
}

// template instantiation
template void MaxFilterCPU<float>(float *pInput, float *pOutput, cudaExtent size, MaxFilterParams params);
template void MaxFilterParallel<float>(float *pInput, float *pOutput, cudaExtent size, MaxFilterParams params, bool gpu);

