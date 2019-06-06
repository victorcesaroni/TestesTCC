#include "common.h"
#include "filters.h"
#include "matrix_accessor.h"


#if 0
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MaxFilter

void MaxFilterCPU(float *pInput, float *pOutput, cudaExtent size, MaxFilterParams params)
{
    MatrixAccessor<float> input(pInput, size);
    MatrixAccessor<float> output(pOutput, size);

    for (size_t z = 0; z < size.depth; z++)
    for (size_t y = 0; y < size.height; y++)
    for (size_t x = 0; x < size.width; x++)
    {
        float maxVal = input.Get(x, y, z);

        for (int oy = -params.radius; oy <= params.radius; oy++)
        for (int ox = -params.radius; ox <= params.radius; ox++)
        {
            if (input.CheckOffsetInBounds(x, y, z, ox, oy, 0))
                maxVal = max(maxVal, input.Get(x+ox, y+oy, z));
        }
        
        output.Get(x, y, z) = maxVal;
    }
}

void MaxFilterParallel(float *pInput, float *pOutput, cudaExtent size, MaxFilterParams params, bool gpu)
{
    auto kernel = [pInput, pOutput, size, params] __host__ __device__ (dim3 blockIdx, dim3 blockDim, dim3 threadIdx)
    {
        MatrixAccessor<float> input(pInput, size);
        MatrixAccessor<float> output(pOutput, size);

        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t z = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (x < size.width && y < size.height && z < size.depth)
        {
            float maxVal = input.Get(x, y, z);

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
        int nThreads = 4;        
        const size_t BLOCK_SIZE = 512;
        size_t nVoxelsPerThread = DIVUP(BLOCK_SIZE, nThreads);
        
        dim3 blocks(DIVUP(size.width, BLOCK_SIZE), DIVUP(size.height, BLOCK_SIZE), DIVUP(size.depth, BLOCK_SIZE));

        printf("nThreads=%lu nVoxelsPerThread=%d blocks.x=%d blocks.y=%d blocks.z=%d\n", nThreads, nVoxelsPerThread, blocks.x, blocks.y, blocks.z);
        for (size_t bz = 0; bz < blocks.z; bz++)
        for (size_t by = 0; by < blocks.y; by++)
        for (size_t bx = 0; bx < blocks.x; bx++)
        {
            dim3 blockIdx(bx, by, bz);

            #pragma omp parallel for
            for (int t = 0; t < nThreads; t++)
            {
                for (size_t tz = 0; tz < min(size.depth, nVoxelsPerThread); tz++)
                for (size_t ty = 0; ty < min(size.height, nVoxelsPerThread); ty++)
                for (size_t tx = 0; tx < min(size.width, nVoxelsPerThread); tx++)
                {
                    dim3 threadIdx(tx + t * nVoxelsPerThread, ty + t * nVoxelsPerThread, tz + t * nVoxelsPerThread);
                    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
                    kernel(blockIdx, blockDim, threadIdx);
                }
            }
        }
    }    
}

#endif

void NeighboorhoodFilterCPU(float *pInput, float *pOutput, cudaExtent size, NeighboorhoodFilterParams params)
{
    MatrixAccessor<float> input(pInput, size);
    MatrixAccessor<float> output(pOutput, size);

    for (size_t z = 0; z < size.depth; z++)
    for (size_t y = 0; y < size.height; y++)
    for (size_t x = 0; x < size.width; x++)
    {
        int featureIndex = 0;

        for (int s = 0; s < params.numScales; s++)
        {
            float maxVal, minVal, meanVal, varianceVal, stdDevVal, sum;
            int count = 0;

            maxVal = minVal = meanVal = varianceVal = stdDevVal = input.Get(x, y, z);

            for (int oy = -params.scales[s]; oy <= params.scales[s]; oy++)
            for (int ox = -params.scales[s]; ox <= params.scales[s]; ox++)
            {
                if (input.CheckOffsetInBounds(x, y, z, ox, oy, 0))
                {
                    float val =  input.Get(x+ox, y+oy, z);

                    maxVal = max(maxVal, val);
                    minVal = min(maxVal, val);
                    sum += val;
                    varianceVal = val;
                    stdDevVal = val;
                    
                    count++;    
                }
            }

            meanVal = sum / (float)count;

            for (int i = 0; i < params.numFilters; i++)
            {
                printf("i=%d, idx=%lu, feat=%d, z=%lu, numFilters=%d, numScales=%d\n", i, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex), featureIndex, z, params.numFilters,params.numScales);
                
                switch (params.filters[i])
                {
                    case NEIGHBOORHOOD_FILTER_MIN:       
                        output.Get(x, y, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex++)) = minVal;
                        break;
                    case NEIGHBOORHOOD_FILTER_MAX:       
                        output.Get(x, y, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex++)) = minVal;
                        break;
                    case NEIGHBOORHOOD_FILTER_MEAN:       
                        output.Get(x, y, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex++)) = meanVal;
                        break;
                    case NEIGHBOORHOOD_FILTER_VARIANCE:       
                        output.Get(x, y, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex++)) = varianceVal;
                        break;
                    case NEIGHBOORHOOD_FILTER_STD_DEV:       
                        output.Get(x, y, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex++)) = stdDevVal;
                        break;
                }
            }
        }
    }
}

void NeighboorhoodFilterParallel(float *pInput, float *pOutput, cudaExtent size, NeighboorhoodFilterParams params, bool gpu)
{
    auto kernel = [pInput, pOutput, size, params] __host__ __device__ (dim3 blockIdx, dim3 blockDim, dim3 threadIdx)
    {
        MatrixAccessor<float> input(pInput, size);
        MatrixAccessor<float> output(pOutput, size);

        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t z = blockIdx.z * blockDim.z + threadIdx.z;
        
        if (x < size.width && y < size.height && z < size.depth)
        {
           int featureIndex = 0;

            for (int s = 0; s < params.numScales; s++)
            {
                float maxVal, minVal, meanVal, varianceVal, stdDevVal, sum;
                int count = 0;

                maxVal = minVal = meanVal = varianceVal = stdDevVal = input.Get(x, y, z);

                for (int oy = -params.scales[s]; oy <= params.scales[s]; oy++)
                for (int ox = -params.scales[s]; ox <= params.scales[s]; ox++)
                {
                    if (input.CheckOffsetInBounds(x, y, z, ox, oy, 0))
                    {
                        float val =  input.Get(x+ox, y+oy, z);

                        maxVal = max(maxVal, val);
                        minVal = min(maxVal, val);
                        sum += val;
                        varianceVal = val;
                        stdDevVal = val;
                        
                        count++;
                    }
                }

                meanVal = sum / (float)count;

                for (int i = 0; i < params.numFilters; i++)
                {   
                    switch (params.filters[i])
                    {
                        case NEIGHBOORHOOD_FILTER_MIN:       
                            output.Get(x, y, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex++)) = minVal;
                            break;
                        case NEIGHBOORHOOD_FILTER_MAX:       
                            output.Get(x, y, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex++)) = minVal;
                            break;
                        case NEIGHBOORHOOD_FILTER_MEAN:       
                            output.Get(x, y, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex++)) = meanVal;
                            break;
                        case NEIGHBOORHOOD_FILTER_VARIANCE:       
                            output.Get(x, y, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex++)) = varianceVal;
                            break;
                        case NEIGHBOORHOOD_FILTER_STD_DEV:       
                            output.Get(x, y, GET_FEATURE_Z_INDEX(size.depth, params.numFilters*params.numScales, z, featureIndex++)) = stdDevVal;
                            break;
                    }
                }
            }
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
        int nThreads = 4;        
        const size_t BLOCK_SIZE = 512;
        size_t nVoxelsPerThread = DIVUP(BLOCK_SIZE, nThreads);
        
        dim3 blocks(DIVUP(size.width, BLOCK_SIZE), DIVUP(size.height, BLOCK_SIZE), DIVUP(size.depth, BLOCK_SIZE));

        //printf("nThreads=%lu nVoxelsPerThread=%d blocks.x=%d blocks.y=%d blocks.z=%d\n", nThreads, nVoxelsPerThread, blocks.x, blocks.y, blocks.z);
        for (size_t bz = 0; bz < blocks.z; bz++)
        for (size_t by = 0; by < blocks.y; by++)
        for (size_t bx = 0; bx < blocks.x; bx++)
        {
            dim3 blockIdx(bx, by, bz);

            #pragma omp parallel for
            for (int t = 0; t < nThreads; t++)
            {
                for (size_t tz = 0; tz < min(size.depth, nVoxelsPerThread); tz++)
                for (size_t ty = 0; ty < min(size.height, nVoxelsPerThread); ty++)
                for (size_t tx = 0; tx < min(size.width, nVoxelsPerThread); tx++)
                {
                    dim3 threadIdx(tx + t * nVoxelsPerThread, ty + t * nVoxelsPerThread, tz + t * nVoxelsPerThread);
                    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
                    kernel(blockIdx, blockDim, threadIdx);
                }
            }
        }
    }    
}


