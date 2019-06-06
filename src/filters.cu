#include "common.h"
#include "filters.h"
#include "matrix_accessor.h"

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
            float minVal = 0;
            float maxVal = 0;
            float meanVal = 0;
            float varianceVal = 0;
            float stdDevVal = 0;
            float sum = 0;
            float sumSqr = 0;
            int count = 0;

            maxVal = input.Get(x, y, z);
            minVal = maxVal;

            for (int oy = -params.scales[s]; oy <= params.scales[s]; oy++)
            for (int ox = -params.scales[s]; ox <= params.scales[s]; ox++)
            {
                if (input.CheckOffsetInBounds(x, y, z, ox, oy, 0))
                {
                    float val =  input.Get(x+ox, y+oy, z);

                    minVal = min(minVal, val);
                    maxVal = max(maxVal, val);
                    sum += val;
                    sumSqr += val * val;
                    varianceVal = val;
                    stdDevVal = val;
                    
                    count++;    
                }
            }

            meanVal = sum / (float)count;
            varianceVal = (sumSqr / count - (sum / count) * (sum / count)) * count / (count - 1);
            stdDevVal = sqrt(varianceVal);

            for (int i = 0; i < params.numFilters; i++)
            {
                size_t fz = GET_FEATURE_Z_INDEX(z, params.numFilters*params.numScales, featureIndex);

                switch (params.filters[i])
                {
                    case NEIGHBOORHOOD_FILTER_MIN:       
                        output.Get(x, y, fz) = minVal;
                        break;
                    case NEIGHBOORHOOD_FILTER_MAX:       
                        output.Get(x, y, fz) = maxVal;
                        break;
                    case NEIGHBOORHOOD_FILTER_MEAN:       
                        output.Get(x, y, fz) = meanVal;
                        break;
                    case NEIGHBOORHOOD_FILTER_VARIANCE:       
                        output.Get(x, y, fz) = varianceVal;
                        break;
                    case NEIGHBOORHOOD_FILTER_STD_DEV:       
                        output.Get(x, y, fz) = stdDevVal;
                        break;
                }

                featureIndex++;
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
                float minVal = 0;
                float maxVal = 0;
                float meanVal = 0;
                float varianceVal = 0;
                float stdDevVal = 0;
                float sum = 0;
                float sumSqr = 0;
                int count = 0;

                maxVal = input.Get(x, y, z);
                minVal = maxVal;

                for (int oy = -params.scales[s]; oy <= params.scales[s]; oy++)
                for (int ox = -params.scales[s]; ox <= params.scales[s]; ox++)
                {
                    if (input.CheckOffsetInBounds(x, y, z, ox, oy, 0))
                    {
                        float val =  input.Get(x+ox, y+oy, z);

                        minVal = min(minVal, val);
                        maxVal = max(maxVal, val);
                        sum += val;
                        sumSqr += val * val;
                        varianceVal = val;
                        stdDevVal = val;
                        
                        count++;    
                    }
                }

                meanVal = sum / (float)count;
                varianceVal = (sumSqr / count - (sum / count) * (sum / count)) * count / (count - 1);
                stdDevVal = sqrt(varianceVal);

                for (int i = 0; i < params.numFilters; i++)
                {
                    size_t fz = GET_FEATURE_Z_INDEX(z, params.numFilters*params.numScales, featureIndex);

                    switch (params.filters[i])
                    {
                        case NEIGHBOORHOOD_FILTER_MIN:       
                            output.Get(x, y, fz) = minVal;
                            break;
                        case NEIGHBOORHOOD_FILTER_MAX:       
                            output.Get(x, y, fz) = maxVal;
                            break;
                        case NEIGHBOORHOOD_FILTER_MEAN:       
                            output.Get(x, y, fz) = meanVal;
                            break;
                        case NEIGHBOORHOOD_FILTER_VARIANCE:       
                            output.Get(x, y, fz) = varianceVal;
                            break;
                        case NEIGHBOORHOOD_FILTER_STD_DEV:       
                            output.Get(x, y, fz) = stdDevVal;
                            break;
                    }

                    featureIndex++;
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
        const size_t BLOCK_SIZE = 8;        
        dim3 blocks(DIVUP(size.width, BLOCK_SIZE), DIVUP(size.height, BLOCK_SIZE), DIVUP(size.depth, BLOCK_SIZE));

        int nThreads = 4; 
        size_t total = blocks.x * blocks.y * blocks.z;
        size_t blocksPerThread = DIVUP(total, nThreads);

        printf("nThreads=%lu blocksPerThread=%lu blocks.x=%d blocks.y=%d blocks.z=%d\n", nThreads, blocksPerThread, blocks.x, blocks.y, blocks.z);

        #pragma omp parallel for
        for (int t = 0; t < nThreads; t++)
        {
            for (size_t idx = t*blocksPerThread; idx < t*blocksPerThread + blocksPerThread; idx++)
            {
                int bz = idx % blocks.z;
                int by = (idx / blocks.z) % blocks.y;
                int bx = idx / (blocks.y * blocks.z); 

                if (bx < blocks.x && by < blocks.y && bz < blocks.z)
                {
                    dim3 blockIdx(bx, by, bz);

                    for (size_t z = 0; z < BLOCK_SIZE; z++)
                    for (size_t y = 0; y < BLOCK_SIZE; y++)
                    for (size_t x = 0; x < BLOCK_SIZE; x++)
                    {
                        dim3 threadIdx(x, y, z);
                        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
                        kernel(blockIdx, blockDim, threadIdx);
                    }
                }
            }
        }
    }    
}

