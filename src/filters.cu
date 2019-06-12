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

////////////////////////////////////////

void MembraneProjectionsCPU(float *pInput, float *pOutput, cudaExtent size, MembraneProjectionsFilterParams params)
{
    MatrixAccessor<float> input(pInput, size);
    MatrixAccessor<float> output(pOutput, size);

    for (size_t z = 0; z < size.depth; z++)
    for (size_t y = 0; y < size.height; y++)
    for (size_t x = 0; x < size.width; x++)
    {
        if (x < size.width && y < size.height && z < size.depth)
        {
            float minVal = 0;
            float maxVal = 0;
            float sum = 0;
            double sumSqr = 0;
            int count = 0;

            maxVal = input.Get(x, y, z);
            minVal = maxVal;

            for (int rot_idx = 0; rot_idx < MP_NUM_ROTATIONS; rot_idx++)
            {
                float val = 0;

                for (int r = -params.kernelRadius+1; r <= params.kernelRadius-1; r++)
                {
                    // scale the unit vector to get the rotated kernel position
                    float2 dir = params.directions[rot_idx];

                    float2 rv = {
                        (float)x+0.5f + dir.x*r,
                        (float)y+0.5f + dir.y*r,
                    };

                    for (int w = -params.kernelWidth+1; w <= params.kernelWidth-1; w++)
                    {
                        // rotate it 90 deg = (-y, x)
                        float2 wv = {
                            rv.x + -dir.y*w,
                            rv.y + +dir.x*w,
                        };
                        
                        size_t vx = (size_t)floorf(wv.x);
                        size_t vy = (size_t)floorf(wv.y);
                        vx = max(0L, min(vx, size.width - 1));
                        vy = max(0L, min(vy, size.height - 1));
                        val += input.Get(vx, vy, z);
                    }
                }

                //val = val / (((params.kernel_radius)*2-1) * ((params.kernel_width)*2-1)); // normalize?

                maxVal = max(maxVal, val);
                minVal = min(minVal, val);
                sum += val;
                sumSqr += (double)(val*val);

                count++;      
            }
            
            float mean = (float)(sum / (float)count);
            // is this overflowing ?! (max(0, something) here to prevent nan output)
            float stdDev = (float)sqrt(max(0.0f, ((sumSqr/count - (sum/count)*(sum/count))*count) / (count-1))); // sqrt(variance)

            int featureIndex = 0;
            output.Get(x, y, GET_FEATURE_Z_INDEX(z, MP_NUM_FILTERS, featureIndex++)) = mean;
            output.Get(x, y, GET_FEATURE_Z_INDEX(z, MP_NUM_FILTERS, featureIndex++)) = stdDev;
            output.Get(x, y, GET_FEATURE_Z_INDEX(z, MP_NUM_FILTERS, featureIndex++)) = minVal;
            output.Get(x, y, GET_FEATURE_Z_INDEX(z, MP_NUM_FILTERS, featureIndex++)) = maxVal;
        }
    }
}

void MembraneProjectionsParallel(float *pInput, float *pOutput, cudaExtent size, MembraneProjectionsFilterParams params, bool gpu)
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
            float minVal = 0;
            float maxVal = 0;
            float sum = 0;
            double sumSqr = 0;
            int count = 0;

            maxVal = input.Get(x, y, z);
            minVal = maxVal;

            for (int rot_idx = 0; rot_idx < MP_NUM_ROTATIONS; rot_idx++)
            {
                float val = 0;

                for (int r = -params.kernelRadius+1; r <= params.kernelRadius-1; r++)
                {
                    // scale the unit vector to get the rotated kernel position
                    float2 dir = params.directions[rot_idx];

                    float2 rv = {
                        (float)x+0.5f + dir.x*r,
                        (float)y+0.5f + dir.y*r,
                    };

                    for (int w = -params.kernelWidth+1; w <= params.kernelWidth-1; w++)
                    {
                        // rotate it 90 deg = (-y, x)
                        float2 wv = {
                            rv.x + -dir.y*w,
                            rv.y + +dir.x*w,
                        };
                        
                        size_t vx = (size_t)floorf(wv.x);
                        size_t vy = (size_t)floorf(wv.y);
                        vx = max(0L, min(vx, size.width - 1));
                        vy = max(0L, min(vy, size.height - 1));
                        val += input.Get(vx, vy, z);
                    }
                }

                //val = val / (((params.kernel_radius)*2-1) * ((params.kernel_width)*2-1)); // normalize?

                maxVal = max(maxVal, val);
                minVal = min(minVal, val);
                sum += val;
                sumSqr += (double)(val*val);

                count++;      
            }
            
            float mean = (float)(sum / (float)count);
            // is this overflowing ?! (max(0, something) here to prevent nan output)
            float stdDev = (float)sqrt(max(0.0f, ((sumSqr/count - (sum/count)*(sum/count))*count) / (count-1))); // sqrt(variance)

            int featureIndex = 0;
            output.Get(x, y, GET_FEATURE_Z_INDEX(z, MP_NUM_FILTERS, featureIndex++)) = mean;
            output.Get(x, y, GET_FEATURE_Z_INDEX(z, MP_NUM_FILTERS, featureIndex++)) = stdDev;
            output.Get(x, y, GET_FEATURE_Z_INDEX(z, MP_NUM_FILTERS, featureIndex++)) = minVal;
            output.Get(x, y, GET_FEATURE_Z_INDEX(z, MP_NUM_FILTERS, featureIndex++)) = maxVal;
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

