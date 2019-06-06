#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "common.h"
#include "cuda_timer.h"
#include "filters.h"
#include "misc.h"
#include "memory_manager.h"
#include "file_manager.h"
#include "matrix_accessor.h"

void TestCUDA();

int main(int argc, const char *argv[])
{
    //nvcc main.cu --expt-extended-lambda --std=c++11 -Xcompiler -fopenmp --compiler-options -fPIC -O3 -D__INTEL_COMPILER -o main
    //nvcc main.cu -gencode arch=compute_61,code=[sm_61,compute_61] --expt-extended-lambda --std=c++11 -Xcompiler -fopenmp --compiler-options -fPIC -O3 -D__INTEL_COMPILER -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -o main
    
    if (argc != 7)
    {
        printf("USE: %s <inputPath> <outputPath> <width> <height> <depth> <process type (0=CPU, 1=parallel CPU, 2=parallel GPU)>\n", argv[0]);
        TestCUDA();
        exit(0);
    }

    const char *inputPath = argv[1];
    const char *outputPath = argv[2];
    size_t width = atoi(argv[3]);
    size_t height = atoi(argv[4]);
    size_t depth = atoi(argv[5]);
    int type = atoi(argv[6]);

    cudaExtent size = make_cudaExtent(width, height, depth);

    printf("Generating image\n");
    // test image
    InputGenerator::Generate("test.b", size);

    printf("Allocating input CPU\n");
    float *input = NULL;
    MemoryManager::AllocGrayScaleImageCPU<float>(&input, size);    

    printf("Reading input from disk\n");
    FileManager::ReadAs<float, float>(input, inputPath, size);

    CudaTimer timer;

    NeighboorhoodFilterParams params;
    params.filters[0] = NEIGHBOORHOOD_FILTER_MIN;
    params.filters[1] = NEIGHBOORHOOD_FILTER_MAX;
    params.filters[2] = NEIGHBOORHOOD_FILTER_MEAN;
    params.filters[3] = NEIGHBOORHOOD_FILTER_VARIANCE;
    params.filters[4] = NEIGHBOORHOOD_FILTER_STD_DEV;
    params.numFilters = 5;

    params.scales[0] = 1;
    params.scales[1] = 2;
    params.scales[2] = 4;
    params.scales[3] = 8;
    params.numScales = 4;

    printf("Allocating CPU %lu x %lu x %lu\n", params.GetOutputSize(size).width, params.GetOutputSize(size).height, params.GetOutputSize(size).depth);

    float *output = NULL;
    MemoryManager::AllocGrayScaleImageCPU<float>(&output, params.GetOutputSize(size));

    if (type == 2)
    {
        printf("NeighboorhoodFilterParallel (GPU) ");

        printf("Allocating in GPU\n");
        float *d_input = NULL;
        float *d_output = NULL;
        MemoryManager::AllocGrayScaleImageGPU<float>(&d_input, size);
        MemoryManager::AllocGrayScaleImageGPU<float>(&d_output, params.GetOutputSize(size));

        timer.start();
        printf("Copying to GPU (timer start) ");
        MemoryManager::CopyGrayScaleImageToGPU<float>(input, d_input, size);

        NeighboorhoodFilterParallel(d_input, d_output, size, params, true);

        MemoryManager::CopyGrayScaleImageFromGPU<float>(output, d_output, params.GetOutputSize(size));
        printf("Copying from GPU (timer stop) ");
        timer.stop();

        printf("-- %fms\n", timer.elapsedMs);
        
        printf("Deallocating in GPU\n");
        MemoryManager::FreeGPU(d_input);
        MemoryManager::FreeGPU(d_output);
    }

    if (type == 1)
    {
        printf("NeighboorhoodFilterParallel (CPU) ");
        timer.start();
        NeighboorhoodFilterParallel(input, output, size, params, false);
        timer.stop();
        
        printf("-- %fms\n", timer.elapsedMs);
    }

    if (type == 0)
    {
        printf("NeighboorhoodFilterCPU (CPU) ");
        timer.start();
        NeighboorhoodFilterCPU(input, output, size, params);
        timer.stop();
        
        printf("-- %fms\n", timer.elapsedMs);
    }

    printf("Writing output to disk\n");
    FileManager::Write<float>(output, outputPath, params.GetOutputSize(size));

    MemoryManager::FreeCPU(input);
    MemoryManager::FreeCPU(output);    
}


template <typename val_t>
__global__ void CopyKernel(val_t *d_input, val_t *d_output, cudaExtent size)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    MatrixAccessor<val_t> input(d_input, size);
    MatrixAccessor<val_t> output(d_output, size);
    
    if (x < size.width && y < size.height && z < size.depth)
    {
        output.Get(x, y, z) = input.Get(x, y, z);
    }
}

void TestCUDA()
{
    printf("TEST CUDA\n");

    cudaExtent size = make_cudaExtent(4096, 1023, 17);

    size_t bytes = size.depth * size.height * size.width * sizeof(float);
    
    printf("Alloc CPU\n");
    float *pInput = (float*)malloc(bytes);
    float *pOutput = (float*)malloc(bytes);

    float *d_input;
    float *d_output;
    cudaError_t e;    

    printf("Alloc GPU\n");
    e = cudaMalloc(&d_input, bytes);
    CHECK_CUDA_ERROR(e);
    e = cudaMalloc(&d_output, bytes);
    CHECK_CUDA_ERROR(e);

    MatrixAccessor<float> input(pInput, size);
    MatrixAccessor<float> output(pOutput, size);

    for (size_t z = 0; z < size.depth; z++)
    for (size_t y = 0; y < size.height; y++)
    for (size_t x = 0; x < size.width; x++)
    {
        input.Get(x,y,z) = (float)(rand() % 255);
    }

    printf("H2D\n");
    e = cudaMemcpy(d_input, pInput, bytes, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(e);
    
    printf("KERNEL\n");
    const size_t BLOCK_SIZE = 8;
    dim3 blocks(DIVUP(size.width, BLOCK_SIZE), DIVUP(size.height, BLOCK_SIZE), DIVUP(size.depth, BLOCK_SIZE));
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    CopyKernel<<<blocks,threads>>>(d_input, d_output, size);
    cudaDeviceSynchronize();

    printf("D2H\n");
    e = cudaMemcpy(pOutput, d_output, bytes, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(e);

    for (size_t z = 0; z < size.depth; z++)
    for (size_t y = 0; y < size.height; y++)
    {
        for (size_t x = 0; x < size.width; x++)
        {
            float a = input.Get(x,y,z);
            float b = output.Get(x,y,z);

            //printf("%f ", b);
            if (fabsf(fabsf(a) - fabsf(b)) > 0.01f)
            {
                printf("TEST CUDA ERROR %f %f\n", a, b);
            }
        }
        //printf("\n");
    }

    printf("TEST END\n");

    free(pInput);
    free(pOutput);

    cudaFree(d_input);
    cudaFree(d_output);
}
