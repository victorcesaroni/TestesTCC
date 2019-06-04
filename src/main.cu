#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "common.h"

template <typename T>
class InputGenerator
{
  public:    
    static void Generate(const char *filePath, cudaExtent size)
    {
        FILE *file = fopen(filePath, "w+");
        size_t fileSize = size.depth * size.height * size.width * sizeof(T);

        T *data = NULL;  
        MemoryManager::AllocGrayScaleImageCPU<T>(&data, size);
        MatrixAccessor<T> mat(data, size);

        memset(data, 0, fileSize);

        for (size_t z = 0; z < size.depth; z++)
        {
            for (size_t y = 0; y < size.height; y++)
            {
                for (size_t x = 0; x < size.width; x++)
                {
                    if (x % 2 == 0)
                        mat.Get(x, y, z) = x+y+1;
                }
            }
        }

        fwrite(data, fileSize, 1, file);

        fclose(file);
        MemoryManager::FreeCPU(data);
    }
};

typedef struct MaxFilterParams_t
{
    int radius;
} MaxFilterParams;

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

int main(int argc, const char *argv[])
{
    //nvcc main.cu --expt-extended-lambda --std=c++11 -Xcompiler -fopenmp --compiler-options -fPIC -O3 -D__INTEL_COMPILER -o main
    //nvcc main.cu -gencode arch=compute_61,code=[sm_61,compute_61] --expt-extended-lambda --std=c++11 -Xcompiler -fopenmp --compiler-options -fPIC -O3 -D__INTEL_COMPILER -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -o main
    
    TestCUDA();

    InputGenerator<float>::Generate("test.b", make_cudaExtent(10, 10, 1));

    const char *inputPath = argv[1];
    const char *outputPath = argv[2];
    size_t width = atoi(argv[3]);
    size_t height = atoi(argv[4]);
    size_t depth = atoi(argv[5]);

    cudaExtent size = make_cudaExtent(width, height, depth);

    float *input = NULL;
    float *output = NULL;
    MemoryManager::AllocGrayScaleImageCPU<float>(&input, size);
    MemoryManager::AllocGrayScaleImageCPU<float>(&output, size);
    
    FileManager::ReadAs<unsigned short, float>(input, inputPath, size);

    bool gpu = true;

    CudaTimer timer;
    printf("MaxFilter\n");
    MaxFilterParams params;
    params.radius = 8;

    timer.start();

    if (gpu)
    {
        float *d_input = NULL;
        float *d_output = NULL;
        MemoryManager::AllocGrayScaleImageGPU<float>(&d_input, size);
        MemoryManager::AllocGrayScaleImageGPU<float>(&d_output, size);
        MemoryManager::CopyGrayScaleImageToGPU<float>(input, d_input, size);

        MaxFilterParallel<float>(d_input, d_output, size, params, true);

        MemoryManager::CopyGrayScaleImageFromGPU<float>(output, d_output, size);
        MemoryManager::FreeGPU(d_input);
        MemoryManager::FreeGPU(d_output);
    }
    else
    {
        MaxFilterParallel<float>(input, output, size, params, false);        
    }
    
    timer.stop();

    printf("MaxFilter %fms\n", timer.elapsedMs);

    FileManager::Write<float>(output, outputPath, size);

    MemoryManager::FreeCPU(input);
    MemoryManager::FreeCPU(output);    
}
