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
        MemoryManager::AllocGrayScaleImage<T>(&data, size);
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
        MemoryManager::Free(data);
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
    {
        for (size_t y = 0; y < size.height; y++)
        {
            for (size_t x = 0; x < size.width; x++)
            {
                val_t maxVal = input.Get(x, y, z);

                for (int oy = -params.radius; oy <= params.radius; oy++)
                {
                    for (int ox = -params.radius; ox <= params.radius; ox++)
                    {
                        if (input.CheckOffsetInBounds(x, y, z, ox, oy, 0))
                            maxVal = max(maxVal, input.Get(x+ox, y+oy, z));
                    }
                }

                output.Get(x, y, z) = maxVal;
            }
        }
    }
}

template <typename val_t>
void MaxFilterCPUParallel(val_t *pInput, val_t *pOutput, cudaExtent size, MaxFilterParams params)
{
    MatrixAccessor<val_t> input(pInput, size);
    MatrixAccessor<val_t> output(pOutput, size);

    auto kernel = [&](dim3 blockIdx, dim3 blockDim, dim3 threadIdx)
    {
        size_t x = blockIdx.x * blockDim.x + threadIdx.x;
        size_t y = blockIdx.y * blockDim.y + threadIdx.y;
        size_t z = blockIdx.z * blockDim.z + threadIdx.z;

        if (x < size.width && y < size.height && z < size.depth)
        {
            val_t maxVal = input.Get(x, y, z);

            for (int oy = -params.radius; oy <= params.radius; oy++)
            {
                for (int ox = -params.radius; ox <= params.radius; ox++)
                {
                    if (input.CheckOffsetInBounds(x, y, z, ox, oy, 0))
                        maxVal = max(maxVal, input.Get(x+ox, y+oy, z));
                }
            }

            output.Get(x, y, z) = maxVal;
        }
    };

    const size_t BLOCK_SIZE = 512;
    const size_t NUM_THREADS = 10;

    dim3 blocks(DIVUP(size.width, BLOCK_SIZE), DIVUP(size.height, BLOCK_SIZE), DIVUP(size.depth, BLOCK_SIZE));

    for (size_t bz = 0; bz < blocks.z; bz++)
    for (size_t by = 0; by < blocks.y; by++)
    for (size_t bx = 0; bx < blocks.x; bx++)
    {
        dim3 blockIdx(bx, by, bz);

        for (size_t tz = 0; tz < BLOCK_SIZE; tz++)
        for (size_t ty = 0; ty < BLOCK_SIZE; ty++)
        for (size_t tx = 0; tx < BLOCK_SIZE; tx++)
        {
            dim3 threadIdx(tx, ty, tz);
            kernel(blockIdx, dim3(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), threadIdx);
        }
    }
}

int main(int argc, const char *argv[])
{
    //nvcc main.cu --std=c++11 -Xcompiler -fopenmp --compiler-options -fPIC -O3 -D__INTEL_COMPILER -o main

    InputGenerator<float>::Generate("test.b", make_cudaExtent(10, 10, 10));

    const char *inputPath = argv[1];
    const char *outputPath = argv[2];
    size_t width = atoi(argv[3]);
    size_t height = atoi(argv[4]);
    size_t depth = atoi(argv[5]);

    cudaExtent size = make_cudaExtent(width, height, depth);

    float *input = NULL;
    float *output = NULL;
    MemoryManager::AllocGrayScaleImage<float>(&input, size);
    MemoryManager::AllocGrayScaleImage<float>(&output, size);

    FileManager::ReadAs<unsigned short, float>(input, inputPath, size);

    CudaTimer timer;
    printf("MaxFilterCPU\n");
    MaxFilterParams params;
    params.radius = 8;

    timer.start();
    MaxFilterCPUParallel<float>(input, output, size, params);
    timer.stop();

    printf("MaxFilterCPU %fms\n", timer.elapsedMs);

    FileManager::Write<float>(output, outputPath, size);

    MemoryManager::Free(input);
    MemoryManager::Free(output);
}
