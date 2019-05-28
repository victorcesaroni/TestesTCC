#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "common.h"

class MemoryManager
{
  public:
    template <typename T>
    static void AllocGrayScaleImage(T **ptr, cudaExtent size)
    {
        size_t fileSize = size.depth * size.height * size.width * sizeof(T);
        (*ptr) = (T *)malloc(fileSize);
    }

    static void Free(void *ptr)
    {
        free(ptr);
    }
};

class FileManager
{
  public:
    template <typename T>
    static void Read(T **ptr, const char *filePath, cudaExtent size)
    {
        FILE *file = fopen(filePath, "r");
        size_t fileSize = size.depth * size.height * size.width * sizeof(T);

        size_t read = fread((*ptr), fileSize, 1, file);
        
        fclose(file);
    }

    template <typename T>
    static void Write(T *ptr, const char *filePath, cudaExtent size)
    {
        FILE *file = fopen(filePath, "w+");
        size_t fileSize = size.depth * size.height * size.width * sizeof(T);

        size_t write = fwrite(ptr, fileSize, 1, file);

        fclose(file);
    }
};

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

int main(int argc, const char *argv[])
{
    //nvcc main.cu -Xcompiler -fopenmp --compiler-options -fPIC -O3 -D__INTEL_COMPILER -o ../bin/3d

    cudaExtent size = make_cudaExtent(10, 10, 1);

    InputGenerator<float>::Generate("test.b", size);

    float *input = NULL;
    float *output = NULL;
    MemoryManager::AllocGrayScaleImage<float>(&input, size);
    MemoryManager::AllocGrayScaleImage<float>(&output, size);

    FileManager::Read<float>(&input, "test.b", size);

    MaxFilterParams params;
    params.radius = 2;
    MaxFilterCPU(input, output, size, params);

    FileManager::Write<float>(output, "test2.b", size);

    MemoryManager::Free(input);
    MemoryManager::Free(output);
}
