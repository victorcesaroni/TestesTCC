#include "common.h"
#include "misc.h"
#include "memory_manager.h"
#include "matrix_accessor.h"

void InputGenerator::Generate(const char *filePath, cudaExtent size)
{
    FILE *file = fopen(filePath, "w+");
    size_t fileSize = size.depth * size.height * size.width * sizeof(float);

    float *data = NULL;  
    MemoryManager::AllocGrayScaleImageCPU<float>(&data, size);
    MatrixAccessor<float> mat(data, size);

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


