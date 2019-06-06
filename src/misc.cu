#include "common.h"
#include "misc.h"
#include "memory_manager.h"
#include "matrix_accessor.h"

void DrawCube(MatrixAccessor<float> mat, int x, int y, int z, int w, int h, int d, float color)
{
    for (size_t _z = z; _z < z+d; _z++)
    for (size_t _y = y; _y < y+h; _y++)
    for (size_t _x = x; _x < x+w; _x++)
        mat.Get(_x,_y,_z)=color;
}

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
                mat.Get(x, y, z) = (float)y + (y%2==0?z:0);
            }
        }
    }

    DrawCube(mat, size.width / 4, size.height / 4, 0, size.width/2, size.height / 4, size.depth, size.height);

    fwrite(data, fileSize, 1, file);

    fclose(file);
    MemoryManager::FreeCPU(data);
}


