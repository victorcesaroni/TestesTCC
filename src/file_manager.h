#include "common.h"

class FileManager
{
  public:
    template <typename T>
    static void Read(T *ptr, const char *filePath, cudaExtent size)
    {
        size_t fileSize = size.depth * size.height * size.width * sizeof(T);

        FILE *file = fopen(filePath, "r");
        size_t read = fread(ptr, fileSize, 1, file);        
        fclose(file);
    }

    template <typename original_t, typename as_t>
    static void ReadAs(as_t *ptr, const char *filePath, cudaExtent size)
    {
        size_t fileSize = size.depth * size.height * size.width * sizeof(original_t);
        unsigned char *tmp = (unsigned char*)malloc(fileSize);

        FILE *file = fopen(filePath, "r");
        size_t read = fread(tmp, fileSize, 1, file);
        fclose(file);

        size_t j = 0;
        for (size_t i = 0; i < fileSize; i += sizeof(original_t))
            ptr[j++] = (as_t)(*(original_t*)&tmp[i]);        

        free(tmp);
    }

    template <typename T>
    static void Write(T *ptr, const char *filePath, cudaExtent size)
    {
        size_t fileSize = size.depth * size.height * size.width * sizeof(T);

        FILE *file = fopen(filePath, "w+");
        size_t write = fwrite(ptr, fileSize, 1, file);
        fclose(file);
    }
};
