#include <cuda.h>
#include <cuda_runtime.h>

class InputGenerator
{
  public:
    static void Generate(const char *filePath, cudaExtent size);
};
