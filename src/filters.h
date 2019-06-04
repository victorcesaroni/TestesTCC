#include <cuda.h>
#include <cuda_runtime.h>

typedef struct MaxFilterParams_t
{
    int radius;
} MaxFilterParams;

template <typename val_t>
void MaxFilterCPU(val_t *pInput, val_t *pOutput, cudaExtent size, MaxFilterParams params);

template <typename val_t>
void MaxFilterParallel(val_t *pInput, val_t *pOutput, cudaExtent size, MaxFilterParams params, bool gpu);
