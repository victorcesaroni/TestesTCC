#ifndef FILTERS_H_
#define FILTERS_H_

#include <cuda.h>
#include <cuda_runtime.h>

enum NeighboorhoodFilterType {
    NEIGHBOORHOOD_FILTER_MIN = 0,
    NEIGHBOORHOOD_FILTER_MAX = 1,
    NEIGHBOORHOOD_FILTER_MEAN = 2,
    NEIGHBOORHOOD_FILTER_VARIANCE = 3,
    NEIGHBOORHOOD_FILTER_STD_DEV = 4,
    NEIGHBOORHOOD_FILTER_COUNT,
};

#define NEIGHBOORHOOD_FILTER_MAX_SCALES 6
#define GET_FEATURE_Z_INDEX(zOriginal, numFeatures, featureIndex) (zOriginal * (size_t)numFeatures + (size_t)featureIndex);

class NeighboorhoodFilterParams
{
public:
    int numFilters;
    int numScales;
    int filters[NEIGHBOORHOOD_FILTER_COUNT]; // filter selection
    int scales[NEIGHBOORHOOD_FILTER_MAX_SCALES]; // salces selection

    cudaExtent GetOutputSize(cudaExtent inputSize)
    {
        return make_cudaExtent(inputSize.width, inputSize.height, inputSize.depth * numFilters * numScales);
    }
};

void NeighboorhoodFilterCPU(float *pInput, float *pOutput, cudaExtent size, NeighboorhoodFilterParams params);
void NeighboorhoodFilterParallel(float *pInput, float *pOutput, cudaExtent size, NeighboorhoodFilterParams params, bool gpu);

#endif
