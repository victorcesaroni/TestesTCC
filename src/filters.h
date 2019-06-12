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
#define GET_FEATURE_Z_INDEX(zOriginal, numFeatures, featureIndex) (zOriginal * (size_t)numFeatures + (size_t)featureIndex)

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

////////////////////////////

// convert from degree to radians (a/180)*pi
#define DEG_TO_RAD(a) (0.017453292 * a) 

// membrane_projections settings
#define MP_ROTATION        180.0f
#define MP_ROTATION_STEP   6.0f // we are assuming the 180 deg rotation step is 6 deg
#define MP_NUM_ROTATIONS   (int)(MP_ROTATION/MP_ROTATION_STEP)
#define MP_NUM_FILTERS     4

class MembraneProjectionsFilterParams
{
public:
    float2 directions[MP_NUM_ROTATIONS]; // unit vector used to simulate the rotation of the kernel
    int kernelRadius; // simulated kernel radius
    int kernelWidth; // simulated central column width

    MembraneProjectionsFilterParams(int kernelRadius, int kernelWidth)
    {
        int idx = 0;
        for (float a = 0; a < MP_ROTATION; a += MP_ROTATION_STEP)
        {
            // create the rotation vectors
            this->directions[idx].x = cosf(DEG_TO_RAD(a));
            this->directions[idx].y = sinf(DEG_TO_RAD(a));
            idx++;
        }

        this->kernelRadius = kernelRadius;
        this->kernelWidth = kernelWidth;
    }

    cudaExtent GetOutputSize(cudaExtent inputSize)
    {
        return make_cudaExtent(inputSize.width, inputSize.height, inputSize.depth * MP_NUM_FILTERS);
    }
};

void MembraneProjectionsCPU(float *pInput, float *pOutput, cudaExtent size, MembraneProjectionsFilterParams params);
void MembraneProjectionsParallel(float *pInput, float *pOutput, cudaExtent size, MembraneProjectionsFilterParams params, bool gpu);

#endif
