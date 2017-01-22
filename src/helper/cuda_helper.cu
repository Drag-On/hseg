
#include "helper/cuda_helper.cuh"

namespace helper
{
    namespace cuda
    {
        __global__
        void innerProductKernel(float* out, float const* v1, float const* v2, uint32_t dim)
        {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

            if(i < dim)
                atomicAdd(out, v1[i] * v2[i]);
        }
    }
}