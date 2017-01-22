
#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <host_defines.h>
#include <cstdint>

namespace helper
{
    namespace cuda
    {
        /**
         * Cuda kernel that computes the inner product of two vectors, i.e. <v1, v2>.
         * @param out The result is stored here
         * @param v1 Pointer to the first element of the first vector
         * @param v2 Pointer to the first element of the second vector
         * @param dim Dimensionality of both vectors
         *
         * @example
         *  dim3 threadsPerBlock(32);
         *  dim3 numBlocks(std::ceil(dim / (float)threadsPerBlock.x));
         *  innerProductKernel<<<numBlocks, threadsPerBlock>>>(&out, &v1[0], &v2[0], dim);
         */
        __global__
        void innerProductKernel(float* out, float const* v1, float const* v2, uint32_t dim);
    }
}

#endif //CUDA_HELPER_CUH
