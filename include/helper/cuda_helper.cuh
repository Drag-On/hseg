
#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <cmath>

namespace helper
{
    namespace cuda
    {
        /**
         * Cuda kernel that computes the inner product of two vectors, i.e. <v1, v2>.
         * @tparam T Type of the vector elements
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
        template <typename T>
        __global__
        void innerProductKernel(T* out, T const* v1, T const* v2, uint32_t dim)
        {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

            *out = 0;

            if(i < dim)
                atomicAdd(out, v1[i] * v2[i]);
        }

        /**
         * Cuda kernel that computes the outer product of two vectors, i.e. v1 v2^T
         * @tparam T Type of the vector elements
         * @param out Result matrix is stored here, size dim * dim
         * @param v1 Pointer to the first element of the first vector
         * @param v2 Pointer to the first element of the second vector
         * @param dim Dimensionality of both vectors
         *
         * @example
         *  dim3 threadsPerBlock(32, 32);
         *  dim3 numBlocks(std::ceil(dim / (float)threadsPerBlock.x), std::ceil(dim / (float)threadsPerBlock.y));
         *  outerProductKernel<<<numBlocks, threadsPerBlock>>>(&out, &v1[0], &v2[0], dim);
         */
        template <typename T>
        __global__
        void outerProductKernel(T* out, T const* v1, T const* v2, uint32_t dim)
        {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;

            if(i < dim && j < dim)
                out[i + j * dim] = v1[i] * v2[j];
        }

        /**
         * Cuda kernel that computes the difference of two vectors
         * @tparam T Type of the vector elements
         * @param out Result vector, size dim
         * @param v1 Pointer to the first element of the first vector
         * @param v2 Pointer to the first element of second first vector
         * @param dim Dimensionality of both vectors
         */
        template <typename T>
        __global__
        void differenceKernel(T* out, T const* v1, T const* v2, uint32_t dim)
        {
            uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

            if(i < dim)
                out[i] = v1[i] - v2[i];
        }

        /**
         * Computes the squared mahalanobis distance of two vectors on the GPU
         * @details This is a host-function that expects pointers to device memory. It does not transfer any memory
         *          between host and device.
         * @tparam T Type of the vector elements
         * @param out Result will be written here
         * @param v1 Pointer to the first element of the first vector
         * @param v2 Pointer to the first element of the second vector
         * @param mat Pointer to the first element of the inverse covariance matrix
         * @param dim Dimensionality of the vectors
         */
        template <typename T>
        __host__
        void mahalanobisDistance(T* out, T const* v1, T const* v2, T const* mat, uint32_t dim)
        {
            // Compute difference vector
            T* diff;
            cudaMallocManaged(&diff, dim * sizeof(T));
            dim3 threadsPerBlock1(32);
            dim3 numBlocks1(std::ceil(dim / (float)threadsPerBlock1.x));
            helper::cuda::differenceKernel<<<numBlocks1, threadsPerBlock1>>>(diff, v1, v2, dim);
            cudaDeviceSynchronize();

            // Compute outer product
            T* outer;
            cudaMallocManaged(&outer, dim * dim * sizeof(T));
            dim3 threadsPerBlock2(32, 32);
            dim3 numBlocks2(std::ceil(dim / (float)threadsPerBlock2.x), std::ceil(dim / (float)threadsPerBlock2.y));
            helper::cuda::outerProductKernel<<<numBlocks2, threadsPerBlock2>>>(outer, diff, diff, dim);
            cudaDeviceSynchronize();

            // Compute inner product
            dim3 threadsPerBlock3(32);
            dim3 numBlocks3(std::ceil(dim / (float)threadsPerBlock3.x));
            helper::cuda::innerProductKernel<<<numBlocks3, threadsPerBlock3>>>(out, mat, outer, dim * dim);
            cudaDeviceSynchronize();

            cudaFree(diff);
            cudaFree(outer);
        }
    }
}

#endif //CUDA_HELPER_CUH
