//
// Created by jan on 22.01.17.
//

#include <gtest/gtest.h>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "helper/cuda_helper.cuh"

TEST(cuda_helper,innerProductKernel)
{
    // For some reason thrust::device_vector throws lots of errors in gtest, so doing everything by hand here...
    // ...

    uint32_t const dim = 3;
    std::vector<float> v1(dim);
    v1[0] = 1.f;
    v1[1] = 2.f;
    v1[2] = 3.f;
    std::vector<float> v2(dim);
    v2[0] = 4.f;
    v2[1] = -3.f;
    v2[2] = 0.f;

    float* v1_dev, *v2_dev;
    cudaMallocManaged(&v1_dev, (dim + 1) * sizeof(float));
    cudaMallocManaged(&v2_dev, dim * sizeof(float));
    cudaMemcpy(v1_dev+1,v1.data(),dim * sizeof(float),cudaMemcpyHostToDevice);
    //cudaMemcpy(v2_dev,v2.data(),dim * sizeof(float),cudaMemcpyHostToDevice);
    thrust::copy(v2.begin(), v2.end(), v2_dev);

    float const zero = 0;
    cudaMemcpy(v1_dev, &zero, sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32);
    dim3 numBlocks(std::ceil(dim / (float)threadsPerBlock.x));
    helper::cuda::innerProductKernel<<<numBlocks, threadsPerBlock>>>(v1_dev, v1_dev + 1, v1_dev + 1, dim);

    float result = 0;
    cudaMemcpy(&result, v1_dev,sizeof(float),cudaMemcpyDeviceToHost);

    EXPECT_EQ(1*1 + 2*2 + 3*3, result);

    cudaMemcpy(v1_dev, &zero, sizeof(float), cudaMemcpyHostToDevice);
    helper::cuda::innerProductKernel<<<numBlocks, threadsPerBlock>>>(v1_dev, v1_dev + 1, v2_dev, dim);
    cudaMemcpy(&result, v1_dev,sizeof(float),cudaMemcpyDeviceToHost);

    EXPECT_EQ(1*4 - 2*3 + 3*0, result);

    cudaFree(v1_dev);
    cudaFree(v2_dev);
}