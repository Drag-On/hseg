//
// Created by jan on 22.01.17.
//

#include <gtest/gtest.h>
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <Eigen/Dense>
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

    dim3 threadsPerBlock(32);
    dim3 numBlocks(std::ceil(dim / (float)threadsPerBlock.x));
    helper::cuda::innerProductKernel<<<numBlocks, threadsPerBlock>>>(v1_dev, v1_dev + 1, v1_dev + 1, dim);

    float result = 0;
    cudaMemcpy(&result, v1_dev,sizeof(float),cudaMemcpyDeviceToHost);

    EXPECT_EQ(1*1 + 2*2 + 3*3, result);

    helper::cuda::innerProductKernel<<<numBlocks, threadsPerBlock>>>(v1_dev, v1_dev + 1, v2_dev, dim);
    cudaMemcpy(&result, v1_dev,sizeof(float),cudaMemcpyDeviceToHost);

    EXPECT_EQ(1*4 - 2*3 + 3*0, result);

    cudaFree(v1_dev);
    cudaFree(v2_dev);
}

TEST(cuda_helper,outerProductKernel)
{
    uint32_t const dim = 3;
    Eigen::Vector3f v1;
    v1 << 1.f, 2.f, 3.f;
    Eigen::Vector3f v2;
    v2 << 3.f, -3.f, 0.f;

    float* v1_dev, *v2_dev, *result_dev;
    cudaMallocManaged(&v1_dev, dim * sizeof(float));
    cudaMallocManaged(&v2_dev, dim * sizeof(float));
    cudaMallocManaged(&result_dev, dim * dim * sizeof(float));
    thrust::copy(v1.data(), v1.data() + dim, v1_dev);
    thrust::copy(v2.data(), v2.data() + dim, v2_dev);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(std::ceil(dim / (float)threadsPerBlock.x), std::ceil(dim / (float)threadsPerBlock.y));
    helper::cuda::outerProductKernel<<<numBlocks, threadsPerBlock>>>(result_dev, v1_dev, v2_dev, dim);

    Eigen::Matrix3f result;
    cudaMemcpy(result.data(), result_dev, dim*dim*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(v1_dev);
    cudaFree(v2_dev);
    cudaFree(result_dev);

    Eigen::Matrix3f eigenResult = v1 * v2.transpose();
    EXPECT_EQ(eigenResult, result);
}

TEST(cuda_helper,sumKernel)
{
    uint32_t const dim = 3;
    Eigen::Vector3f v1;
    v1 << 1.f, 2.f, 3.f;
    Eigen::Vector3f v2;
    v2 << 3.f, -3.f, 0.f;

    float* v1_dev, *v2_dev, *result_dev;
    cudaMallocManaged(&v1_dev, dim * sizeof(float));
    cudaMallocManaged(&v2_dev, dim * sizeof(float));
    cudaMallocManaged(&result_dev, dim * sizeof(float));
    thrust::copy(v1.data(), v1.data() + dim, v1_dev);
    thrust::copy(v2.data(), v2.data() + dim, v2_dev);

    dim3 threadsPerBlock(32);
    dim3 numBlocks(std::ceil(dim / (float)threadsPerBlock.x));
    helper::cuda::sumKernel<<<numBlocks, threadsPerBlock>>>(result_dev, v1_dev, v2_dev, dim);

    Eigen::Vector3f result_host;
    cudaMemcpy(result_host.data(), result_dev, dim*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(v1_dev);
    cudaFree(v2_dev);
    cudaFree(result_dev);

    Eigen::Vector3f diff = v1 + v2;
    EXPECT_EQ(diff, result_host);
}

TEST(cuda_helper,differenceKernel)
{
    uint32_t const dim = 3;
    Eigen::Vector3f v1;
    v1 << 1.f, 2.f, 3.f;
    Eigen::Vector3f v2;
    v2 << 3.f, -3.f, 0.f;

    float* v1_dev, *v2_dev, *result_dev;
    cudaMallocManaged(&v1_dev, dim * sizeof(float));
    cudaMallocManaged(&v2_dev, dim * sizeof(float));
    cudaMallocManaged(&result_dev, dim * sizeof(float));
    thrust::copy(v1.data(), v1.data() + dim, v1_dev);
    thrust::copy(v2.data(), v2.data() + dim, v2_dev);

    dim3 threadsPerBlock(32);
    dim3 numBlocks(std::ceil(dim / (float)threadsPerBlock.x));
    helper::cuda::differenceKernel<<<numBlocks, threadsPerBlock>>>(result_dev, v1_dev, v2_dev, dim);

    Eigen::Vector3f result_host;
    cudaMemcpy(result_host.data(), result_dev, dim*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(v1_dev);
    cudaFree(v2_dev);
    cudaFree(result_dev);

    Eigen::Vector3f diff = v1 - v2;
    EXPECT_EQ(diff, result_host);
}

TEST(cuda_helper,mahalanobisDistance)
{
    uint32_t const dim = 3;
    Eigen::Vector3f v1;
    v1 << 1.f, 2.f, 3.f;
    Eigen::Vector3f v2;
    v2 << 3.f, -3.f, 0.f;
    Eigen::Matrix3f mat;
    mat << 12.f, 1.f, 2.f,
            1.f, 9.f, 0.5f,
            2.f, 0.5f, 10.f;

    float* v1_dev, *v2_dev, *mat_dev, *result_dev;
    cudaMallocManaged(&v1_dev, dim * sizeof(float));
    cudaMallocManaged(&v2_dev, dim * sizeof(float));
    cudaMallocManaged(&mat_dev, dim * dim * sizeof(float));
    cudaMallocManaged(&result_dev, 1 * sizeof(float));
    thrust::copy(v1.data(), v1.data() + dim, v1_dev);
    thrust::copy(v2.data(), v2.data() + dim, v2_dev);
    thrust::copy(mat.data(), mat.data() + dim * dim, mat_dev);

    helper::cuda::mahalanobisDistance(result_dev, v1_dev, v2_dev, mat_dev, dim);

    float result_host;
    cudaMemcpy(&result_host, result_dev, 1*sizeof(float), cudaMemcpyDeviceToHost);

    Eigen::Vector3f diff = v1 - v2;
    float eigenResult = diff.transpose() * mat * diff;
    EXPECT_EQ(eigenResult, result_host);


    cudaFree(v1_dev);
    cudaFree(v2_dev);
    cudaFree(mat_dev);
    cudaFree(result_dev);
}