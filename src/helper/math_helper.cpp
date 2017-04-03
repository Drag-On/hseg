/**********************************************************
 * @file   math_helper.cpp
 * @author jan
 * @date   03.04.17
 * ********************************************************
 * @brief
 * @details
 **********************************************************/

#include "helper/math_helper.h"

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

cublasHandle_t handle;
float* d_one, *d_neg_one, *d_zero;

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown error";
}
#endif

namespace helper
{
    namespace math
    {
        bool init()
        {
#ifdef USE_CUDA
            auto result = cublasCreate(&handle);
            if(result == CUBLAS_STATUS_SUCCESS)
            {
                auto pmStatus = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
                if(pmStatus != CUBLAS_STATUS_SUCCESS)
                    std::cerr << "cublasSetPointerMode: " << cublasGetErrorString(result) << std::endl;
                float zero = 0, one = 1, neg_one = -1;
                cudaMalloc(&d_one, sizeof(float));
                cudaMalloc(&d_neg_one, sizeof(float));
                cudaMalloc(&d_zero, sizeof(float));
                cudaMemcpy(d_one, &one, sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_neg_one, &neg_one, sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_zero, &zero, sizeof(float), cudaMemcpyHostToDevice);
                return true;
            }
            else
            {
                std::cerr << "cublasCreate: " << cublasGetErrorString(result) << std::endl;
                return false;
            }
#else
            return true;
#endif
        }

        bool destroy()
        {
#ifdef USE_CUDA
            cudaFree(d_one);
            cudaFree(d_neg_one);
            cudaFree(d_zero);
            return cublasDestroy(handle)  == CUBLAS_STATUS_SUCCESS;
#else
            return true;
#endif
        }

        Eigen::VectorXf multiplyMatVec(Eigen::MatrixXf const& m, Eigen::VectorXf const& v)
        {
#ifdef USE_CUDA
            // Allocate gpu memory
            float* d_M, *d_V, *d_R;
            cudaMalloc(&d_M, m.rows() * m.cols() * sizeof(float));
            cudaMalloc(&d_V, v.rows() * sizeof(float));
            cudaMalloc(&d_R, m.rows() * sizeof(float));
            cudaMemcpy(d_M, m.data(), m.rows() * m.cols() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_V, v.data(), v.rows() * sizeof(float), cudaMemcpyHostToDevice);

            // Compute
            auto status = cublasSgemv(handle, CUBLAS_OP_N, m.rows(), m.cols(), d_one, d_M, m.rows(), d_V, 1, d_zero, d_R, 1);
            if(status != CUBLAS_STATUS_SUCCESS)
                std::cerr << cublasGetErrorString(status) << std::endl;

            // Copy result back
            Eigen::VectorXf result(m.rows());
            cudaMemcpy(result.data(), d_R, v.rows() * sizeof(float), cudaMemcpyDeviceToHost);

            //Free GPU memory
            cudaFree(d_M);
            cudaFree(d_V);
            cudaFree(d_R);
            return result;
#else
            return m * v;
#endif
        }

        float mahalanobis(Eigen::VectorXf const& v1, Eigen::VectorXf const& v2, Eigen::MatrixXf const& cov)
        {
#ifdef USE_CUDA
            // Allocate gpu memory
            float* d_cov, *d_v1, *d_v2, *d_result;
            cudaMalloc(&d_cov, cov.rows() * cov.cols() * sizeof(float));
            cudaMalloc(&d_v1, v1.rows() * sizeof(float));
            cudaMalloc(&d_v2, v2.rows() * sizeof(float));
            cudaMalloc(&d_result, sizeof(float));
            cudaMemcpy(d_cov, cov.data(), cov.rows() * cov.cols() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_v1, v1.data(), v1.rows() * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_v2, v2.data(), v2.rows() * sizeof(float), cudaMemcpyHostToDevice);

            // Compute vector difference: v1 = -1 * v2 + v1
            auto status = cublasSaxpy(handle, v1.rows(), d_neg_one, d_v2, 1, d_v1, 1);
            if(status != CUBLAS_STATUS_SUCCESS)
                std::cerr << cublasGetErrorString(status) << std::endl;

            // Compute cov * diff: v2 = 1 * cov * v1 + 0 * v2
            status = cublasSgemv(handle, CUBLAS_OP_N, cov.rows(), cov.cols(), d_one, d_cov, cov.rows(), d_v1, 1, d_zero, d_v2, 1);
            if(status != CUBLAS_STATUS_SUCCESS)
                std::cerr << cublasGetErrorString(status) << std::endl;

            // Compute diff^T * (cov * diff): result = v2^T * v1
            status = cublasSdot(handle, v1.rows(), d_v2, 1, d_v1, 1, d_result);
            if(status != CUBLAS_STATUS_SUCCESS)
                std::cerr << cublasGetErrorString(status) << std::endl;

            // Copy result back
            float result = 0;
            cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

            //Free GPU memory
            cudaFree(d_cov);
            cudaFree(d_v1);
            cudaFree(d_v2);
            cudaFree(d_result);
            return result;
#else
            auto diff = v1 - v2;
            return diff.transpose() * cov * diff;
#endif
        }
    }
}