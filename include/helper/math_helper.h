/**********************************************************
 * @file   math_helper.h
 * @author jan
 * @date   03.04.17
 * ********************************************************
 * @brief
 * @details
 **********************************************************/
#ifndef HSEG_MATH_HELPER_H
#define HSEG_MATH_HELPER_H

#include <Eigen/Dense>

namespace helper
{
    namespace math
    {
        bool init();

        bool destroy();

        /**
         * Matrix-Vector product
         * @param m Matrix
         * @param v Vector
         * @return Result vector
         */
        Eigen::VectorXf multiplyMatVec(Eigen::MatrixXf const& m, Eigen::VectorXf const& v);

        /**
         * Computes the squared mahalanobis distance between \p v1 and \p v2 under inverse covariance matrix \p cov
         * @param v1 First vector
         * @param v2 Second vector
         * @param cov Inverse covariance matrix
         * @return The distance
         */
        float mahalanobis(Eigen::VectorXf const& v1, Eigen::VectorXf const& v2, Eigen::MatrixXf const& cov);
    }
}

#endif //HSEG_MATH_HELPER_H
