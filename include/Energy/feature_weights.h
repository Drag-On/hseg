//
// Created by jan on 20.10.16.
//

#ifndef HSEG_FEATURE_WEIGHTS_H
#define HSEG_FEATURE_WEIGHTS_H

#include <string>
#include <Eigen/Dense>

using Matrix5f = Eigen::Matrix<float, 5, 5>;

Matrix5f readFeatureWeights(std::string const& filename);

bool writeFeatureWeights(std::string const& filename, Matrix5f const& weights);


#endif //HSEG_FEATURE_WEIGHTS_H
