//
// Created by jan on 20.10.16.
//

#ifndef HSEG_FEATURE_WEIGHTS_H
#define HSEG_FEATURE_WEIGHTS_H

#include <string>
#include <Eigen/Dense>
#include <typedefs.h>

using Matrix5 = Eigen::Matrix<Cost, 5, 5>;

Matrix5 readFeatureWeights(std::string const& filename);

bool writeFeatureWeights(std::string const& filename, Matrix5 const& weights);


#endif //HSEG_FEATURE_WEIGHTS_H
