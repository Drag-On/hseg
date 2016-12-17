//
// Created by jan on 18.08.16.
//

#include "Inference/k-prototypes/Feature.h"

Feature::Feature()
{
    m_features = Eigen::Matrix<Cost, 5, 1>::Zero();
}
