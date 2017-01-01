//
// Created by jan on 16.12.16.
//

#ifndef HSEG_TYPEDEFS_H
#define HSEG_TYPEDEFS_H

#include <cstdint>
#include <Eigen/Dense>

using Label = uint16_t;
using SiteId = uint32_t;
using Coord = uint32_t;
using Cost = float;
using Matrix5 = Eigen::Matrix<Cost, 5, 5>;
using Vector5 = Eigen::Matrix<Cost, 5, 1>;

#endif //HSEG_TYPEDEFS_H
