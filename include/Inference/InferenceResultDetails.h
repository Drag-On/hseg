//
// Created by jan on 28.08.16.
//

#ifndef HSEG_INFERENCERESULTDETAILS_H
#define HSEG_INFERENCERESULTDETAILS_H

#include <Image/Image.h>
#include "Cluster.h"

/**
 * Stores detailed results from inference
 */
struct InferenceResultDetails
{
public:
    uint32_t numIter = 0; //< Amount of iterations that have been done
    std::vector<LabelImage> labelings; //< Class labeling after each iteration
    std::vector<LabelImage> clusterings; //< Superpixel segmentation after each iteration
    std::vector<FeatureImage> marginals; //< Marginals after each iteration
    std::vector<std::vector<Cluster>> clusters; // Clusters after each iteration
    std::vector<Cost> energy; //< Energy before every iteration. This vector is one element longer than the others
};

#endif //HSEG_INFERENCERESULTDETAILS_H
