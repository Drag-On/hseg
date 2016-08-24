//
// Created by jan on 18.08.16.
//

#ifndef HSEG_CLUSTER_H
#define HSEG_CLUSTER_H

#include <Image/Image.h>
#include "Feature.h"

/**
 * Stores the mean position and color and the dominant label of a cluster
 */
struct Cluster
{
    explicit Cluster(size_t numClasses);

    Feature mean; //< Mean feature
    Feature accumFeature; //< Accumulated feature. Divide by size to get the current mean!
    Feature variance; //< Feature variance of the cluster
    Feature accumSqFeature; //< Accumulated squared feature.
    Label label; //< Dominant class label
    std::vector<Label> labelFrequencies; //< Label frequencies inside the cluster
    size_t size = 0; //< Amount of attached pixels

    /**
     * Recomputes the mean feature based on accumulated features and cluster size
     */
    void updateMean();

    /**
     * Recomputes the variance based on the accumulated squared features and cluster size
     */
    void updateVariance();

    /**
     * Updates the dominant label based on the label frequencies
     */
    void updateLabel();
};

#endif //HSEG_CLUSTER_H
