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

    Feature feature; //< Mean feature
    Feature accumFeature; //< Accumulated feature. Divide by size to get the current mean!
    Label label; //< Dominant class label
    std::vector<Label> labelFrequencies; //< Label frequencies inside the cluster
    size_t size = 0; //< Amount of attached pixels

    /**
     * Recomputes the mean feature based on accumulated features cluster size
     */
    void updateFeature();

    /**
     * Updates the dominant label based on the label frequencies
     */
    void updateLabel();
};

#endif //HSEG_CLUSTER_H
