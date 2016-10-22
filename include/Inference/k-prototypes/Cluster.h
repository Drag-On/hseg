//
// Created by jan on 18.08.16.
//

#ifndef HSEG_CLUSTER_H
#define HSEG_CLUSTER_H

#include <Image/Image.h>
#include "Feature.h"

class EnergyFunction;

/**
 * Stores the mean position and color and the dominant label of a cluster
 */
struct Cluster
{
    explicit Cluster(EnergyFunction const* energy);

    Feature mean; //< Mean feature
    Feature accumFeature; //< Accumulated feature. Divide by size to get the current mean!
    Feature accumSqFeature; //< Accumulated squared feature.
    Label label = 0; //< Dominant class label
    std::vector<Label> labelFrequencies; //< Label frequencies inside the cluster
    size_t size = 0; //< Amount of attached pixels
    EnergyFunction const* pEnergy;

    /**
     * Recomputes the mean feature based on accumulated features and cluster size
     */
    void updateMean();

    /**
     * Updates the dominant label based on the label frequencies
     */
    void updateLabel();
};

#endif //HSEG_CLUSTER_H
