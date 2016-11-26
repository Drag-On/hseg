//
// Created by jan on 18.08.16.
//

#ifndef HSEG_CLUSTER_H
#define HSEG_CLUSTER_H

#include <Image/Image.h>
#include <functional>
#include "Feature.h"

class EnergyFunction;

/**
 * Stores the mean position and color and the dominant label of a cluster
 */
struct Cluster
{
    template<typename EnergyFun>
    explicit Cluster(EnergyFun const* energy);

    Feature mean; //< Mean feature
    Feature accumFeature; //< Accumulated feature. Divide by size to get the current mean!
    Feature accumSqFeature; //< Accumulated squared feature.
    Label label = 0; //< Dominant class label
    std::vector<Label> labelFrequencies; //< Label frequencies inside the cluster
    std::vector<Label> additionalLabelFrequencies; //< Used to compute additional cost in training
    size_t size = 0; //< Amount of attached pixels
    size_t numClasses = 0;
    std::function<float(Label,Label)> classDistance;
    std::function<float(Label,Label)> additionalClassDistance;

    /**
     * Recomputes the mean feature based on accumulated features and cluster size
     */
    void updateMean();

    /**
     * Updates the dominant label based on the label frequencies
     */
    void updateLabel();
};

template<typename EnergyFun>
Cluster::Cluster(EnergyFun const* energy)
{
    numClasses = energy->numClasses();
    labelFrequencies.resize(energy->numClasses(), 0);
    additionalLabelFrequencies.resize(energy->numClasses(), 0);
    classDistance = [energy](Label l1, Label l2) -> float {return energy->classDistance(l1, l2);};
    additionalClassDistance = [energy](Label l1, Label l2) -> float {return energy->additionalClassLabelCost(l1, l2);};
}

#endif //HSEG_CLUSTER_H
