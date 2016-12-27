//
// Created by jan on 18.08.16.
//

#ifndef HSEG_CLUSTER_H
#define HSEG_CLUSTER_H

#include <functional>
#include <typedefs.h>
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
    Label label = 0; //< Assigned class label
    std::vector<SiteId> allocated; //< Sites allocated to this cluster
    Label numClasses = 0;
    std::function<float(Label,Label)> classDistance;
    std::function<float(SiteId,Label)> classData;

    /**
     * Recomputes the mean feature based on accumulated features and cluster size
     */
    void update(std::vector<Feature> const& features, LabelImage const& labeling);
};

template<typename EnergyFun>
Cluster::Cluster(EnergyFun const* energy)
{
    numClasses = energy->numClasses();
    classDistance = [energy](Label l1, Label l2) -> float {return energy->classDistance(l1, l2);};
    classData = [energy](SiteId s, Label l) -> float {return energy->classData(s, l);};
}

#endif //HSEG_CLUSTER_H
