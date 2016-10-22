//
// Created by jan on 18.08.16.
//

#include "Inference/k-prototypes/Cluster.h"
#include <Energy/EnergyFunction.h>

Cluster::Cluster(EnergyFunction const* energy)
        : pEnergy(energy)
{
    labelFrequencies.resize(energy->numClasses(), 0);
}

void Cluster::updateMean()
{
    mean = accumFeature / size;
}

void Cluster::updateLabel()
{
    label = static_cast<Label>(std::distance(labelFrequencies.begin(),
                                             std::max_element(labelFrequencies.begin(), labelFrequencies.end())));
}

