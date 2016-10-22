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
    auto computeDist = [this](Label take_l)
    {
        float dist = 0;
        for(Label l = 0; l < pEnergy->numClasses(); ++l)
        {
            if(l != take_l)
                dist += labelFrequencies[l] * pEnergy->classDistance(l, take_l);
        }
        return dist;
    };

    Label bestLabel = 0;
    float bestDist = computeDist(0);
    for(Label l = 1; l < pEnergy->numClasses(); ++l)
    {
        float dist = computeDist(l);
        if(dist < bestDist)
        {
            bestDist = dist;
            bestLabel = l;
        }
    }

    label = bestLabel;
}

