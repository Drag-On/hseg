//
// Created by jan on 18.08.16.
//

#include "Inference/k-prototypes/Cluster.h"

Cluster::Cluster(size_t numClasses)
{
    labelFrequencies.resize(numClasses, 0);
}

void Cluster::updateMean()
{
    mean = accumFeature / size;
}

void Cluster::updateVariance()
{
    variance = accumSqFeature / size - mean.getSquaredElements();
}

void Cluster::updateLabel()
{
    label = static_cast<Label>(std::distance(labelFrequencies.begin(),
                                             std::max_element(labelFrequencies.begin(), labelFrequencies.end())));
}

