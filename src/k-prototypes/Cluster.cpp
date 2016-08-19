//
// Created by jan on 18.08.16.
//

#include "k-prototypes/Cluster.h"

Cluster::Cluster(size_t numClasses)
{
    labelFrequencies.resize(numClasses, 0);
}

void Cluster::updateFeature()
{
    feature = accumFeature / size;
}

void Cluster::updateLabel()
{
    label = std::distance(labelFrequencies.begin(), std::max_element(labelFrequencies.begin(), labelFrequencies.end()));
}
