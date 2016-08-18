//
// Created by jan on 18.08.16.
//

#include "k-prototypes/Cluster.h"

Cluster::Cluster(size_t numClasses)
{
    labelFrequencies.resize(numClasses, 0);
}