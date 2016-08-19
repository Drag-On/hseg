//
// Created by jan on 18.08.16.
//

#include "k-prototypes/Clusterer.h"


LabelImage const& Clusterer::clustership() const
{
    return m_clustership;
}

size_t Clusterer::findClosestCluster(Feature const& feature, Label classLabel) const
{
    float minDistance = computeDistance(feature, classLabel, 0);
    size_t minCluster = 0;
    for (size_t j = 1; j < m_clusters.size(); ++j)
    {
        float distance = computeDistance(feature, classLabel, j);
        if (distance < minDistance)
        {
            minDistance = distance;
            minCluster = j;
        }
    }
    return minCluster;
}

float Clusterer::computeDistance(Feature const& feature, Label label, size_t clusterIdx) const
{
    return feature.sqDistanceTo(m_clusters[clusterIdx].feature) + m_gamma * delta(label, m_clusters[clusterIdx].label);
}
