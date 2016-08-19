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
    float minDistance = feature.sqDistanceTo(m_clusters[0].feature) + m_gamma * delta(classLabel, m_clusters[0].label);
    size_t minCluster = 0;
    for (size_t j = 1; j < m_clusters.size(); ++j)
    {
        float distance = feature.sqDistanceTo(m_clusters[j].feature) + m_gamma * delta(classLabel, m_clusters[j].label);
        if (distance < minDistance)
        {
            minDistance = distance;
            minCluster = j;
        }
    }
    return minCluster;
}
