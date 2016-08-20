//
// Created by jan on 18.08.16.
//

#include "k-prototypes/Clusterer.h"

Clusterer::Clusterer(EnergyFunction const& energy)
        : m_energy(energy)
{
}

LabelImage const& Clusterer::clustership() const
{
    return m_clustership;
}

size_t Clusterer::findClosestCluster(Feature const& feature, Label classLabel) const
{
    float minDistance = m_energy.pixelToClusterDistance(feature, classLabel, m_clusters[0].feature,
                                                        m_clusters[0].label);
    size_t minCluster = 0;
    for (size_t j = 1; j < m_clusters.size(); ++j)
    {
        float distance = m_energy.pixelToClusterDistance(feature, classLabel, m_clusters[j].feature,
                                                         m_clusters[j].label);
        if (distance < minDistance)
        {
            minDistance = distance;
            minCluster = j;
        }
    }
    return minCluster;
}
