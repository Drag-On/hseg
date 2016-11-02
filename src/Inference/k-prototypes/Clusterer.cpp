//
// Created by jan on 18.08.16.
//

#include "Inference/k-prototypes/Clusterer.h"

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
    std::vector<float> distances(m_clusters.size(), 0.f);
    size_t i = 0;
    std::generate(distances.begin(), distances.end(), [&]{return m_energy.pixelToClusterDistance(feature, classLabel, m_clusters, i++);});
    return std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
}

std::vector<Cluster> const& Clusterer::clusters() const
{
    return m_clusters;
}
