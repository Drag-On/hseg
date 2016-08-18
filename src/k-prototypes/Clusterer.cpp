//
// Created by jan on 18.08.16.
//

#include <random>
#include <chrono>
#include <iostream>
#include <helper/coordinate_helper.h>
#include "k-prototypes/Clusterer.h"

void Clusterer::run(size_t numClusters, size_t numLabels, RGBImage const& rgb, LabelImage const& labels)
{
    assert(rgb.pixels() == labels.pixels());

    m_clusters.resize(numClusters, Cluster(numLabels));
    m_clustership = LabelImage(rgb.width(), rgb.height());

    initPrototypes(rgb, labels);
    allocatePrototypes(rgb, labels);

    int moves = rgb.pixels(), lastMoves;
    int iter = 0;
    do
    {
        iter++;
        lastMoves = moves;
        moves = reallocatePrototypes(rgb, labels);

        //std::cout << iter << ": moves: " << moves << ", diff: " << std::abs(lastMoves - moves) << ", threshold: " << rgb.pixels() * m_conv << std::endl;
    } while (std::abs(lastMoves - moves) > rgb.pixels() * m_conv);

    std::cout << "Converged after " << iter << " iterations (up to convergence criterium)" << std::endl;
}

LabelImage const& Clusterer::clustership() const
{
    return m_clustership;
}

void Clusterer::initPrototypes(RGBImage const& rgb, LabelImage const& labels)
{
    // Randomly select k objects as initial prototypes
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<size_t> distribution(0, rgb.pixels() - 1);
    for (auto& c : m_clusters)
    {
        size_t site = distribution(generator);
        c.feature = c.accumFeature = Feature(rgb, site);
        auto coords = helper::coord::siteTo2DCoordinate(site, rgb.width());
        c.label = labels.at(coords.first, coords.second, 0);
    }
}

void Clusterer::allocatePrototypes(RGBImage const& rgb, LabelImage const& labels)
{
    for (size_t i = 0; i < rgb.pixels(); ++i)
    {
        Feature curFeature(rgb, i);
        Label curLabel = labels.at(i, 0);

        // Compute closest cluster
        size_t minCluster = findClosestCluster(curFeature, curLabel);

        // Assign to cluster
        m_clustership.at(i, 0) = minCluster;
        m_clusters[minCluster].size++;
        m_clusters[minCluster].accumFeature += curFeature;
        m_clusters[minCluster].feature = m_clusters[minCluster].accumFeature / m_clusters[minCluster].size;
        m_clusters[minCluster].labelFrequencies[curLabel]++;
        m_clusters[minCluster].label = std::distance(m_clusters[minCluster].labelFrequencies.begin(),
                                                     std::max_element(m_clusters[minCluster].labelFrequencies.begin(),
                                                                      m_clusters[minCluster].labelFrequencies.end()));
    }
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

int Clusterer::reallocatePrototypes(RGBImage const& rgb, LabelImage const& labels)
{
    int moves = 0;
    for (size_t i = 0; i < rgb.pixels(); ++i)
    {
        Feature curFeature(rgb, i);
        Label curLabel = labels.at(i, 0);

        // Compute closest cluster
        size_t minCluster = findClosestCluster(curFeature, curLabel);
        size_t oldCluster = m_clustership.at(i, 0);

        if (oldCluster != minCluster)
        {
            moves++;
            m_clustership.at(i, 0) = minCluster;
            m_clusters[minCluster].size++;
            m_clusters[oldCluster].size--;
            m_clusters[minCluster].accumFeature += curFeature;
            m_clusters[oldCluster].accumFeature -= curFeature;
            m_clusters[minCluster].feature = m_clusters[minCluster].accumFeature / m_clusters[minCluster].size;
            m_clusters[oldCluster].feature = m_clusters[oldCluster].accumFeature / m_clusters[oldCluster].size;
            m_clusters[minCluster].labelFrequencies[curLabel]++;
            m_clusters[oldCluster].labelFrequencies[curLabel]--;
            m_clusters[minCluster].label = std::distance(m_clusters[minCluster].labelFrequencies.begin(),
                                                         std::max_element(
                                                                 m_clusters[minCluster].labelFrequencies.begin(),
                                                                 m_clusters[minCluster].labelFrequencies.end()));
            m_clusters[oldCluster].label = std::distance(m_clusters[oldCluster].labelFrequencies.begin(),
                                                         std::max_element(
                                                                 m_clusters[oldCluster].labelFrequencies.begin(),
                                                                 m_clusters[oldCluster].labelFrequencies.end()));
        }
    }
    return moves;
}
