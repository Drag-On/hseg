//
// Created by jan on 18.08.16.
//

#ifndef HSEG_CLUSTERER_H
#define HSEG_CLUSTERER_H

#include <random>
#include <chrono>
#include <iostream>
#include <helper/coordinate_helper.h>
#include "Cluster.h"

/**
 * Clusters an image based on color, position, and a given labeling
 */
class Clusterer
{
public:
    /**
     * Runs the clustering process
     * @param numClusters Amount of clusters to generate.
     * @param numLabels Amount of class labels that can occur on the label image
     * @param color Color image
     * @param labels Label image
     * @tparam T Type of the image data
     * @details The resulting clustering will never have more clusters than specified, but it might well not use all
     *          of them.
     */
    template<typename T>
    void run(size_t numClusters, size_t numLabels, ColorImage<T> const& color, LabelImage const& labels);

    /**
     * @return Clustering result
     */
    LabelImage const& clustership() const;

    /**
     * Computes the energy of the current configuration given a color image and a label image
     * @param color Color image
     * @param labels Label image
     * @return Energy of the current configuration
     */
    template<typename T>
    float computeEnergy(ColorImage<T> const& color, LabelImage const& labels) const;

private:
    std::vector<Cluster> m_clusters;
    LabelImage m_clustership;
    float m_gamma = 100.f; // TODO: Find good mixing coefficient
    float m_conv = 0.001f; // Percentage of pixels that may change in one iteration for the algorithm to terminate

    inline float delta(Label l1, Label l2) const
    {
        if(l1 == l2)
            return 0;
        else
            return 1;
    }

    template<typename T>
    void initPrototypes(ColorImage<T> const& color, LabelImage const& labels);

    template<typename T>
    void allocatePrototypes(ColorImage<T> const& color, LabelImage const& labels);

    template<typename T>
    int reallocatePrototypes(ColorImage<T> const& color, LabelImage const& labels);

    size_t findClosestCluster(Feature const& feature, Label classLabel) const;

    float computeDistance(Feature const& feature, Label label, size_t clusterIdx) const;
};

template<typename T>
void Clusterer::run(size_t numClusters, size_t numLabels, ColorImage<T> const& color, LabelImage const& labels)
{
    assert(color.pixels() == labels.pixels());

    m_clusters.resize(numClusters, Cluster(numLabels));
    m_clustership = LabelImage(color.width(), color.height());

    initPrototypes(color, labels);
    allocatePrototypes(color, labels);

    float energy = computeEnergy(color, labels);
    std::cout << "Energy after cluster allocation: " << energy << std::endl;

    int moves = color.pixels(), lastMoves;
    int iter = 0;
    do
    {
        iter++;
        lastMoves = moves;
        moves = reallocatePrototypes(color, labels);

        //std::cout << iter << ": moves: " << moves << ", diff: " << std::abs(lastMoves - moves) << ", threshold: " << rgb.pixels() * m_conv << std::endl;
    } while (std::abs(lastMoves - moves) > color.pixels() * m_conv);

    std::cout << "Converged after " << iter << " iterations (up to convergence criterion)" << std::endl;
    energy = computeEnergy(color, labels);
    std::cout << "Energy: " << energy << std::endl;
    long int lostClusters = std::count_if(m_clusters.begin(), m_clusters.end(), [](Cluster const& c) {return c.size == 0;});
    std::cout << lostClusters << " clusters are empty." << std::endl;
}

template<typename T>
void Clusterer::initPrototypes(ColorImage<T> const& color, LabelImage const& labels)
{
    // Randomly select k objects as initial prototypes
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<size_t> distribution(0, color.pixels() - 1);
    for (auto& c : m_clusters)
    {
        if(c.size > 0)
            continue;

        size_t site = distribution(generator);
        c.feature = Feature(color, site);
        auto coords = helper::coord::siteTo2DCoordinate(site, color.width());
        c.label = labels.at(coords.first, coords.second, 0);
    }
}

template<typename T>
void Clusterer::allocatePrototypes(ColorImage<T> const& color, LabelImage const& labels)
{
    for (size_t i = 0; i < color.pixels(); ++i)
    {
        Feature curFeature(color, i);
        Label curLabel = labels.atSite(i);

        // Compute closest cluster
        size_t minCluster = findClosestCluster(curFeature, curLabel);

        // Assign to cluster
        m_clustership.atSite(i) = minCluster;
        m_clusters[minCluster].size++;
        m_clusters[minCluster].accumFeature += curFeature;
        m_clusters[minCluster].updateFeature();
        m_clusters[minCluster].labelFrequencies[curLabel]++;
        m_clusters[minCluster].updateLabel();
    }
}

template<typename T>
int Clusterer::reallocatePrototypes(ColorImage<T> const& color, LabelImage const& labels)
{
    int moves = 0;
    for (size_t i = 0; i < color.pixels(); ++i)
    {
        Feature curFeature(color, i);
        Label curLabel = labels.atSite(i);

        // Compute closest cluster
        size_t minCluster = findClosestCluster(curFeature, curLabel);
        size_t oldCluster = m_clustership.atSite(i);

        if (oldCluster != minCluster)
        {
            moves++;
            m_clustership.atSite(i) = minCluster;
            m_clusters[minCluster].size++;
            m_clusters[oldCluster].size--;
            m_clusters[minCluster].accumFeature += curFeature;
            m_clusters[oldCluster].accumFeature -= curFeature;
            m_clusters[minCluster].updateFeature();
            m_clusters[oldCluster].updateFeature();
            m_clusters[minCluster].labelFrequencies[curLabel]++;
            m_clusters[oldCluster].labelFrequencies[curLabel]--;
            m_clusters[minCluster].updateLabel();
            m_clusters[oldCluster].updateLabel();
        }
    }

    // This re-initializes clusters that have size 0
    //initPrototypes(color, labels);

    return moves;
}

template<typename T>
float Clusterer::computeEnergy(ColorImage<T> const& color, LabelImage const& labels) const
{
    float energy = 0;

    for (size_t i = 0; i < m_clustership.pixels(); ++i)
    {
        Feature f(color, i);
        Label l = labels.atSite(i);
        size_t j = m_clustership.atSite(i);
        float pxEnergy = computeDistance(f, l, j);
        energy += pxEnergy;
    }

    return energy;
}

#endif //HSEG_CLUSTERER_H
