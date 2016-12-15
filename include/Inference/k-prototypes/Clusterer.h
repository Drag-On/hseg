//
// Created by jan on 18.08.16.
//

#ifndef HSEG_CLUSTERER_H
#define HSEG_CLUSTERER_H

#include <random>
#include <chrono>
#include <iostream>
#include <helper/coordinate_helper.h>
#include <Energy/EnergyFunction.h>
#include "Cluster.h"

/**
 * Clusters an image based on color, position, and a given labeling
 */
template<typename EnergyFun>
class Clusterer
{
public:
    /**
     * Constructor
     * @param energy Energy function to optimize with respect to the superpixel segmentation. The reference needs to
     *               stay valid as long as the clusterer exists
     */
    Clusterer(EnergyFun energy);

    /**
     * Runs the clustering process
     * @param numClusters Amount of clusters to generate.
     * @param numLabels Amount of class labels that can occur on the label image
     * @param color Color image
     * @param labels Label image
     * @returns The amount of iterations that have been done
     * @tparam T Type of the image data
     * @details The resulting clustering will never have more clusters than specified, but it might well not use all
     *          of them. If this is called multiple times on the same object, the algorithm will be warm-started with
     *          the previous result.
     */
    template<typename T>
    size_t run(size_t numClusters, size_t numLabels, ColorImage<T> const& color, LabelImage const& labels);

    /**
     * @return Clustering result
     */
    inline LabelImage const& clustership() const
    {
        return m_clustership;
    }

    template<typename T>
    static std::vector<Cluster>
    computeClusters(LabelImage const& sp, ColorImage<T> const& color, LabelImage const& labeling, size_t numClusters,
                    size_t numClasses, EnergyFun const& energy);

private:
    EnergyFun m_energy;
    std::vector<Cluster> m_clusters;
    LabelImage m_clustership;
    float const m_conv = 0.001f; // Percentage of pixels that may change in one iteration for the algorithm to terminate

    template<typename T>
    void initPrototypes(ColorImage<T> const& color, LabelImage const& labels, size_t numClusters);

    template<typename T>
    void allocatePrototypes(ColorImage<T> const& color, LabelImage const& labels);

    template<typename T>
    size_t reallocatePrototypes(ColorImage<T> const& color, LabelImage const& labels);

    size_t findClosestCluster(Feature const& feature, Label classLabel) const;

    /**
     * @return Current clusters
     */
    inline std::vector<Cluster> const& clusters() const
    {
        return m_clusters;
    }
};

template<typename EnergyFun>
Clusterer<EnergyFun>::Clusterer(EnergyFun energy)
        : m_energy(energy)
{
}

template<typename EnergyFun>
size_t Clusterer<EnergyFun>::findClosestCluster(Feature const& feature, Label classLabel) const
{
    std::vector<float> distances(m_clusters.size(), 0.f);
    size_t i = 0;
    std::generate(distances.begin(), distances.end(), [&]{return m_energy.pixelToClusterDistance(feature, classLabel, m_clusters, i++);});
    return std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
}

template<typename EnergyFun>
template<typename T>
size_t Clusterer<EnergyFun>::run(size_t numClusters, size_t /* numLabels */, ColorImage<T> const& color, LabelImage const& labels)
{
    assert(color.pixels() == labels.pixels());

    //m_clusters.resize(numClusters, Cluster(&m_energy));
    m_clusters.reserve(numClusters);
    m_clustership = LabelImage(color.width(), color.height());

    initPrototypes(color, labels, numClusters);
    allocatePrototypes(color, labels);

    //float energy = computeEnergy(color, labels);
    //std::cout << "Energy after cluster allocation: " << energy << std::endl;

    size_t moves = color.pixels(), lastMoves;
    size_t iter = 0;
    do
    {
        iter++;
        lastMoves = moves;
        moves = reallocatePrototypes(color, labels);

        //std::cout << iter << ": moves: " << moves << ", diff: " << std::abs(lastMoves - moves) << ", threshold: " << rgb.pixels() * m_conv << std::endl;
    } while (std::abs(lastMoves - moves) > color.pixels() * m_conv);

    //std::cout << "Converged after " << iter << " iterations (up to convergence criterion)" << std::endl;
    //energy = computeEnergy(color, labels);
    //std::cout << "Energy: " << energy << std::endl;
    //long int lostClusters = std::count_if(m_clusters.begin(), m_clusters.end(),
    //                                      [](Cluster const& c) { return c.size == 0; });
    //std::cout << lostClusters << " clusters are empty." << std::endl;

    return iter;
}

template<typename EnergyFun>
template<typename T>
void Clusterer<EnergyFun>::initPrototypes(ColorImage<T> const& color, LabelImage const& labels, size_t numClusters)
{
    // Randomly select a pixel as initial prototype
    std::default_random_engine generator(0/*std::chrono::system_clock::now().time_since_epoch().count()*/);
    std::uniform_int_distribution<size_t> distribution(0, color.pixels() - 1);
    std::vector<float> distances(color.pixels(), 0.f); // Distance to closest cluster center
    // Only pick this site if the label is valid. It might be invalid for clustering on ground truth images,
    // where some pixels are marked as "any label".
    while(true)
    {
        size_t const site = distribution(generator);
        Label const l = labels.atSite(site);
        if(l < m_energy.unaryFile().classes())
        {
            m_clusters.emplace_back(&m_energy);
            m_clusters.back().label = l;
            m_clusters.back().mean = Feature(color, site);
            break;
        }
    }
    // Compute the distance between each object and the newly created cluster center
    size_t i = 0;
    std::generate(distances.begin(), distances.end(), [&]
    {
        float const dist = m_energy.pixelToClusterDistance(Feature(color, i), labels.atSite(i), m_clusters, 0);
        ++i;
        return dist;
    });

    // Pick a new object as the next cluster proportional to the squared distance
    std::vector<float> weights(color.pixels(), 0.f);
    for(size_t c = 1; c < numClusters; ++c)
    {
        size_t n = 0;
        std::generate(weights.begin(), weights.end(), [&] { float const d = distances[n++]; return d * d; });
        std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
        while(true)
        {
            size_t const site = distribution(generator);
            Label const l = labels.atSite(site);
            if(l < m_energy.unaryFile().classes())
            {
                m_clusters.emplace_back(&m_energy);
                m_clusters.back().label = l;
                m_clusters.back().mean = Feature(color, site);
                break;
            }
        }
        // Recompute cluster distances
        size_t i = 0;
        std::generate(distances.begin(), distances.end(), [&]
        {
            float const dist = m_energy.pixelToClusterDistance(Feature(color, i), labels.atSite(i), m_clusters, c);
            ++i;
            return std::min(dist, distances[i - 1]);
        });
    }

}

template<typename EnergyFun>
template<typename T>
void Clusterer<EnergyFun>::allocatePrototypes(ColorImage<T> const& color, LabelImage const& labels)
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
        m_clusters[minCluster].updateMean();
        if(curLabel < m_energy.unaryFile().classes())
        {
            // Only update for valid labels
            m_clusters[minCluster].labelFrequencies[curLabel]++;
            m_clusters[minCluster].updateLabel();
        }
    }
}

template<typename EnergyFun>
template<typename T>
size_t Clusterer<EnergyFun>::reallocatePrototypes(ColorImage<T> const& color, LabelImage const& labels)
{
    size_t moves = 0;
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
            m_clusters[minCluster].updateMean();
            m_clusters[oldCluster].updateMean();
            if(curLabel < m_energy.unaryFile().classes())
            {
                // Only update valid labels
                m_clusters[minCluster].labelFrequencies[curLabel]++;
                m_clusters[oldCluster].labelFrequencies[curLabel]--;
                m_clusters[minCluster].updateLabel();
                m_clusters[oldCluster].updateLabel();
            }
        }
    }

    // This re-initializes clusters that have size 0
    //initPrototypes(color, labels);

    return moves;
}

template<typename EnergyFun>
template<typename T>
std::vector<Cluster>
Clusterer<EnergyFun>::computeClusters(LabelImage const& sp, ColorImage<T> const& color, LabelImage const& labeling,
                           size_t numClusters, size_t numClasses, EnergyFun const& energy)
{
    std::vector<Cluster> clusters(numClusters, Cluster(&energy));
    for (size_t i = 0; i < sp.pixels(); ++i)
    {
        assert(sp.atSite(i) < numClusters);
        clusters[sp.atSite(i)].accumFeature += Feature(color, i);
        clusters[sp.atSite(i)].size++;
        if (labeling.atSite(i) < numClasses)
            clusters[sp.atSite(i)].labelFrequencies[labeling.atSite(i)]++;
    }
    for (auto& c : clusters)
    {
        c.updateMean();
        c.updateLabel();
    }
    return clusters;
}

#endif //HSEG_CLUSTERER_H
