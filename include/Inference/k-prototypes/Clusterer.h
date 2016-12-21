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
#include "typedefs.h"

/**
 * Clusters an image based on color, position, and a given labeling
 */
template<typename EnergyFun, typename ClusterId = Label>
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
    uint32_t run(ClusterId numClusters, Label numLabels, ColorImage<T> const& color, LabelImage const& labels);

    /**
     * @return Clustering result
     */
    inline LabelImage const& clustership() const
    {
        return m_clustership;
    }

    /**
     * @return Current clusters
     */
    inline std::vector<Cluster> const& clusters() const
    {
        return m_clusters;
    }

    template<typename T>
    static std::vector<Cluster>
    computeClusters(LabelImage const& sp, ColorImage<T> const& color, LabelImage const& labeling, ClusterId numClusters,
                    Label numClasses, EnergyFun const& energy);

private:
    EnergyFun m_energy;
    std::vector<Cluster> m_clusters;
    LabelImage m_clustership;
    bool m_initialized = false;
    float m_conv = 0.001f; // Percentage of pixels that may change in one iteration for the algorithm to terminate

    template<typename T>
    void initPrototypes(ColorImage<T> const& color, LabelImage const& labels);

    template<typename T>
    void allocatePrototypes(ColorImage<T> const& color, LabelImage const& labels);

    template<typename T>
    uint32_t reallocatePrototypes(ColorImage<T> const& color, LabelImage const& labels);

    ClusterId findClosestCluster(Feature const& feature, Label classLabel) const;
};

template<typename EnergyFun, typename ClusterId>
Clusterer<EnergyFun, ClusterId>::Clusterer(EnergyFun energy)
        : m_energy(energy)
{
}

template<typename EnergyFun, typename ClusterId>
ClusterId Clusterer<EnergyFun, ClusterId>::findClosestCluster(Feature const& feature, Label classLabel) const
{
    ClusterId minId = 0;
    Cost minDist = m_energy.pixelToClusterDistance(feature, classLabel, m_clusters, 0);
    for(ClusterId id = 1; id < m_clusters.size(); ++id)
    {
        Cost dist = m_energy.pixelToClusterDistance(feature, classLabel, m_clusters, id);
        if(dist < minDist)
        {
            minDist = dist;
            minId = id;
        }
    }
    return minId;
}

template<typename EnergyFun, typename ClusterId>
template<typename T>
uint32_t Clusterer<EnergyFun, ClusterId>::run(ClusterId numClusters, Label /* numLabels */, ColorImage<T> const& color, LabelImage const& labels)
{
    assert(color.pixels() == labels.pixels());

    if(!m_initialized)
    {
        m_clusters.resize(numClusters, Cluster(&m_energy));
        m_clustership = LabelImage(color.width(), color.height());

        initPrototypes(color, labels);
        allocatePrototypes(color, labels);
        m_initialized = true;
    }

    //float energy = computeEnergy(color, labels);
    //std::cout << "Energy after cluster allocation: " << energy << std::endl;

    uint32_t moves = color.pixels(), lastMoves;
    uint32_t iter = 0;
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

template<typename EnergyFun, typename ClusterId>
template<typename T>
void Clusterer<EnergyFun, ClusterId>::initPrototypes(ColorImage<T> const& color, LabelImage const& labels)
{
    // Randomly select k objects as initial prototypes
    std::default_random_engine generator(0/*std::chrono::system_clock::now().time_since_epoch().count()*/);
    std::uniform_int_distribution<size_t> distribution(0, color.pixels() - 1);
    for (auto& c : m_clusters)
    {
        if (c.size > 0)
            continue;

        SiteId site;
        while(true)
        {
            // Only pick this site if the label is valid. It might be invalid for clustering on ground truth images,
            // where some pixels are marked as "any label".
            site = distribution(generator);
            auto coords = helper::coord::siteTo2DCoordinate(site, color.width());
            c.label = labels.at(coords.x(), coords.y(), 0);
            if(c.label < m_energy.unaryFile().classes())
                break;
        }
        c.mean = Feature(color, site);
    }
}

template<typename EnergyFun, typename ClusterId>
template<typename T>
void Clusterer<EnergyFun, ClusterId>::allocatePrototypes(ColorImage<T> const& color, LabelImage const& labels)
{
    for (SiteId i = 0; i < color.pixels(); ++i)
    {
        Feature curFeature(color, i);
        Label curLabel = labels.atSite(i);

        // Compute closest cluster
        ClusterId minCluster = findClosestCluster(curFeature, curLabel);

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

template<typename EnergyFun, typename ClusterId>
template<typename T>
uint32_t Clusterer<EnergyFun, ClusterId>::reallocatePrototypes(ColorImage<T> const& color, LabelImage const& labels)
{
    uint32_t moves = 0;
    for (SiteId i = 0; i < color.pixels(); ++i)
    {
        Feature curFeature(color, i);
        Label curLabel = labels.atSite(i);

        // Compute closest cluster
        ClusterId minCluster = findClosestCluster(curFeature, curLabel);
        ClusterId oldCluster = m_clustership.atSite(i);

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

template<typename EnergyFun, typename ClusterId>
template<typename T>
std::vector<Cluster>
Clusterer<EnergyFun, ClusterId>::computeClusters(LabelImage const& sp, ColorImage<T> const& color, LabelImage const& labeling,
                                                 ClusterId numClusters, Label numClasses, EnergyFun const& energy)
{
    std::vector<Cluster> clusters(numClusters, Cluster(&energy));
    for (SiteId i = 0; i < sp.pixels(); ++i)
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
