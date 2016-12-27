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
     * @param color Color image
     * @param labels Initial labeling
     * @param numClusters Amount of clusters
     */
    template<typename T>
    Clusterer(EnergyFun energy, ColorImage<T> const& color, LabelImage const& labels, ClusterId numClusters);

    /**
     * Runs the clustering process
     * @param labels Label image
     * @returns The amount of iterations that have been done
     * @details The resulting clustering will never have more clusters than specified, but it might well not use all
     *          of them. If this is called multiple times on the same object, the algorithm will be warm-started with
     *          the previous result.
     */
    uint32_t run(LabelImage const& labels);

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
    computeClusters(LabelImage const& sp, ColorImage<T> const& color,
                        LabelImage const& labeling, ClusterId numClusters,
                        EnergyFun const& energy);

private:
    EnergyFun m_energy;
    std::vector<Cluster> m_clusters;
    LabelImage m_clustership;
    std::vector<Feature> m_features;
    float const m_conv = 0.001f; // Percentage of pixels that may change in one iteration for the algorithm to terminate

    /**
     * Initializes both clusters and clustership.
     * @param labels Labeling
     * @param numClusters Amount of clusters
     * @warning This will never terminate if there are less pixels with valid labels than there are cluster centers!
     */
    void initPrototypes(LabelImage const& labels, ClusterId numClusters);

    /**
     * Updates the clusters according to the clustership
     * @tparam T Color image type
     * @param labels Labeling
     */
    void updatePrototypes(LabelImage const& labels);

    /**
     * Updates the clustership according to the clusters
     * @tparam T Color image type
     * @param color Color image
     * @param labels Labeling
     * @return Amount of pixels that have switched cluster association
     */
    uint32_t updateClustership(LabelImage const& labels);

    /**
     * Computes the clostest cluster according to the energy function
     * @param feature Feature
     * @param classLabel Class label
     * @return Id of the closest cluster
     */
    ClusterId findClosestCluster(Feature const& feature, Label classLabel) const;
};

template<typename EnergyFun, typename ClusterId>
template <typename T>
Clusterer<EnergyFun, ClusterId>::Clusterer(EnergyFun energy, ColorImage<T> const& color, LabelImage const& labels, ClusterId numClusters)
        : m_energy(energy)
{
    assert(color.pixels() == labels.pixels());

    // Cache features
    m_features.reserve(color.pixels());
    for(SiteId i = 0; i < color.pixels(); ++i)
        m_features.emplace_back(color, i);

    // Initialize clusters and clustership
    m_clusters.reserve(numClusters);
    m_clustership = LabelImage(color.width(), color.height());
    initPrototypes(labels, numClusters);
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
uint32_t Clusterer<EnergyFun, ClusterId>::run(LabelImage const& labels)
{
    assert(m_features.size() == labels.pixels());

    uint32_t moves;
    uint32_t iter = 0;
    do
    {
        iter++;
        updatePrototypes(labels);
        moves = updateClustership(labels);

        //std::cout << iter << ": moves: " << moves << ", threshold: " << color.pixels() * m_conv << std::endl;
    } while (moves > labels.pixels() * m_conv);

    return iter;
}

template<typename EnergyFun, typename ClusterId>
void Clusterer<EnergyFun, ClusterId>::initPrototypes(LabelImage const& labels, ClusterId numClusters)
{
    struct allocation
    {
        ClusterId clusterId;
        Cost distance;
        allocation(size_t id, Cost dist) : clusterId(id), distance(dist) {}
    };

    // Randomly select a pixel as initial prototype
    std::default_random_engine generator(0/*std::chrono::system_clock::now().time_since_epoch().count()*/);
    std::uniform_int_distribution<SiteId> distribution(0, labels.pixels() - 1);
    std::vector<allocation> clAlloc; // Distance to closest cluster center
    clAlloc.reserve(labels.pixels());
    // Only pick this site if the label is valid. It might be invalid for clustering on ground truth images,
    // where some pixels are marked as "any label".
    // Note that this will never terminate if there are no pixels with valid labels!
    while(true)
    {
        SiteId const site = distribution(generator);
        Label const l = labels.atSite(site);
        if(l < m_energy.unaryFile().classes())
        {
            m_clusters.emplace_back(&m_energy);
            m_clusters.back().label = l;
            m_clusters.back().mean = m_features[site];
            break;
        }
    }
    // Compute the distance between each object and the newly created cluster center
    for(SiteId i = 0; i < labels.pixels(); ++i)
    {
        Cost const dist = m_energy.pixelToClusterDistance(m_features[i], labels.atSite(i), m_clusters, 0);
        clAlloc.emplace_back(0, dist);
    }

    // Pick a new object as the next cluster proportional to the squared distance
    std::vector<Cost> weights(labels.pixels(), 0.f);
    for(ClusterId c = 1; c < numClusters; ++c)
    {
        SiteId n = 0;
        std::generate(weights.begin(), weights.end(), [&] { Cost const d = clAlloc[n++].distance; return d * d; });
        std::discrete_distribution<SiteId> distribution(weights.begin(), weights.end());
        while(true)
        {
            SiteId const site = distribution(generator);
            Label const l = labels.atSite(site);
            if(l < m_energy.unaryFile().classes())
            {
                m_clusters.emplace_back(&m_energy);
                m_clusters.back().label = l;
                m_clusters.back().mean = m_features[site];
                break;
            }
        }
        // Recompute cluster distances
        for(SiteId i = 0; i < labels.pixels(); ++i)
        {
            Cost const dist = m_energy.pixelToClusterDistance(m_features[i], labels.atSite(i), m_clusters, c);
            if(dist < clAlloc[i].distance)
            {
                clAlloc[i].distance = dist;
                clAlloc[i].clusterId = c;
            }
        }
    }

    // Allocate prototypes
    for (SiteId i = 0; i < labels.pixels(); ++i)
    {
        // Compute closest cluster
        ClusterId minCluster = clAlloc[i].clusterId;

        // Assign to cluster
        m_clustership.atSite(i) = minCluster;
        m_clusters[minCluster].allocated.push_back(i);
    }
    for(auto& cl : m_clusters)
        cl.update(m_features, labels);
}

template<typename EnergyFun, typename ClusterId>
void Clusterer<EnergyFun, ClusterId>::updatePrototypes(LabelImage const& labels)
{
    // Clear all clusters
    for(auto& c : m_clusters)
        c.allocated.clear();
    // Update them
    for (SiteId i = 0; i < labels.pixels(); ++i)
    {
        ClusterId const cId = m_clustership.atSite(i);
        m_clusters[cId].allocated.push_back(i);
    }
    for(auto& cl : m_clusters)
        cl.update(m_features, labels);
}

template<typename EnergyFun, typename ClusterId>
uint32_t Clusterer<EnergyFun, ClusterId>::updateClustership(LabelImage const& labels)
{
    uint32_t moves = 0;

    for(SiteId i = 0; i < labels.pixels(); ++i)
    {
        Feature const& curFeature = m_features[i];
        Label curLabel = labels.atSite(i);

        // Compute closest cluster
        ClusterId minCluster = findClosestCluster(curFeature, curLabel);
        ClusterId oldCluster = m_clustership.atSite(i);

        if (oldCluster != minCluster)
        {
            m_clustership.atSite(i) = minCluster;
            moves++;
        }
    }

    return moves;
}

template<typename EnergyFun, typename ClusterId>
template<typename T>
std::vector<Cluster>
Clusterer<EnergyFun, ClusterId>::computeClusters(LabelImage const& sp, ColorImage<T> const& color,
                                                 LabelImage const& labeling, ClusterId numClusters,
                                                 EnergyFun const& energy)
{
    std::vector<Feature> features;
    features.reserve(color.pixels());
    for(SiteId s = 0; s < color.pixels(); ++s)
        features.emplace_back(color, s);
    std::vector<Cluster> clusters(numClusters, Cluster(&energy));
    for (SiteId i = 0; i < sp.pixels(); ++i)
    {
        assert(sp.atSite(i) < numClusters);
        clusters[sp.atSite(i)].allocated.push_back(i);
    }
    for(auto& cl : clusters)
        cl.update(features, labeling);
    return clusters;
}

#endif //HSEG_CLUSTERER_H
