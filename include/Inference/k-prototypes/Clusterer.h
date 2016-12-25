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

    /**
     * Initializes both clusters and clustership.
     * @tparam T Type of the color image
     * @param color Color image
     * @param labels Labeling
     * @param numClusters Amount of clusters
     * @warning This will never terminate if there are less pixels with valid labels than there are cluster centers!
     */
    template<typename T>
    void initPrototypes(ColorImage<T> const& color, LabelImage const& labels, ClusterId numClusters);

    /**
     * Updates the clusters according to the clustership
     * @tparam T Color image type
     * @param color Color image
     * @param labels Labeling
     */
    template<typename T>
    void updatePrototypes(ColorImage<T> const& color, LabelImage const& labels);

    /**
     * Updates the clustership according to the clusters
     * @tparam T Color image type
     * @param color Color image
     * @param labels Labeling
     * @return Amount of pixels that have switched cluster association
     */
    template<typename T>
    uint32_t updateClustership(ColorImage<T> const& color, LabelImage const& labels);

    /**
     * Computes the clostest cluster according to the energy function
     * @param feature Feature
     * @param classLabel Class label
     * @return Id of the closest cluster
     */
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
        m_clusters.reserve(numClusters);
        m_clustership = LabelImage(color.width(), color.height());

        initPrototypes(color, labels, numClusters);
        m_initialized = true;
    }

    uint32_t moves;
    uint32_t iter = 0;
    do
    {

        iter++;
        updatePrototypes(color, labels);
        moves = updateClustership(color, labels);

        //std::cout << iter << ": moves: " << moves << ", threshold: " << color.pixels() * m_conv << std::endl;
    } while (moves > color.pixels() * m_conv);

    return iter;
}

template<typename EnergyFun, typename ClusterId>
template<typename T>
void Clusterer<EnergyFun, ClusterId>::initPrototypes(ColorImage<T> const& color, LabelImage const& labels, ClusterId numClusters)
{
    struct allocation
    {
        size_t clusterId;
        float distance;
        allocation(size_t id, float dist) : clusterId(id), distance(dist) {}
    };

    // Randomly select a pixel as initial prototype
    std::default_random_engine generator(0/*std::chrono::system_clock::now().time_since_epoch().count()*/);
    std::uniform_int_distribution<size_t> distribution(0, color.pixels() - 1);
    std::vector<allocation> clAlloc; // Distance to closest cluster center
    clAlloc.reserve(color.pixels());
    // Only pick this site if the label is valid. It might be invalid for clustering on ground truth images,
    // where some pixels are marked as "any label".
    // Note that this will never terminate if there are no pixels with valid labels!
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
    for(size_t i = 0; i < color.pixels(); ++i)
    {
        float const dist = m_energy.pixelToClusterDistance(Feature(color, i), labels.atSite(i), m_clusters, 0);
        clAlloc.emplace_back(0, dist);
    }

    // Pick a new object as the next cluster proportional to the squared distance
    std::vector<float> weights(color.pixels(), 0.f);
    for(size_t c = 1; c < numClusters; ++c)
    {
        size_t n = 0;
        std::generate(weights.begin(), weights.end(), [&] { float const d = clAlloc[n++].distance; return d * d; });
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
        for(size_t i = 0; i < color.pixels(); ++i)
        {
            float const dist = m_energy.pixelToClusterDistance(Feature(color, i), labels.atSite(i), m_clusters, c);
            if(dist < clAlloc[i].distance)
            {
                clAlloc[i].distance = dist;
                clAlloc[i].clusterId = c;
            }
        }
    }

    // Allocate prototypes
    for (size_t i = 0; i < color.pixels(); ++i)
    {
        Feature curFeature(color, i);
        Label curLabel = labels.atSite(i);

        // Compute closest cluster
        size_t minCluster = clAlloc[i].clusterId;

        // Assign to cluster
        m_clustership.atSite(i) = minCluster;
        m_clusters[minCluster].size++;
        m_clusters[minCluster].accumFeature += curFeature;
        if(curLabel < m_energy.unaryFile().classes())
        {
            // Only update for valid labels
            m_clusters[minCluster].labelFrequencies[curLabel]++;
        }
    }
    for(auto& cl : m_clusters)
    {
        cl.updateMean();
        cl.updateLabel();
    }
}

template<typename EnergyFun, typename ClusterId>
template<typename T>
void Clusterer<EnergyFun, ClusterId>::updatePrototypes(ColorImage<T> const& color, LabelImage const& labels)
{
    // Clear all clusters
    for(auto& c : m_clusters)
    {
        c.accumFeature = Feature();
        std::fill(c.labelFrequencies.begin(), c.labelFrequencies.end(), 0);
        c.size = 0;
    }
    // Update them
    for (SiteId i = 0; i < color.pixels(); ++i)
    {
        ClusterId const cId = m_clustership.atSite(i);
        Feature curFeature(color, i);
        Label curLabel = labels.atSite(i);
        m_clusters[cId].size++;
        m_clusters[cId].accumFeature += curFeature;
        if(curLabel < m_energy.unaryFile().classes())
            m_clusters[cId].labelFrequencies[curLabel]++;
    }
    for(auto& c : m_clusters)
    {
        c.updateMean();
        c.updateLabel();
    }
}

template<typename EnergyFun, typename ClusterId>
template<typename T>
uint32_t Clusterer<EnergyFun, ClusterId>::updateClustership(ColorImage<T> const& color, LabelImage const& labels)
{
    uint32_t moves = 0;

    for(SiteId i = 0; i < color.pixels(); ++i)
    {
        Feature curFeature(color, i);
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
