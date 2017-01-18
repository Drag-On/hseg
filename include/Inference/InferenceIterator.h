//
// Created by jan on 28.08.16.
//

#ifndef HSEG_INFERENCEITERATOR_H
#define HSEG_INFERENCEITERATOR_H

#include <Energy/EnergyFunction.h>
#include <Image/FeatureImage.h>
#include <MRFEnergy.h>
#include <typeGeneral.h>
#include <Image/Image.h>
#include <helper/coordinate_helper.h>
#include "InferenceResult.h"
#include "InferenceResultDetails.h"
#include "Cluster.h"

/**
 * Infers both class labels and superpixels on an image
 */
template<typename EnergyFun>
class InferenceIterator
{
public:
    /**
     * Constructor
     * @param e Energy function
     * @param pImg Color image
     */
    InferenceIterator(EnergyFun const* e, FeatureImage const* pImg, float eps = 0.f);

    /**
     * Does the actual inference
     * @param numIter Amount of iterations to do. If 0, run until convergence.
     * @return Resulting labeling and segmentation
     */
    InferenceResult run(uint32_t numIter = 0);

    /**
     * Does inference and saves detailed results
     * @param numIter Amount of iterations to do. If 0, run until convergence.
     * @return Detailed results
     */
    InferenceResultDetails runDetailed(uint32_t numIter = 0);

    /**
     * Does inference on a fixed labeling, i.e. predicts the latent variables only
     * @param gt Ground truth labeling
     * @param numIter Amount of iterations to do. If 0, run until convergence.
     * @return Ground truth labeling and latent variables that best explain it
     */
    InferenceResult runOnGroundTruth(LabelImage const& gt, uint32_t numIter = 0);

private:
    EnergyFun const* m_pEnergy;
    FeatureImage const* m_pImg;
    float m_eps = 0.f;

    void updateClusterAffiliation(LabelImage& outClustering, LabelImage const& labeling, std::vector<Cluster> const& clusters);

    void updateLabels(LabelImage& outLabeling, std::vector<Cluster>& outClusters, LabelImage const& clustering);

    void updateClusterFeatures(std::vector<Cluster>& outClusters, LabelImage const& labeling, LabelImage const& clustering);

    void initialize(LabelImage& outLabeling, LabelImage& outClustering, std::vector<Cluster>& outClusters);

    void updateLabelsOnGroundTruth(LabelImage const& gt, std::vector<Cluster>& outClusters, LabelImage const& clustering);

};

template<typename EnergyFun>
InferenceIterator<EnergyFun>::InferenceIterator(EnergyFun const* e, FeatureImage const* pImg, float eps)
        : m_pEnergy(e),
          m_pImg(pImg),
          m_eps(eps)
{
}

template<typename EnergyFun>
void InferenceIterator<EnergyFun>::updateClusterAffiliation(LabelImage& outClustering, LabelImage const& labeling, std::vector<Cluster> const& clusters)
{
    // Exhaustive search
    for(SiteId i = 0; i < m_pImg->width() * m_pImg->height(); ++i)
    {
        Feature const& f1 = m_pImg->atSite(i);
        Label const l1 = labeling.atSite(i);
        Feature const& f2 = clusters[0].m_feature;
        Label const l2 = clusters[0].m_label;

        Cost minCost = m_pEnergy->higherOrderCost(f1, f2, l1, l2) + m_pEnergy->featureCost(f1, f2);
        ClusterId minCluster = 0;
        for(ClusterId k = 1; k < clusters.size(); ++k)
        {
            Feature const& f2 = clusters[k].m_feature;
            Label const l2 = clusters[k].m_label;
            Cost c = m_pEnergy->higherOrderCost(f1, f2, l1, l2) + m_pEnergy->featureCost(f1, f2);
            if(c < minCost)
            {
                minCost = c;
                minCluster = k;
            }
        }

        outClustering.atSite(i) = minCluster;
    }
}

template<typename EnergyFun>
void InferenceIterator<EnergyFun>::updateLabels(LabelImage& outLabeling, std::vector<Cluster>& outClusters, LabelImage const& clustering)
{
    ClusterId const numClusters = m_pEnergy->numClusters();
    Label const numClasses = m_pEnergy->numClasses();
    SiteId const numPx = outLabeling.pixels();
    SiteId const numNodes = numPx + numClusters;

    // Set up the graph
    TypeGeneral::GlobalSize globalSize;
    MRFEnergy<TypeGeneral> mrfEnergy(globalSize);

    std::vector<MRFEnergy<TypeGeneral>::NodeId> nodeIds;
    nodeIds.reserve(numNodes);

    // Unary term for each pixel and each cluster
    for (SiteId i = 0; i < numNodes; ++i)
    {
        Feature const& f = m_pImg->atSite(i);
        std::vector<TypeGeneral::REAL> confidences(numClasses, 0.f);
        if(i < numPx)
        {
            // It's a pixel
            for (Label l = 0; l < numClasses; ++l)
                confidences[l] = m_pEnergy->unaryCost(i, f, l);
        }
        else
        {
            // It's an auxiliary node
            // Cluster nodes don't currently have unary terms
        }
        auto id = mrfEnergy.AddNode(TypeGeneral::LocalSize(numClasses), TypeGeneral::NodeData(confidences.data()));
        nodeIds.push_back(id);
    }

    // Pairwise term for each combination of adjacent pixels and each pixel to the cluster it is allocated to
    for (SiteId i = 0; i < numPx; ++i)
    {
        auto coords = helper::coord::siteTo2DCoordinate(i, outLabeling.width());
        Feature const& f = m_pImg->atSite(i);

        // Set up pixel neighbor connections
        decltype(coords) coordsR = {coords.x() + 1, coords.y()};
        decltype(coords) coordsD = {coords.x(), coords.y() + 1};
        if (coordsR.x() < outLabeling.width())
        {
            SiteId siteR = helper::coord::coordinateToSite(coordsR.x(), coordsR.y(), outLabeling.width());
            Feature const& fR = m_pImg->atSite(siteR);
            std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
            for(Label l1 = 0; l1 < numClasses; ++l1)
            {
                for(Label l2 = 0; l2 < l1; ++l2)
                {
                    Cost cost = m_pEnergy->pairwiseCost(f, fR, l1, l2);
                    costMat[l1 + l2 * numClasses] = costMat[l2 + l1 * numClasses] = cost;
                }
            }
            TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
            mrfEnergy.AddEdge(nodeIds[i], nodeIds[siteR], edgeData);
        }
        if (coordsD.y() < outLabeling.height())
        {
            SiteId siteD = helper::coord::coordinateToSite(coordsD.x(), coordsD.y(), outLabeling.width());
            Feature const& fD = m_pImg->atSite(siteD);
            std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
            for(Label l1 = 0; l1 < numClasses; ++l1)
            {
                for(Label l2 = 0; l2 < l1; ++l2)
                {
                    Cost cost = m_pEnergy->pairwiseCost(f, fD, l1, l2);
                    costMat[l1 + l2 * numClasses] = costMat[l2 + l1 * numClasses] = cost;
                }
            }
            TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
            mrfEnergy.AddEdge(nodeIds[i], nodeIds[siteD], edgeData);
        }

        // Set up connection to auxiliary nodes
        ClusterId k = clustering.atSite(i);
        SiteId auxSite = k + numPx;
        Feature const& clusFeat = outClusters[k].m_feature;
        std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
        for(Label l1 = 0; l1 < numClasses; ++l1)
        {
            for(Label l2 = 0; l2 < l1; ++l2)
            {
                Cost cost = m_pEnergy->higherOrderCost(f, clusFeat, l1, l2);
                costMat[l1 + l2 * numClasses] = costMat[l2 + l1 * numClasses] = cost;
            }
        }
        TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
        mrfEnergy.AddEdge(nodeIds[i], nodeIds[auxSite], edgeData);
    }

    // Do the actual minimization
    MRFEnergy<TypeGeneral>::Options options;
    options.m_eps = 0.0f;
    MRFEnergy<TypeGeneral>::REAL lowerBound = 0, energy = 0;
    mrfEnergy.Minimize_TRW_S(options, lowerBound, energy);

    // Copy over result
    for (SiteId i = 0; i < numPx; ++i)
        outLabeling.atSite(i) = mrfEnergy.GetSolution(nodeIds[i]);
    for (ClusterId k = 0; k < numClusters; ++k)
        outClusters[k].m_label = mrfEnergy.GetSolution(nodeIds[numPx + k]);
}

template<typename EnergyFun>
void InferenceIterator<EnergyFun>::updateClusterFeatures(std::vector<Cluster>& outClusters, LabelImage const& labeling, LabelImage const& clustering)
{
    uint32_t const numClusters = m_pEnergy->numClusters();
    std::vector<uint32_t> clusterSize(numClusters, 0);

    // Reset all current cluster features to zero vectors
    for(auto& c : outClusters)
        c.m_feature = Feature::Zero(c.m_feature.size());

    // Do one sweep over the image and update the cluster features on the fly
    for(SiteId i = 0; i < labeling.pixels(); ++i)
    {
        ClusterId k = clustering.atSite(i);
        Label l1 = labeling.atSite(i);
        Label l2 = outClusters[k].m_label;

        // Count the number of pixels allocated to each cluster
        clusterSize[k]++;

        // Update feature
        Feature& f = outClusters[k].m_feature;
        Feature const& fPx = m_pImg->atSite(i);
        auto const& sigma = m_pEnergy->weights().featureSimMatInv();
        auto const& w = m_pEnergy->weights().higherOrder(l1, l2);
        auto const& wTail = w.segment(f.size(), f.size());

        f += fPx - 0.5f * sigma * wTail;
    }
    // Normalize all cluster features
    for (ClusterId k = 0; k < outClusters.size(); ++k)
        outClusters[k].m_feature *= 1.f / clusterSize[k];
}

template<typename EnergyFun>
void InferenceIterator<EnergyFun>::initialize(LabelImage& outLabeling, LabelImage& outClustering, std::vector<Cluster>& outClusters)
{
    // Initialize labeling with background (all zeros)
    outLabeling = LabelImage(m_pImg->width(), m_pImg->height());

    // Initialize clustering with all zeros as well
    outClustering = LabelImage(m_pImg->width(), m_pImg->height());

    // Pick random pixels and initialize clusters with their features
    struct allocation
    {
        ClusterId clusterId;
        Cost distance;
        allocation(size_t id, Cost dist) : clusterId(id), distance(dist) {}
    };

    ClusterId const numClusters = m_pEnergy->numClusters();
    outClusters.reserve(numClusters);
    outClusters.clear();

    // Randomly select a pixel as initial prototype
    std::default_random_engine generator(0/*std::chrono::system_clock::now().time_since_epoch().count()*/);
    std::uniform_int_distribution<SiteId> distribution(0, outLabeling.pixels() - 1);
    std::vector<allocation> clAlloc; // Distance to closest cluster center
    clAlloc.reserve(outLabeling.pixels());
    SiteId const site = distribution(generator);
    outClusters.emplace_back();
    outClusters.back().m_label = 0;
    outClusters.back().m_feature = m_pImg->atSite(site);

    // Compute the distance between each pixel and the newly created cluster center
    for(SiteId i = 0; i < outLabeling.pixels(); ++i)
    {
        Cost const dist = m_pEnergy->featureCost(m_pImg->atSite(i), outClusters.back().m_feature);
        clAlloc.emplace_back(0, dist);
    }

    // Pick another pixel as the next cluster, where every pixel is weighted proportional to the squared distance to
    // the currently closest cluster
    std::vector<Cost> weights(outLabeling.pixels(), 0.f);
    for(ClusterId k = 1; k < numClusters; ++k)
    {
        SiteId n = 0;
        std::generate(weights.begin(), weights.end(), [&] { Cost const d = clAlloc[n++].distance; return d * d; });
        std::discrete_distribution<SiteId> distribution(weights.begin(), weights.end());
        SiteId const site = distribution(generator);
        outClusters.emplace_back();
        outClusters.back().m_label = 0;
        outClusters.back().m_feature = m_pImg->atSite(site);
        // Recompute cluster distances
        for(SiteId i = 0; i < outLabeling.pixels(); ++i)
        {
            Cost const dist = m_pEnergy->featureCost(m_pImg->atSite(i), outClusters.back().m_feature);
            if(dist < clAlloc[i].distance)
            {
                clAlloc[i].distance = dist;
                clAlloc[i].clusterId = k;
            }
        }
    }
}

template<typename EnergyFun>
void InferenceIterator<EnergyFun>::updateLabelsOnGroundTruth(LabelImage const& gt, std::vector<Cluster>& outClusters,
                                                             LabelImage const& clustering)
{
    ClusterId const numClusters = m_pEnergy->numClusters();
    Label const numClasses = m_pEnergy->numClasses();
    SiteId const numPx = gt.pixels();

    std::vector<std::vector<Cost>> clusterCost(numClusters, std::vector<Cost>(numClasses, 0));

    // Every auxiliary node has just unary terms, however they are a sum of all allocated pixels
    for(SiteId i = 0; i < numPx; ++i)
    {
        Label const l = gt.atSite(i);
        ClusterId const k = clustering.atSite(i);
        Feature const& f = m_pImg->atSite(i);
        Feature const& fClus = outClusters[k].m_feature;

        for (Label lClus = 0; lClus < numClasses; ++lClus)
            clusterCost[k][lClus] += m_pEnergy->higherOrderCost(f, fClus, l, lClus);
    }

    // Find the best label for every cluster
    for(ClusterId k = 0; k < numClusters; ++k)
    {
        auto minEle = std::min_element(clusterCost[k].begin(), clusterCost[k].end());
        outClusters[k].m_label = std::distance(clusterCost[k].begin(), minEle);
    }
}

template<typename EnergyFun>
InferenceResult InferenceIterator<EnergyFun>::run(uint32_t numIter)
{
    InferenceResult result;

    // Initialize variables
    initialize(result.labeling, result.clustering, result.clusters);

    // Iterate until either convergence or the maximum number of iterations has been hit
    Cost energy =  m_pEnergy->giveEnergy(*m_pImg, result.labeling, result.clustering, result.clusters);
    Cost lastEnergy = std::numeric_limits<Cost>::max();
    uint32_t iter = 0;
    for (; (numIter > 0) ? (iter < numIter) : (lastEnergy - energy >= m_eps || iter == 0); ++iter)
    {
        lastEnergy = energy;

        // Update clustering
        updateClusterAffiliation(result.clustering, result.labeling, result.clusters);

        // Update custer features
        updateClusterFeatures(result.clusters, result.labeling, result.clustering);

        // Update labels
        updateLabels(result.labeling, result.clusters, result.clustering);

        // Compute current energy to check for convergence
        energy = m_pEnergy->giveEnergy(*m_pImg, result.labeling, result.clustering, result.clusters);
    }

    result.numIter = iter;
    return result;
}

template<typename EnergyFun>
InferenceResultDetails InferenceIterator<EnergyFun>::runDetailed(uint32_t numIter)
{
    InferenceResultDetails result;

    // Initialize variables
    LabelImage labeling, clustering;
    std::vector<Cluster> clusters;
    initialize(labeling, clustering, clusters);

    // Iterate until either convergence or the maximum number of iterations has been hit
    Cost energy = m_pEnergy->giveEnergy(*m_pImg, labeling, clustering, clusters);
    result.energy.push_back(energy);
    Cost lastEnergy = std::numeric_limits<Cost>::max();
    uint32_t iter = 0;
    for (; (numIter > 0) ? (iter < numIter) : (lastEnergy - energy >= m_eps || iter == 0); ++iter)
    {
        lastEnergy = energy;

        // Update clustering
        updateClusterAffiliation(clustering, labeling, clusters);

        // Update custer features
        updateClusterFeatures(clusters, labeling, clustering);

        // Update labels
        updateLabels(labeling, clusters, clustering);

        // Compute current energy to check for convergence
        energy = m_pEnergy->giveEnergy(*m_pImg, labeling, clustering, clusters);

        // Store intermediate results
        result.clusters.push_back(clusters);
        result.clusterings.push_back(clustering);
        result.labelings.push_back(labeling);
        result.energy.push_back(energy);
    }

    result.numIter = iter;
    return result;
}

template<typename EnergyFun>
InferenceResult InferenceIterator<EnergyFun>::runOnGroundTruth(LabelImage const& gt, uint32_t numIter)
{
    InferenceResult result;

    // Initialize variables
    initialize(result.labeling, result.clustering, result.clusters);
    assert(gt.width() == result.labeling.width() && gt.height() == result.labeling.height());
    result.labeling = gt;

    // Iterate until either convergence or the maximum number of iterations has been hit
    Cost energy = m_pEnergy->giveEnergy(*m_pImg, result.labeling, result.clustering, result.clusters);
    Cost lastEnergy = std::numeric_limits<Cost>::max();
    uint32_t iter = 0;
    for (; (numIter > 0) ? (iter < numIter) : (lastEnergy - energy >= m_eps || iter == 0); ++iter)
    {
        lastEnergy = energy;

        // Update clustering
        updateClusterAffiliation(result.clustering, result.labeling, result.clusters);

        // Update custer features
        updateClusterFeatures(result.clusters, result.labeling, result.clustering);

        // Update labels
        updateLabelsOnGroundTruth(result.labeling, result.clusters, result.clustering);

        // Compute current energy to check for convergence
        energy = m_pEnergy->giveEnergy(*m_pImg, result.labeling, result.clustering, result.clusters);
    }

    result.numIter = iter;
    return result;
}


#endif //HSEG_INFERENCEITERATOR_H
