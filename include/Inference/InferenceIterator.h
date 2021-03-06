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
#include <Timer.h>
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
     * @param pPxFeat Pixel features
     * @param pClusterFeat Cluster features
     */
    InferenceIterator(EnergyFun const* e, FeatureImage const* pPxFeat, FeatureImage const* pClusterFeat, float eps = 1e-5f, uint32_t maxIter = 50);

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

protected:
    EnergyFun const* m_pEnergy;
    FeatureImage const* m_pPxFeat;
    FeatureImage const* m_pClusterFeat;
    float m_eps;
    uint32_t m_maxIter;

    void updateClusterAffiliation(LabelImage& outClustering, LabelImage const& labeling, std::vector<Cluster> const& clusters);

    void updateLabels(LabelImage& outLabeling, std::vector<Cluster>& outClusters, LabelImage const& clustering, FeatureImage* pOutMarginals = nullptr);

    void updateClusterFeatures(std::vector<Cluster>& outClusters, LabelImage const& labeling, LabelImage const& clustering);

    void initialize(LabelImage& outLabeling, LabelImage& outClustering, std::vector<Cluster>& outClusters);

    void updateLabelsOnGroundTruth(LabelImage const& gt, std::vector<Cluster>& outClusters, LabelImage const& clustering);

};

template<typename EnergyFun>
InferenceIterator<EnergyFun>::InferenceIterator(EnergyFun const* e, FeatureImage const* pPxFeat, FeatureImage const* pClusterFeat, float eps, uint32_t maxIter)
        : m_pEnergy(e),
          m_pPxFeat(pPxFeat),
          m_pClusterFeat(pClusterFeat),
          m_eps(eps),
          m_maxIter(maxIter)
{
}

template<typename EnergyFun>
void InferenceIterator<EnergyFun>::updateClusterAffiliation(LabelImage& outClustering, LabelImage const& labeling, std::vector<Cluster> const& clusters)
{
    PROFILE_THIS

    // Exhaustive search
    for(SiteId i = 0; i < m_pClusterFeat->width() * m_pClusterFeat->height(); ++i)
    {
        Feature const& f1 = m_pClusterFeat->atSite(i);
        Label l1 = labeling.atSite(i);
        Feature const& f2 = clusters[0].m_feature;
        Label const l2 = clusters[0].m_label;

        bool invalidSite = l1 >= m_pEnergy->numClasses();

        // If the pixel label is invalid just pretend that it has the same label as the cluster
        if(invalidSite)
            l1 = l2;

        Cost minCost = m_pEnergy->higherOrderCost(f1, f2, l1, l2) + m_pEnergy->featureCost(f1, f2, l1, l2) + m_pEnergy->higherOrderSpecialUnaryCost(i, l2);
        ClusterId minCluster = 0;
        for(ClusterId k = 1; k < clusters.size(); ++k)
        {
            Feature const& f2 = clusters[k].m_feature;
            Label const l2 = clusters[k].m_label;
            if(invalidSite) // See comment above
                l1 = l2;
            Cost c = m_pEnergy->higherOrderCost(f1, f2, l1, l2) + m_pEnergy->featureCost(f1, f2, l1, l2) + m_pEnergy->higherOrderSpecialUnaryCost(i, l2);
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
void InferenceIterator<EnergyFun>::updateLabels(LabelImage& outLabeling, std::vector<Cluster>& outClusters, LabelImage const& clustering, FeatureImage* pOutMarginals)
{
    PROFILE_THIS

    ClusterId const numClusters = m_pEnergy->numClusters();
    Label const numClasses = m_pEnergy->numClasses();
    SiteId const numPx = outLabeling.pixels();
    SiteId const numNodes = numPx + numClusters;

    // Set up the graph
    TypeGeneral::GlobalSize globalSize;
    MRFEnergy<TypeGeneral> mrfEnergy(globalSize);

    std::vector<MRFEnergy<TypeGeneral>::NodeId> nodeIds;
    nodeIds.reserve(numNodes);

    // Figure out unary terms for cluster nodes
    std::vector<std::vector<Cost>> clusterUnary(numClusters, std::vector<Cost>(numClasses, 0));
    if(numClusters > 0)
    {
        for (SiteId i = 0; i < numPx; ++i)
        {
            ClusterId k = clustering.atSite(i);
            for (Label l_k = 0; l_k < numClasses; ++l_k)
                clusterUnary[k][l_k] += m_pEnergy->higherOrderSpecialUnaryCost(i, l_k);
        }
    }

    // Unary term for each pixel
    for (SiteId i = 0; i < numPx; ++i)
    {
        Feature const& f = m_pPxFeat->atSite(i);
        std::vector<TypeGeneral::REAL> confidences(numClasses, 0.f);
        for (Label l = 0; l < numClasses; ++l)
            confidences[l] = m_pEnergy->unaryCost(i, f, l);
        auto id = mrfEnergy.AddNode(TypeGeneral::LocalSize(numClasses), TypeGeneral::NodeData(confidences.data()));
        nodeIds.push_back(id);
    }

    // Unary term for each cluster
    if(numClusters > 0)
    {
        for (SiteId i = numPx; i < numNodes; ++i)
        {
            std::vector<TypeGeneral::REAL> confidences(numClasses, 0.f);
            for (Label l = 0; l < numClasses; ++l)
                confidences[l] = clusterUnary[i - numPx][l];
            auto id = mrfEnergy.AddNode(TypeGeneral::LocalSize(numClasses), TypeGeneral::NodeData(confidences.data()));
            nodeIds.push_back(id);
        }
    }

    // Pairwise term for each combination of adjacent pixels and each pixel to the cluster it is allocated to
    for (SiteId i = 0; i < numPx; ++i)
    {
        auto coords = helper::coord::siteTo2DCoordinate(i, outLabeling.width());
        Feature const& f = m_pPxFeat->atSite(i);

        // Set up pixel neighbor connections
        if(m_pEnergy->usePairwise())
        {
            decltype(coords) coordsR = {coords.x() + 1, coords.y()};
            decltype(coords) coordsD = {coords.x(), coords.y() + 1};
            if (coordsR.x() < outLabeling.width())
            {
                SiteId siteR = helper::coord::coordinateToSite(coordsR.x(), coordsR.y(), outLabeling.width());
                Feature const& fR = m_pPxFeat->atSite(siteR);
                std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
                for(Label l1 = 0; l1 < numClasses; ++l1)
                {
                    for(Label l2 = 0; l2 < numClasses; ++l2)
                    {
                        Cost cost = m_pEnergy->pairwiseCost(f, fR, l1, l2);
                        costMat[l1 + l2 * numClasses] = cost;
                    }
                }
                TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
                mrfEnergy.AddEdge(nodeIds[i], nodeIds[siteR], edgeData);
            }
            if (coordsD.y() < outLabeling.height())
            {
                SiteId siteD = helper::coord::coordinateToSite(coordsD.x(), coordsD.y(), outLabeling.width());
                Feature const& fD = m_pPxFeat->atSite(siteD);
                std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
                for(Label l1 = 0; l1 < numClasses; ++l1)
                {
                    for(Label l2 = 0; l2 < numClasses; ++l2)
                    {
                        Cost cost = m_pEnergy->pairwiseCost(f, fD, l1, l2);
                        costMat[l1 + l2 * numClasses] = cost;
                    }
                }
                TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
                mrfEnergy.AddEdge(nodeIds[i], nodeIds[siteD], edgeData);
            }
        }

        // Set up connection to auxiliary nodes
        Feature const& fClus = m_pClusterFeat->atSite(i);
        if(numClusters > 0)
        {
            ClusterId k = clustering.atSite(i);
            SiteId auxSite = k + numPx;
            Feature const& clusFeat = outClusters[k].m_feature;
            std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
            for (Label l1 = 0; l1 < numClasses; ++l1)
            {
                for (Label l2 = 0; l2 < numClasses; ++l2)
                {
                    Cost cost = m_pEnergy->higherOrderCost(fClus, clusFeat, l1, l2);
                    costMat[l1 + l2 * numClasses] = cost;
                }
            }
            TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
            mrfEnergy.AddEdge(nodeIds[i], nodeIds[auxSite], edgeData);
        }
    }

    // Do the actual minimization
    MRFEnergy<TypeGeneral>::Options options;
    options.m_eps = 0.01f;
    MRFEnergy<TypeGeneral>::REAL lowerBound = 0, energy = 0;
    mrfEnergy.Minimize_TRW_S(options, lowerBound, energy);

//    std::cout << "TRW-S : lower bound = " << lowerBound << ", energy = " << energy << std::endl;

    // Copy over result
    if(pOutMarginals != nullptr)
        *pOutMarginals = FeatureImage(outLabeling.width(), outLabeling.height(), numClasses);
    for (SiteId i = 0; i < numPx; ++i)
    {
        outLabeling.atSite(i) = mrfEnergy.GetSolution(nodeIds[i]);

        // Compute marginals?
        if(pOutMarginals != nullptr)
        {
            // This is just a soft max over the message vector
            double sum = 0;
            for(Label l = 0; l < numClasses; ++l)
            {
                double marginal = std::exp(-nodeIds[i]->m_D.GetValue(globalSize, nodeIds[i]->m_K, l));
                pOutMarginals->atSite(i)[l] = marginal;
                sum += marginal;
            }
            // Normalize
            for(Label l = 0; l < numClasses; ++l)
                pOutMarginals->atSite(i)[l] /= sum;
        }
    }
    for (ClusterId k = 0; k < numClusters; ++k)
        outClusters[k].m_label = mrfEnergy.GetSolution(nodeIds[numPx + k]);
}

template<typename EnergyFun>
void InferenceIterator<EnergyFun>::updateClusterFeatures(std::vector<Cluster>& outClusters, LabelImage const& labeling, LabelImage const& clustering)
{
    PROFILE_THIS

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

        // If the pixel label is invalid pretend that it is the same as the cluster label
        if(l1 >= m_pEnergy->numClasses())
            l1 = l2;

        // Count the number of pixels allocated to each cluster
        clusterSize[k]++;

        // Update feature
        Feature& f = outClusters[k].m_feature;
        Feature const& fPx = m_pClusterFeat->atSite(i);
        auto const sigmaInv = m_pEnergy->weights().feature(l1, l2).cwiseInverse().asDiagonal();
        auto const& w = m_pEnergy->weights().higherOrder(l1, l2);
        auto const wTail = w.segment(f.size(), f.size());

        f += fPx - 0.5f * sigmaInv * wTail;
    }
    // Normalize all cluster features
    for (ClusterId k = 0; k < outClusters.size(); ++k)
    {
        if(clusterSize[k] > 0)
            outClusters[k].m_feature /= clusterSize[k];
    }
}

template<typename EnergyFun>
void InferenceIterator<EnergyFun>::initialize(LabelImage& outLabeling, LabelImage& outClustering, std::vector<Cluster>& outClusters)
{
    PROFILE_THIS

    // Initialize labeling with background (all zeros)
    outLabeling = LabelImage(m_pPxFeat->width(), m_pPxFeat->height());

    // That's enough if no clusters are requested
    if(m_pEnergy->numClusters() == 0)
        return;

    // Initialize clustering with all zeros as well
    outClustering = LabelImage(m_pPxFeat->width(), m_pPxFeat->height());

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
    outClusters.back().m_feature = m_pClusterFeat->atSite(site);

    // Compute the distance between each pixel and the newly created cluster center
    for(SiteId i = 0; i < outLabeling.pixels(); ++i)
    {
        Feature const& f1 = m_pClusterFeat->atSite(i);
        Feature const& f2 = outClusters.back().m_feature;
        Label l1 = outLabeling.atSite(i);
        Label l2 = outClusters.back().m_label;
        Cost const dist = m_pEnergy->featureCost(f1, f2, l1, l2);
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
        outClusters.back().m_feature = m_pClusterFeat->atSite(site);
        // Recompute cluster distances
        for(SiteId i = 0; i < outLabeling.pixels(); ++i)
        {
            Feature const& f1 = m_pClusterFeat->atSite(i);
            Feature const& f2 = outClusters.back().m_feature;
            Label l1 = outLabeling.atSite(i);
            Label l2 = outClusters.back().m_label;
            Cost const dist = m_pEnergy->featureCost(f1, f2, l1, l2);
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
    PROFILE_THIS

    ClusterId const numClusters = m_pEnergy->numClusters();
    Label const numClasses = m_pEnergy->numClasses();
    SiteId const numPx = gt.pixels();

    if(numClusters == 0)
        return;

    std::vector<std::vector<Cost>> clusterCost(numClusters, std::vector<Cost>(numClasses, 0));

    // Every auxiliary node has just unary terms, however they are a sum of all allocated pixels
    for(SiteId i = 0; i < numPx; ++i)
    {
        Label const l = gt.atSite(i);
        ClusterId const k = clustering.atSite(i);
        Feature const& f = m_pClusterFeat->atSite(i);
        Feature const& fClus = outClusters[k].m_feature;

        for (Label lClus = 0; lClus < numClasses; ++lClus)
        {
            if(l < numClasses)
                clusterCost[k][lClus] += m_pEnergy->higherOrderCost(f, fClus, l, lClus) + m_pEnergy->higherOrderSpecialUnaryCost(i, lClus);
            else // If the ground truth label is invalid just pretend that cluster and pixel have the same label no matter what
                clusterCost[k][lClus] += m_pEnergy->higherOrderCost(f, fClus, lClus, lClus) + m_pEnergy->higherOrderSpecialUnaryCost(i, lClus);
        }
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
    PROFILE_THIS

    InferenceResult result;

    // Initialize variables
    initialize(result.labeling, result.clustering, result.clusters);

    // If no clusters are required, just do normal TRW-S
    if(m_pEnergy->numClusters() == 0)
    {
        updateLabels(result.labeling, result.clusters, result.clustering/*, &result.marginals*/);
        return result;
    }

    // Iterate until either convergence or the maximum number of iterations has been hit
    auto converged = [&](uint32_t iter, Cost last, Cost cur)
    {
        if(numIter > 0)
            return iter < numIter;
        else
        {
            Cost diff = last - cur;
            return diff <= m_eps;
        }
    };
    Cost energy =  m_pEnergy->giveEnergy(*m_pPxFeat, *m_pClusterFeat, result.labeling, result.clustering, result.clusters);
    Cost lastEnergy = std::numeric_limits<Cost>::max();
    uint32_t iter = 0;
    for (; iter < m_maxIter && !converged(iter, lastEnergy, energy); ++iter)
    {
        lastEnergy = energy;

        // Update clustering
        updateClusterAffiliation(result.clustering, result.labeling, result.clusters);

        // Update custer features
        updateClusterFeatures(result.clusters, result.labeling, result.clustering);

        // Update labels
        updateLabels(result.labeling, result.clusters, result.clustering/*, &result.marginals*/);

        // Compute current energy to check for convergence
        energy = m_pEnergy->giveEnergy(*m_pPxFeat, *m_pClusterFeat, result.labeling, result.clustering, result.clusters);
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

    // If no clusters are required, just do normal TRW-S
    if(m_pEnergy->numClusters() == 0)
    {
        Cost energy = m_pEnergy->giveEnergy(*m_pPxFeat, *m_pClusterFeat, labeling, clustering, clusters);
        result.energy.push_back(energy);

        FeatureImage marginals;
        updateLabels(labeling, clusters, clustering, &marginals);

        energy = m_pEnergy->giveEnergy(*m_pPxFeat, *m_pClusterFeat, labeling, clustering, clusters);

        result.labelings.push_back(labeling);
        result.marginals.push_back(marginals);
        result.energy.push_back(energy);
        return result;
    }

    // Iterate until either convergence or the maximum number of iterations has been hit
    auto converged = [&](uint32_t iter, Cost last, Cost cur)
    {
        if(numIter > 0)
            return iter < numIter;
        else
        {
            Cost diff = last - cur;
            return diff <= m_eps;
        }
    };
    Cost energy = m_pEnergy->giveEnergy(*m_pPxFeat, *m_pClusterFeat, labeling, clustering, clusters);
    result.energy.push_back(energy);
    Cost lastEnergy = std::numeric_limits<Cost>::max();
    uint32_t iter = 0;
    for (; iter < m_maxIter && !converged(iter, lastEnergy, energy); ++iter)
    {
        lastEnergy = energy;

        // Update clustering
        updateClusterAffiliation(clustering, labeling, clusters);

        // Update custer features
        updateClusterFeatures(clusters, labeling, clustering);

        // Update labels
        FeatureImage marginals;
        updateLabels(labeling, clusters, clustering, &marginals);

        // Compute current energy to check for convergence
        energy = m_pEnergy->giveEnergy(*m_pPxFeat, *m_pClusterFeat, labeling, clustering, clusters);

        // Store intermediate results
        result.clusters.push_back(clusters);
        result.clusterings.push_back(clustering);
        result.labelings.push_back(labeling);
        result.marginals.push_back(marginals);
        result.energy.push_back(energy);
    }

    result.numIter = iter;
    return result;
}

template<typename EnergyFun>
InferenceResult InferenceIterator<EnergyFun>::runOnGroundTruth(LabelImage const& gt, uint32_t numIter)
{
    InferenceResult result;

    // If no clusters are required, the ground truth is the result
    if(m_pEnergy->numClusters() == 0)
    {
        result.labeling = gt;
        return result;
    }

    // Initialize variables
    initialize(result.labeling, result.clustering, result.clusters);
    assert(gt.width() == result.labeling.width() && gt.height() == result.labeling.height());
    result.labeling = gt;

    // Iterate until either convergence or the maximum number of iterations has been hit
    auto converged = [&](uint32_t iter, Cost last, Cost cur)
    {
        if(numIter > 0)
            return iter < numIter;
        else
        {
            Cost diff = last - cur;
            return diff <= m_eps;
        }
    };
    Cost energy = m_pEnergy->giveEnergy(*m_pPxFeat, *m_pClusterFeat, result.labeling, result.clustering, result.clusters);
    Cost lastEnergy = std::numeric_limits<Cost>::max();
    uint32_t iter = 0;
    for (; iter < m_maxIter && !converged(iter, lastEnergy, energy); ++iter)
    {
        lastEnergy = energy;

        // Update clustering
        updateClusterAffiliation(result.clustering, result.labeling, result.clusters);

        // Update custer features
        updateClusterFeatures(result.clusters, result.labeling, result.clustering);

        // Update labels
        updateLabelsOnGroundTruth(result.labeling, result.clusters, result.clustering);

        // Compute current energy to check for convergence
        energy = m_pEnergy->giveEnergy(*m_pPxFeat, *m_pClusterFeat, result.labeling, result.clustering, result.clusters);
    }

    result.numIter = iter;
    return result;
}


#endif //HSEG_INFERENCEITERATOR_H
