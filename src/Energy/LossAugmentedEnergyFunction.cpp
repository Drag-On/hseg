//
// Created by jan on 29.08.16.
//

#include "Energy/LossAugmentedEnergyFunction.h"

LossAugmentedEnergyFunction::LossAugmentedEnergyFunction(Weights const* weights, LabelImage const* groundTruth, ClusterId numClusters, bool usePairwise, bool useClusterLoss)
        : EnergyFunction(weights, numClusters, usePairwise),
          m_pGroundTruth(groundTruth),
          m_useClusterLoss(useClusterLoss)
{
    m_lossFactor = computeLossFactor(*groundTruth, weights->numClasses());
}

Cost LossAugmentedEnergyFunction::giveEnergy(FeatureImage const& pxFeat, FeatureImage const& clusterFeat, LabelImage const& labeling,
                                             LabelImage const& clustering, std::vector<Cluster> const& clusters) const
{
    Cost normalCost = EnergyFunction::giveEnergy(pxFeat, clusterFeat, labeling, clustering, clusters);

    // Also account for the loss
    Cost loss = computeLoss(labeling, clustering, *m_pGroundTruth, clusters, m_lossFactor, numClasses(), m_useClusterLoss);

    return normalCost - loss;
}

Cost LossAugmentedEnergyFunction::lossFactor()
{
    return m_lossFactor;
}

Cost LossAugmentedEnergyFunction::computeLoss(LabelImage const& labeling, LabelImage const& clustering,
                                              LabelImage const& groundTruth, std::vector<Cluster> const& clusters,
                                              Cost lossFactor, Label numClasses, bool useClusterLoss)
{
    SiteId lossSites = 0;
    SiteId hoLossSites = 0;
    for (SiteId i = 0; i < labeling.pixels(); ++i)
    {
        if(groundTruth.atSite(i) < numClasses)
        {
            if (groundTruth.atSite(i) != labeling.atSite(i))
                lossSites++;
            if (useClusterLoss && clusters.size() > 0 && groundTruth.atSite(i) != clusters[clustering.atSite(i)].m_label)
                hoLossSites++;
        }
    }

    return (lossSites + hoLossSites) * lossFactor;
}

Cost LossAugmentedEnergyFunction::computeLossFactor(LabelImage const& groundTruth, Label numClasses)
{
    SiteId validSites = 0;
    for (SiteId i = 0; i < groundTruth.pixels(); ++i)
        if (groundTruth.atSite(i) < numClasses)
            validSites++;
    Cost lossFactor = 1e3f / validSites;
    return lossFactor;
}

