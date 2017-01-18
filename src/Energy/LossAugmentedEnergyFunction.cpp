//
// Created by jan on 29.08.16.
//

#include "Energy/LossAugmentedEnergyFunction.h"

LossAugmentedEnergyFunction::LossAugmentedEnergyFunction(Weights const* weights, LabelImage const* groundTruth, ClusterId numClusters)
        : EnergyFunction(weights, numClusters),
          m_pGroundTruth(groundTruth)
{
    m_lossFactor = computeLossFactor(*groundTruth, weights->numClasses());
}

Cost LossAugmentedEnergyFunction::giveEnergy(FeatureImage const& features, LabelImage const& labeling, LabelImage const& clustering, std::vector<Cluster> const& clusters) const
{
    Cost normalCost = EnergyFunction::giveEnergy(features, labeling, clustering, clusters);

    // Also account for the loss
    Cost loss = computeLoss(labeling, *m_pGroundTruth, m_lossFactor, numClasses());

    return normalCost - loss;
}

Cost LossAugmentedEnergyFunction::lossFactor()
{
    return m_lossFactor;
}

Cost
LossAugmentedEnergyFunction::computeLoss(LabelImage const& labeling, LabelImage const& groundTruth, Cost lossFactor, Label numClasses)
{
    SiteId validSites = 0;
    for (SiteId i = 0; i < labeling.pixels(); ++i)
    {
        if(groundTruth.atSite(i) < numClasses)
        {
            if (groundTruth.atSite(i) != labeling.atSite(i))
                validSites++;
        }
    }

    return validSites * lossFactor;
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

