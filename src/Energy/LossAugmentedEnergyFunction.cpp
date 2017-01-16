//
// Created by jan on 29.08.16.
//

#include "Energy/LossAugmentedEnergyFunction.h"

LossAugmentedEnergyFunction::LossAugmentedEnergyFunction(Weights const* weights, LabelImage const* groundTruth)
        : EnergyFunction(weights),
          m_pGroundTruth(groundTruth)
{
    m_lossFactor = computeLossFactor(*groundTruth, weights->numClasses());
}

Cost LossAugmentedEnergyFunction::lossFactor()
{
    return m_lossFactor;
}

Cost
LossAugmentedEnergyFunction::computeLoss(LabelImage const& labeling, LabelImage const& groundTruth, Cost lossFactor, Label numClasses)
{
    Cost loss = 0;
    for (SiteId i = 0; i < labeling.pixels(); ++i)
    {
        if(groundTruth.atSite(i) < numClasses)
        {
            if (groundTruth.atSite(i) != labeling.atSite(i))
                loss += lossFactor;
        }
    }

    return loss;
}

Cost LossAugmentedEnergyFunction::computeLossFactor(LabelImage const& groundTruth, Label numClasses)
{
    Cost lossFactor = 0;
    for (SiteId i = 0; i < groundTruth.pixels(); ++i)
        if (groundTruth.atSite(i) < numClasses)
            lossFactor++;
    lossFactor = 1e5f / lossFactor;
    return lossFactor;
}

