//
// Created by jan on 29.08.16.
//

#include "Energy/LossAugmentedEnergyFunction.h"

LossAugmentedEnergyFunction::LossAugmentedEnergyFunction(UnaryFile const& unaries, Weights const& weights,
                                                         Cost pairwiseSigmaSq, Matrix5 const& featureWeights,
                                                         LabelImage const& groundTruth)
        : EnergyFunction(unaries, weights, pairwiseSigmaSq, featureWeights),
          m_groundTruth(groundTruth)
{
    m_lossFactor = computeLossFactor(groundTruth, unaries.classes());
}

Cost LossAugmentedEnergyFunction::lossFactor()
{
    return m_lossFactor;
}

Cost
LossAugmentedEnergyFunction::computeLoss(LabelImage const& labeling, LabelImage const& superpixels,
                                        LabelImage const& groundTruth, Cost lossFactor, Label numClasses,
                                        std::vector<Cluster> const& clustering)
{
    Cost loss = 0;
    for (SiteId i = 0; i < labeling.pixels(); ++i)
    {
        if(groundTruth.atSite(i) < numClasses)
        {
            if (groundTruth.atSite(i) != labeling.atSite(i))
                loss += lossFactor;
            if (groundTruth.atSite(i) != clustering[superpixels.atSite(i)].label)
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
    lossFactor = 1e8f / lossFactor;
    return lossFactor;
}

