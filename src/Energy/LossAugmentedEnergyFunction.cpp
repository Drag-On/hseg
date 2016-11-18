//
// Created by jan on 29.08.16.
//

#include "Energy/LossAugmentedEnergyFunction.h"

LossAugmentedEnergyFunction::LossAugmentedEnergyFunction(UnaryFile const& unaries, WeightsVec const& weights,
                                                         float pairwiseSigmaSq, Matrix5f const& featureWeights,
                                                         LabelImage const& groundTruth)
        : EnergyFunction(unaries, weights, pairwiseSigmaSq, featureWeights),
          m_groundTruth(groundTruth)
{
    m_lossFactor = computeLossFactor(groundTruth, unaries.classes());
}

float LossAugmentedEnergyFunction::lossFactor()
{
    return m_lossFactor;
}

float
LossAugmentedEnergyFunction::computeLoss(LabelImage const& labeling, LabelImage const& groundTruth, float lossFactor,
                                         size_t numClasses)
{
    float loss = 0;
    for (size_t i = 0; i < labeling.pixels(); ++i)
        if (groundTruth.atSite(i) != labeling.atSite(i) && groundTruth.atSite(i) < numClasses)
            loss += lossFactor;
    return loss;
}

float LossAugmentedEnergyFunction::computeLossFactor(LabelImage const& groundTruth, size_t numClasses)
{
    float lossFactor = 0;
    for (size_t i = 0; i < groundTruth.pixels(); ++i)
        if (groundTruth.atSite(i) < numClasses)
            lossFactor++;
    lossFactor = 1e8f / lossFactor;
    return lossFactor;
}

