//
// Created by jan on 29.08.16.
//

#include "Energy/LossAugmentedEnergyFunction.h"

LossAugmentedEnergyFunction::LossAugmentedEnergyFunction(UnaryFile const& unaries, WeightsVec const& weights,
                                                         float pairwiseSigmaSq, LabelImage const& groundTruth)
        : EnergyFunction(unaries, weights, pairwiseSigmaSq),
          m_groundTruth(groundTruth)
{
    m_lossFactor = 0;
    for(size_t i = 0; i < m_groundTruth.pixels(); ++i)
        if(m_groundTruth.atSite(i) < m_unaryScores.classes())
            m_lossFactor++;
    m_lossFactor = 1e8f / m_lossFactor;
}

float LossAugmentedEnergyFunction::unaryCost(size_t i, Label l) const
{
    float loss = 0;
    if (m_groundTruth.atSite(i) != l && m_groundTruth.atSite(i) < m_unaryScores.classes())
        loss = m_lossFactor;

    return -EnergyFunction::unaryCost(i, l) - loss + m_constant;
}

float LossAugmentedEnergyFunction::pairwiseClassWeight(Label l1, Label l2) const
{
    return -EnergyFunction::pairwiseClassWeight(l1, l2) + m_constant;
}

float LossAugmentedEnergyFunction::pairwisePixelWeight(CieLabImage const& img, size_t i, size_t j) const
{
    return -EnergyFunction::pairwisePixelWeight(img, i, j) + m_constant;
}

float LossAugmentedEnergyFunction::classDistance(Label l1, Label l2) const
{
    return -EnergyFunction::classDistance(l1, l2) + m_constant;
}

float LossAugmentedEnergyFunction::pixelToClusterDistance(Feature const& fPx, Label lPx, Cluster const& cl) const
{
    return -EnergyFunction::pixelToClusterDistance(fPx, lPx, cl) + m_constant;
}

