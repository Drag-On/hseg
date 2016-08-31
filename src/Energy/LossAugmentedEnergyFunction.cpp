//
// Created by jan on 29.08.16.
//

#include "Energy/LossAugmentedEnergyFunction.h"

LossAugmentedEnergyFunction::LossAugmentedEnergyFunction(UnaryFile const& unaries, WeightsVec const& weights,
                                                         float pairwiseSigmaSq, LabelImage const& groundTruth)
        : EnergyFunction(unaries, weights, pairwiseSigmaSq),
          m_groundTruth(groundTruth)
{
    for(size_t i = 0; i < m_groundTruth.pixels(); ++i)
        if(m_groundTruth.atSite(i) < m_unaryScores.classes())
            m_lossFactor++;
    m_lossFactor = 1 / m_lossFactor;
}

float LossAugmentedEnergyFunction::unaryCost(size_t i, Label l) const
{
    float loss = 0;
    if (m_groundTruth.atSite(i) != l && m_groundTruth.atSite(i) < m_unaryScores.classes())
        loss = m_lossFactor;

    return EnergyFunction::unaryCost(i, l) - loss;
}

