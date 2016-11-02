//
// Created by jan on 29.08.16.
//

#include "Energy/LossAugmentedEnergyFunction.h"

LossAugmentedEnergyFunction::LossAugmentedEnergyFunction(UnaryFile const& unaries, WeightsVec const& weights,
                                                         float pairwiseSigmaSq, Matrix5f const& featureWeights,
                                                         LabelImage const& groundTruth, LabelImage const& spGroundTruth)
        : EnergyFunction(unaries, weights, pairwiseSigmaSq, featureWeights),
          m_groundTruth(groundTruth),
          m_spGroundTruth(spGroundTruth)
{
    m_lossFactor = 0;
    for(size_t i = 0; i < m_groundTruth.pixels(); ++i)
        if(m_groundTruth.atSite(i) < m_unaryScores.classes())
            m_lossFactor++;
    m_lossFactor = 1e8f / m_lossFactor;
}

